# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np
import logging
import json
import pandas as pd
import os
import time

from models.sequential.SASRec import SASRec_SAE, TRAIN_MODE, INFERENCE_MODE, TEST_MODE
from utils import layers

class SAE_Attention(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--sae_attn_heads', type=int, default=4,
                            help='Number of attention heads in SAE')
        parser.add_argument('--sae_attn_dropout', type=float, default=0.1,
                            help='Dropout rate for attention mechanism')
        return parser
    
    def __init__(self, args, d_in):
        super(SAE_Attention, self).__init__()
        self.k = args.sae_k
        self.scale_size = args.sae_scale_size
        self.device = args.device
        self.dtype = torch.float32
        self.attn_heads = args.sae_attn_heads
        self.dropout = args.sae_attn_dropout
        
        self.d_in = d_in
        self.hidden_dim = d_in * self.scale_size
        
        # Encoder
        self.encoder = nn.Linear(self.d_in, self.hidden_dim, device=self.device, dtype=self.dtype)
        self.encoder.bias.data.zero_()
        
        # Decoder with attention
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        self.set_decoder_norm_to_unit_norm()
        self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype=self.dtype, device=self.device))
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_in,
            num_heads=self.attn_heads,
            batch_first=True
        )
        self.attn_dropout = nn.Dropout(self.dropout)
        
        self.activate_latents = set()
        self.previous_activate_latents = None
        self.epoch_activations = {"indices": None, "values": None}
        return
    
    def get_dead_latent_ratio(self, need_update=0):
        ans = 1 - len(self.activate_latents)/self.hidden_dim
        # only update training situation for auxk_loss
        if need_update:
            self.previous_activate_latents = torch.tensor(list(self.activate_latents)).to(self.device)
        self.activate_latents = set()
        return ans
    
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps
    
    def topk_activation(self, x, save_result):
        topk_values, topk_indices = torch.topk(x, self.k, dim=1)
        self.activate_latents.update(topk_indices.cpu().numpy().flatten())
        
        if save_result:
            if self.epoch_activations["indices"] is None:
                self.epoch_activations["indices"] = topk_indices.detach().cpu().numpy()
                self.epoch_activations["values"] = topk_values.detach().cpu().numpy()
            else:
                self.epoch_activations["indices"] = np.concatenate((self.epoch_activations["indices"], topk_indices.detach().cpu().numpy()), axis=0)
                self.epoch_activations["values"] = np.concatenate((self.epoch_activations["values"], topk_values.detach().cpu().numpy()), axis=0)
        
        sparse_x = torch.zeros_like(x)
        sparse_x.scatter_(1, topk_indices, topk_values.to(self.dtype))
        return sparse_x
    
    def forward(self, x, train_mode=False, save_result=False):
        batch_size = x.shape[0]
        sae_in = x - self.b_dec
        pre_acts = nn.functional.relu(self.encoder(sae_in))
        z = self.topk_activation(pre_acts, save_result=save_result)
        
        # Initial reconstruction
        x_recon_initial = z @ self.W_dec + self.b_dec
        
        # Apply attention between original input and reconstructed output
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, d_in]
        x_recon_expanded = x_recon_initial.unsqueeze(1)  # [batch_size, 1, d_in]
        
        # Concatenate for self-attention
        combined = torch.cat([x_expanded, x_recon_expanded], dim=1)  # [batch_size, 2, d_in]
        
        # Apply attention using PyTorch's MultiheadAttention
        attn_output, _ = self.attention(combined, combined, combined)
        
        # Apply dropout
        attn_output = self.attn_dropout(attn_output)
        
        # Extract enhanced reconstruction (second position)
        x_reconstructed = attn_output[:, 1]
        
        # Calculate loss metrics
        e = x_reconstructed - x
        total_variance = (x - x.mean(0)).pow(2).sum()
        self.fvu = e.pow(2).sum() / total_variance
        
        # Handle auxiliary loss for dead latents
        if train_mode:
            # First epoch, do not have dead latent info
            if (self.previous_activate_latents) is None:
                self.auxk_loss = 0.0
                return x_reconstructed
            num_dead = self.hidden_dim - len(self.previous_activate_latents)
            k_aux = x.shape[-1] // 2
            if num_dead == 0:
                self.auxk_loss = 0.0
                return x_reconstructed
                
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            dead_mask = torch.isin(torch.arange(pre_acts.shape[-1]).to(self.device), 
                                  self.previous_activate_latents, invert=True)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = torch.zeros_like(auxk_latents)
            e_hat.scatter_(1, auxk_indices, auxk_acts.to(self.dtype))
            e_hat = e_hat @ self.W_dec + self.b_dec
            
            auxk_loss = (e_hat - e).pow(2).sum()
            self.auxk_loss = scale * auxk_loss / total_variance
            
        return x_reconstructed


class SASRec_SAE_Attention(SASRec_SAE):
    reader = 'SeqReader'
    runner = 'RecSAERunner'
    sae_extra_params = ['sae_lr', 'sae_k', 'sae_scale_size', 'sae_attn_heads', 'sae_attn_dropout']
    
    @staticmethod
    def parse_model_args(parser):
        parser = SAE_Attention.parse_model_args(parser)
        parser = SASRec_SAE.parse_model_args(parser)
        return parser
    
    def __init__(self, args, corpus):
        SASRec_SAE.__init__(self, args, corpus)
        # Replace standard SAE with attention-enhanced SAE
        self.sae_module = SAE_Attention(args, self.emb_size)
        return

    def save_model(self, model_path=None):
        """Save the model state to the specified path"""
        if model_path is None:
            model_path = self.model_path
            
        try:
            # Check if path is absolute
            if not os.path.isabs(model_path):
                # Create absolute path relative to current directory
                abs_path = os.path.abspath(model_path)
                logging.info(f"Converting relative path {model_path} to absolute path {abs_path}")
                model_path = abs_path
                
            # Check if path is too long for Windows (>260 characters)
            if len(model_path) > 240:
                # If path is too long, create a shorter filename
                dir_path = os.path.dirname(model_path)
                short_name = f"model_{int(time.time())}.pt"
                model_path = os.path.join(dir_path, short_name)
                logging.warning(f"Path too long. Using shorter name: {model_path}")
            
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(model_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
                logging.info(f"Created directory: {dir_path}")
                
            # Check access rights by trying to create a test file
            test_file = os.path.join(dir_path, "test_access.txt")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                logging.info(f"Directory access check passed: {dir_path}")
            except Exception as access_e:
                logging.error(f"Failed directory access check: {str(access_e)}")
                # Use current working directory as fallback
                model_path = os.path.join(os.getcwd(), f"model_{int(time.time())}.pt")
                logging.warning(f"Using current directory for save: {model_path}")
            
            # Direct save without using temporary file
            logging.info(f"Saving model directly to {model_path}")
            torch.save(self.state_dict(), model_path)
            logging.info(f"Model successfully saved to {model_path}")
                
        except Exception as e:
            logging.error(f"Error saving model to {model_path}: {str(e)}")
            
            # Try to save with minimal path
            try:
                simple_path = f"model_backup_{int(time.time())}.pt"
                logging.warning(f"Trying to save with minimal path: {simple_path}")
                torch.save(self.state_dict(), simple_path)
                logging.info(f"Model saved to simple path: {simple_path}")
            except Exception as backup_e:
                logging.error(f"Failed even simple save: {str(backup_e)}")
                
            # Last attempt - try using pickle instead of torch.save
            try:
                import pickle
                pickle_path = f"model_pickle_backup_{int(time.time())}.pkl"
                with open(pickle_path, 'wb') as f:
                    pickle.dump(self.state_dict(), f)
                logging.info(f"Model saved using pickle to: {pickle_path}")
            except Exception as pickle_e:
                logging.error(f"All save attempts failed: {str(pickle_e)}")
                
    def load_model(self, model_path=None):
        """Loads model from the specified path with error handling"""
        if model_path is None:
            model_path = self.model_path
            
        try:
            # Check if path is absolute
            if not os.path.isabs(model_path):
                abs_path = os.path.abspath(model_path)
                logging.info(f"Converting relative path {model_path} to absolute path {abs_path}")
                model_path = abs_path
            
            # Check if file exists
            if not os.path.exists(model_path):
                logging.error(f"Model file not found at {model_path}")
                
                # Try to find model file in current directory
                file_name = os.path.basename(model_path)
                local_path = os.path.join(os.getcwd(), file_name)
                if os.path.exists(local_path):
                    logging.info(f"Found model file in current directory: {local_path}")
                    model_path = local_path
                else:
                    # Try to find any .pt file with similar name
                    import glob
                    pattern = os.path.join(os.path.dirname(model_path), "*.pt")
                    matching_files = glob.glob(pattern)
                    if matching_files:
                        model_path = matching_files[0]
                        logging.info(f"Using alternative model file: {model_path}")
                    else:
                        # Last attempt - search for files in current directory
                        local_files = glob.glob("*.pt") + glob.glob("model_*.pt") + glob.glob("*.pkl")
                        if local_files:
                            model_path = local_files[0]
                            logging.info(f"Using local file: {model_path}")
                        else:
                            return False
                
            # Load model with exception handling
            try:
                if model_path.endswith('.pkl'):
                    # Try to load from pickle
                    import pickle
                    with open(model_path, 'rb') as f:
                        state_dict = pickle.load(f)
                else:
                    # Standard torch loading
                    state_dict = torch.load(model_path, map_location=self.device)
                    
                if model_path == self.model_path:
                    self.load_state_dict(state_dict, strict=False)
                    for name, param in self.named_parameters():
                        if name in state_dict:
                            param.requires_grad = False
                else:
                    self.load_state_dict(state_dict)
                    
                logging.info(f'Successfully loaded model from {model_path}')
                return True
                
            except Exception as load_e:
                logging.error(f"Error in loading step: {str(load_e)}")
                return False
            
        except Exception as e:
            logging.error(f"Error in load_model: {str(e)}")
            return False 
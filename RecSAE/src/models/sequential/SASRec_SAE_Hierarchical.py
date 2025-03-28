# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np
import logging
import json
import pandas as pd
import os
import hashlib

from models.sequential.SASRec import SASRec_SAE, TRAIN_MODE, INFERENCE_MODE, TEST_MODE
from utils import layers

class SAE_Hierarchical(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--hier_levels', type=int, default=2,
                            help='Number of hierarchical levels')
        parser.add_argument('--hier_decay', type=float, default=0.5,
                            help='Decay factor for sparser levels')
        parser.add_argument('--hier_k_ratio', type=float, default=0.5,
                            help='Ratio to reduce k for higher levels')
        return parser
    
    def __init__(self, args, d_in):
        super(SAE_Hierarchical, self).__init__()
        self.k = args.sae_k
        self.scale_size = args.sae_scale_size
        self.device = args.device
        self.dtype = torch.float32
        self.hier_levels = args.hier_levels
        self.hier_decay = args.hier_decay
        self.hier_k_ratio = args.hier_k_ratio
        
        self.d_in = d_in
        self.hidden_dim = d_in * self.scale_size
        
        # Base encoder and decoder
        self.encoder = nn.Linear(self.d_in, self.hidden_dim, device=self.device, dtype=self.dtype)
        self.encoder.bias.data.zero_()
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        self.set_decoder_norm_to_unit_norm()
        self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype=self.dtype, device=self.device))
        
        # Hierarchical layers
        self.hier_encoders = nn.ModuleList()
        self.hier_decoders_W = nn.ParameterList()
        self.hier_decoders_b = nn.ParameterList()
        
        prev_dim = self.d_in
        for i in range(self.hier_levels - 1):
            # Each level reduces dimension and has fewer active units
            level_dim = int(self.hidden_dim * (self.hier_decay ** (i + 1)))
            encoder = nn.Linear(prev_dim, level_dim, device=self.device, dtype=self.dtype)
            encoder.bias.data.zero_()
            self.hier_encoders.append(encoder)
            
            # Decoder for this level
            W_dec = nn.Parameter(encoder.weight.data.clone())
            self.hier_decoders_W.append(W_dec)
            self.hier_decoders_b.append(nn.Parameter(torch.zeros(prev_dim, dtype=self.dtype, device=self.device)))
            
            prev_dim = level_dim
        
        # Initialize trackers
        self.activate_latents = set()
        self.previous_activate_latents = None
        self.epoch_activations = {"indices": None, "values": None}
        return
    
    def get_dead_latent_ratio(self, need_update=0):
        ans = 1 - len(self.activate_latents)/self.hidden_dim
        if need_update:
            self.previous_activate_latents = torch.tensor(list(self.activate_latents)).to(self.device)
        self.activate_latents = set()
        return ans
    
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps
        
        # Also normalize hierarchical decoder weights
        for W in self.hier_decoders_W:
            norm = torch.norm(W.data, dim=1, keepdim=True)
            W.data /= norm + eps
    
    def topk_activation(self, x, k, save_result=False):
        """Apply top-k sparsity activation"""
        topk_values, topk_indices = torch.topk(x, k, dim=1)
        
        if save_result and x.shape[1] == self.hidden_dim:  # Only track base-level activations
            self.activate_latents.update(topk_indices.cpu().numpy().flatten())
            
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
        
        # Base level reconstruction
        sae_in = x - self.b_dec
        pre_acts = nn.functional.relu(self.encoder(sae_in))
        z_base = self.topk_activation(pre_acts, self.k, save_result=save_result)
        x_recon_base = z_base @ self.W_dec + self.b_dec
        
        # Hierarchical reconstructions
        recons = [x_recon_base]
        curr_input = x_recon_base
        
        for i in range(self.hier_levels - 1):
            # Each level gets a smaller k for increased sparsity
            level_k = max(1, int(self.k * (self.hier_k_ratio ** (i + 1))))
            
            # Encode and apply top-k
            pre_acts = nn.functional.relu(self.hier_encoders[i](curr_input))
            z_level = self.topk_activation(pre_acts, level_k, save_result=False)
            
            # Reconstruct
            x_recon_level = z_level @ self.hier_decoders_W[i] + self.hier_decoders_b[i]
            recons.append(x_recon_level)
            
            # Input for next level
            curr_input = x_recon_level
        
        # Combine reconstructions with decaying weights
        weights = torch.tensor([self.hier_decay ** i for i in range(len(recons))]).to(self.device)
        weights = weights / weights.sum()  # Normalize weights
        
        x_reconstructed = torch.zeros_like(x)
        for i, recon in enumerate(recons):
            x_reconstructed += weights[i] * recon
        
        # Calculate reconstruction loss
        e = x_reconstructed - x
        total_variance = (x - x.mean(0)).pow(2).sum()
        self.fvu = e.pow(2).sum() / total_variance
        
        # Handle auxiliary loss for dead latents in training mode
        if train_mode:
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


class SASRec_SAE_Hierarchical(SASRec_SAE):
    reader = 'SeqReader'
    runner = 'RecSAERunner'
    sae_extra_params = ['sae_lr', 'sae_k', 'sae_scale_size', 'hier_levels', 'hier_decay', 'hier_k_ratio']
    
    @staticmethod
    def parse_model_args(parser):
        parser = SAE_Hierarchical.parse_model_args(parser)
        parser = SASRec_SAE.parse_model_args(parser)
        return parser
    
    def __init__(self, args, corpus):
        SASRec_SAE.__init__(self, args, corpus)
        # Replace standard SAE with hierarchical SAE
        self.sae_module = SAE_Hierarchical(args, self.emb_size)
        return 
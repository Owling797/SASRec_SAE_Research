# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import json
import pandas as pd
import os
import hashlib

from models.sequential.SASRec import SASRec_SAE, TRAIN_MODE, INFERENCE_MODE, TEST_MODE
from utils import layers

class SAE_Contrastive(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--contrastive_temp', type=float, default=0.07,
                            help='Temperature parameter for contrastive learning')
        parser.add_argument('--contrastive_weight', type=float, default=0.2,
                            help='Weight for contrastive learning loss')
        parser.add_argument('--augmentation_dropout', type=float, default=0.1,
                            help='Dropout rate for augmenting embeddings in contrastive learning')
        return parser
    
    def __init__(self, args, d_in):
        super(SAE_Contrastive, self).__init__()
        self.k = args.sae_k
        self.scale_size = args.sae_scale_size
        self.device = args.device
        self.dtype = torch.float32
        self.temp = args.contrastive_temp
        self.contrastive_weight = args.contrastive_weight
        self.augmentation_dropout = args.augmentation_dropout
        
        self.d_in = d_in
        self.hidden_dim = d_in * self.scale_size
        
        # Encoder
        self.encoder = nn.Linear(self.d_in, self.hidden_dim, device=self.device, dtype=self.dtype)
        self.encoder.bias.data.zero_()
        
        # Decoder 
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        self.set_decoder_norm_to_unit_norm()
        self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype=self.dtype, device=self.device))
        
        # Projector for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(self.d_in, self.d_in * 2, device=self.device),
            nn.ReLU(),
            nn.Linear(self.d_in * 2, self.d_in, device=self.device)
        )
        
        self.activate_latents = set()
        self.previous_activate_latents = None
        self.epoch_activations = {"indices": None, "values": None}
        
        # Track contrastive loss separately
        self.contrastive_loss_val = 0.0
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
    
    def create_augmentation(self, x):
        """Create an augmented version of the embedding for contrastive learning"""
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.augmentation_dropout)).to(self.device)
        return x * mask
    
    def contrastive_loss(self, z1, z2, batch_size):
        """Calculate NT-Xent loss for contrastive learning"""
        # Normalize the projections
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Gather representations from all GPUs if using distributed training
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temp
        
        # Mask out self-similarity
        sim_i_j = torch.diag(similarity_matrix, batch_size)
        sim_j_i = torch.diag(similarity_matrix, -batch_size)
        
        # Create labels for positive pairs
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0)
        
        # Mask out the positive samples from similarity matrix
        mask = torch.ones_like(similarity_matrix) - torch.eye(2 * batch_size).to(self.device)
        negative_samples = mask * similarity_matrix
        
        # Create labels for InfoNCE loss: positives are ones, negatives are all zeros
        labels = torch.zeros(2 * batch_size).to(self.device).long()
        
        # Combine positive and negative samples for the loss
        logits = torch.cat([positive_samples.unsqueeze(1), negative_samples], dim=1)
        
        # Calculate cross entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def forward(self, x, train_mode=False, save_result=False):
        batch_size = x.shape[0]
        sae_in = x - self.b_dec
        pre_acts = nn.functional.relu(self.encoder(sae_in))
        z = self.topk_activation(pre_acts, save_result=save_result)
        
        # Standard reconstruction
        x_reconstructed = z @ self.W_dec + self.b_dec
        
        # Calculate reconstruction loss
        e = x_reconstructed - x
        total_variance = (x - x.mean(0)).pow(2).sum()
        self.fvu = e.pow(2).sum() / total_variance
        
        # Handle contrastive learning if in training mode
        if train_mode:
            # Generate augmentations
            x_aug1 = self.create_augmentation(x)
            x_aug2 = self.create_augmentation(x)
            
            # Project reconstructions for contrastive loss
            proj1 = self.projector(x_reconstructed)
            
            # Process augmented versions
            aug1_pre_acts = nn.functional.relu(self.encoder(x_aug1 - self.b_dec))
            aug1_z = self.topk_activation(aug1_pre_acts, save_result=False)
            aug1_recon = aug1_z @ self.W_dec + self.b_dec
            proj2 = self.projector(aug1_recon)
            
            # Calculate contrastive loss
            self.contrastive_loss_val = self.contrastive_loss(proj1, proj2, batch_size)
            
            # Handle auxiliary loss for dead latents
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
            
            # Combine deadhead loss with contrastive loss
            auxk_loss = (e_hat - e).pow(2).sum()
            self.auxk_loss = scale * auxk_loss / total_variance + self.contrastive_weight * self.contrastive_loss_val
            
        return x_reconstructed


class SASRec_SAE_Contrastive(SASRec_SAE):
    reader = 'SeqReader'
    runner = 'RecSAERunner'
    sae_extra_params = ['sae_lr', 'sae_k', 'sae_scale_size', 'contrastive_temp', 
                        'contrastive_weight', 'augmentation_dropout']
    
    @staticmethod
    def parse_model_args(parser):
        parser = SAE_Contrastive.parse_model_args(parser)
        parser = SASRec_SAE.parse_model_args(parser)
        return parser
    
    def __init__(self, args, corpus):
        SASRec_SAE.__init__(self, args, corpus)
        # Replace standard SAE with contrastive learning enhanced SAE
        self.sae_module = SAE_Contrastive(args, self.emb_size)
        return 
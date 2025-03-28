# RecSAE: Sparse Autoencoder Enhanced Sequential Recommendation

This repository contains implementations of Sparse Autoencoder (SAE) enhanced sequential recommendation models based on the SASRec architecture. The code is built upon the [ReChorus](https://github.com/THUwangcy/ReChorus) recommendation framework, extending it with sparse autoencoder capabilities and various enhancements.

## Overview

SAEs are used to learn sparse representations of user preferences through sequential recommendation models. The RecSAE project includes several variants:

1. **SASRec**: The base Self-Attentive Sequential Recommendation model
2. **SASRec_SAE**: SASRec enhanced with a standard Sparse Autoencoder
3. **SASRec_SAE_Attention**: SAE enhanced with an attention mechanism for improved reconstruction
4. **SASRec_SAE_Contrastive**: SAE enhanced with contrastive learning techniques

## Experiments

The following experiments have been implemented:

### Attention-Enhanced SAE

The SASRec_SAE_Attention model adds a multi-head attention mechanism after the basic reconstruction step of the SAE. This enables the model to refine its reconstruction by attending to important features in the input vectors.

Key components:
- Multi-head self-attention layer between input and reconstructed vectors
- Enhanced reconstruction quality through attention-based refinement
- Configurable attention parameters (number of heads, dropout rate)

### Contrastive Learning SAE

The SASRec_SAE_Contrastive model incorporates contrastive learning techniques to enhance representation learning. This approach helps the model learn more discriminative features by contrasting similar and dissimilar examples.

Key components:
- Contrastive loss function for representation learning
- Projection head for contrastive embedding space
- Temperature-scaled similarity scoring

## Running the Models

### Prerequisites

- Python 3.7+
- PyTorch 1.10+
- Other dependencies as listed in requirements.txt

### Basic Commands

#### Running SASRec (Base Model)

```bash
python -m RecSAE.src.main \
  --model_name SASRec \
  --dataset [YOUR_DATASET] \
  --emb_size 64 \
  --num_layers 2 \
  --num_heads 4 \
  --dropout 0.2 \
  --lr 1e-3
```

#### Running SASRec_SAE

```bash
# First train the base SASRec model
python -m RecSAE.src.main \
  --model_name SASRec \
  --dataset [YOUR_DATASET] \
  --emb_size 64

# Then train the SAE on top of it
python -m RecSAE.src.main_sae \
  --model_name SASRec_SAE \
  --dataset [YOUR_DATASET] \
  --emb_size 64 \
  --sae_k 5 \
  --sae_scale_size 2 \
  --sae_lr 1e-4 \
  --sae_train 1
```

#### Running SASRec_SAE_Attention

```bash
python -m RecSAE.src.main_sae_attention \
  --model_name SASRec_SAE_Attention \
  --dataset [YOUR_DATASET] \
  --emb_size 64 \
  --sae_k 5 \
  --sae_scale_size 2 \
  --sae_lr 1e-4 \
  --sae_attn_heads 4 \
  --sae_attn_dropout 0.1 \
  --sae_train 1
```

#### Running SASRec_SAE_Contrastive

```bash
python -m RecSAE.src.main_sae_contrastive \
  --model_name SASRec_SAE_Contrastive \
  --dataset [YOUR_DATASET] \
  --emb_size 64 \
  --sae_k 5 \
  --sae_scale_size 2 \
  --sae_lr 1e-4 \
  --sae_contrastive_temp 0.07 \
  --sae_contrastive_weight 0.5 \
  --sae_train 1
```

### Important Parameters

- `--sae_k`: Number of top-k activations to keep in the sparse encoding
- `--sae_scale_size`: Scale factor for hidden layer size relative to input size
- `--sae_lr`: Learning rate for the SAE module
- `--sae_train`: Set to 1 for training mode, 0 for evaluation mode
- `--sae_attn_heads`: Number of attention heads (for Attention variant)
- `--sae_attn_dropout`: Dropout rate for attention layers (for Attention variant)
- `--sae_contrastive_temp`: Temperature parameter for contrastive loss (for Contrastive variant)
- `--sae_contrastive_weight`: Weight of contrastive loss in the overall loss function (for Contrastive variant)

## Results

The enhanced SAE models have shown improvements over the base SASRec model:

- SASRec_SAE_Attention: Improved reconstruction quality and recommendation performance, especially for sparse user histories
- SASRec_SAE_Contrastive: Better representation learning and cold-start performance

For detailed results, please refer to the experimental logs in the `log` directory.

## Acknowledgements

This implementation is based on the [ReChorus](https://github.com/THUwangcy/ReChorus) framework, a modular and highly-customizable recommendation library. We extend our thanks to the original authors for their excellent work. 
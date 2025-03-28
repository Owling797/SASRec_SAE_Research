# RecSAE

## Interpret Recommendation Models with Sparse Autoencoder

We use the [ReChorus](https://github.com/THUwangcy/ReChorus) framework as our code base and implement the SAE module upon it.

### Command

```bash

cd src

python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1

python main_sae.py  --epoch 50 --sae_lr 5e-4 --batch_size 8 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1

python main_sae.py  --epoch 50 --sae_lr 5e-4 --batch_size 8 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 0

cd analysis

python 0_analysis_trainLog.py

```

### Citations

```
@inproceedings{wang2024SAE,
	title={Interpret the Internal States of Recommendation Model with Sparse Autoencoder},
	author={Jiayin Wang and Xiaoyu Zhang and Weizhi Ma and Min Zhang},
	year={2024},
	journal={arXiv preprint arXiv:2411.06112}
}
```

### Contact

JiayinWangTHU AT gmail.com

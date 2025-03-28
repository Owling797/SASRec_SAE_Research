# python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --load 1 --train 0

# Early stop at 75 based on dev result.

# Best Iter(dev)=   66     dev=(HR@5:0.0627,NDCG@5:0.0408) [722.9 s] 
# Load model from ../model/SASRec/SASRec__Grocery_and_Gourmet_Food__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1.pt
                                                                                                    
# Dev  After Training: (HR@5:0.0627,NDCG@5:0.0408,HR@10:0.0924,NDCG@10:0.0504,HR@20:0.1330,NDCG@20:0.0606,HR@50:0.2127,NDCG@50:0.0763)
                                                                                                    
# Test After Training: (HR@5:0.0454,NDCG@5:0.0295,HR@10:0.0678,NDCG@10:0.0367,HR@20:0.1046,NDCG@20:0.0460,HR@50:0.1745,NDCG@50:0.0598)

# python main_sae.py  --epoch 50 --sae_lr 1e-4 --batch_size 16 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 0


# python main_sae.py  --epoch 50 --sae_lr 3e-4 --batch_size 64 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1

# python main_sae.py  --epoch 50 --sae_lr 1e-4 --batch_size 64 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1

python main_sae.py  --epoch 50 --sae_lr 5e-4 --batch_size 8 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1

python main_sae.py  --epoch 50 --sae_lr 1e-3 --batch_size 32 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1

python main_sae.py  --epoch 50 --sae_lr 1e-3 --batch_size 16 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1


# python main_sae.py  --epoch 50 --sae_lr 5e-4 --batch_size 32 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1

# python main_sae.py  --epoch 50 --sae_lr 1e-3 --batch_size 256 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1


# python main_sae.py  --epoch 50 --sae_lr 3e-4 --batch_size 32 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1
# python main_sae.py  --epoch 50 --sae_lr 3e-4 --batch_size 16 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1
# python main_sae.py  --epoch 50 --sae_lr 3e-4 --batch_size 8 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1


# python main_sae.py  --epoch 50 --sae_lr 5e-5 --batch_size 32 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1
# python main_sae.py  --epoch 50 --sae_lr 5e-5 --batch_size 16 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1
# python main_sae.py  --epoch 50 --sae_lr 5e-5 --batch_size 8 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --test_all 1 --sae_train 1


# python main_sae.py  --epoch 50 --sae_lr 5e-5 --batch_size 16 --sae_k 1 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food' --path '../data' --load 1 --train 1
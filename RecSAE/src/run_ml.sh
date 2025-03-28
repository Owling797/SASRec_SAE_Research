# python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --gpu 1


# Test Before Training: (HR@5:0.0028,NDCG@5:0.0015,HR@10:0.0038,NDCG@10:0.0019,HR@20:0.0066,NDCG@20:0.0025,HR@50:0.0153,NDCG@50:0.0042)                                                                                      
# Load model from ../model/SASRec/SASRec__ML_1MTOPK__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1.pt
                                                                                                    
# Dev  After Training: (HR@5:0.0851,NDCG@5:0.0564,HR@10:0.1280,NDCG@10:0.0702,HR@20:0.1971,NDCG@20:0.0875,HR@50:0.3267,NDCG@50:0.1130)
                                                                                                    
# Test After Training: (HR@5:0.1061,NDCG@5:0.0704,HR@10:0.1611,NDCG@10:0.0882,HR@20:0.2321,NDCG@20:0.1062,HR@50:0.3500,NDCG@50:0.1296)

python main_sae.py  --epoch 50 --sae_lr 5e-5 --batch_size 16 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1


python main_sae.py  --epoch 50 --sae_lr 1e-4 --batch_size 16 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
python main_sae.py  --epoch 50 --sae_lr 1e-4 --batch_size 32 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
python main_sae.py  --epoch 50 --sae_lr 1e-4 --batch_size 8 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1


python main_sae.py  --epoch 50 --sae_lr 3e-4 --batch_size 16 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
python main_sae.py  --epoch 50 --sae_lr 3e-4 --batch_size 32 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
python main_sae.py  --epoch 50 --sae_lr 3e-4 --batch_size 8 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1


python main_sae.py  --epoch 50 --sae_lr 5e-5 --batch_size 32 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
python main_sae.py  --epoch 50 --sae_lr 5e-5 --batch_size 8 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1




python main_sae.py  --epoch 50 --sae_lr 1e-5 --batch_size 16 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
python main_sae.py  --epoch 50 --sae_lr 1e-5 --batch_size 32 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1
python main_sae.py  --epoch 50 --sae_lr 1e-5 --batch_size 8 --sae_k 32 --sae_scale_size 32 --model_name SASRec_SAE --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'ML_1MTOPK' --path '../data/MovieLens_1M/' --test_all 1 --sae_train 1


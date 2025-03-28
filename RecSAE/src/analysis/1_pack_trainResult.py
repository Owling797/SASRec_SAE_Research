

if __name__ == "__main__":
	dataset_name = "Grocery_and_Gourmet_Food"
	sae_lr = "0.0005"
	batch_size = 32
	sae_k = 32
	sae_scale_size = 32
	file_name = f"../../log/SASRec_SAE/SASRec_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__batch_size={batch_size}.txt"
	

	activation_file = f"../../model/SASRec_SAE/SASRec_SAE__Grocery_and_Gourmet_Food__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr=0.0001__sae_k=32__sae_scale_size=32__batch_size=16.csv"

	prediction_file = f"../../log/SASRec_SAE/SASRec_SAE__Grocery_and_Gourmet_Food__0__lr=0/rec-SASRec_SAE-test.csv"
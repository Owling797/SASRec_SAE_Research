import re
import pandas as pd
import matplotlib.pyplot as plt


def analysis(batch_size,sae_lr):

	dataset_name = "Grocery_and_Gourmet_Food"
	
	# sae_lr = sae_lr_list[4]
	# batch_size = bath_size_list[3]
	sae_k = 32
	sae_scale_size = 32
	file_name = f"../../log/SASRec_SAE/SASRec_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__batch_size={batch_size}.txt"
	if not os.path.exists(file_name):
		return None, None
	
	with open(file_name, 'r') as file:
		log_data = file.readlines()

	start_index = None
	for i in reversed(range(len(log_data))):
		if log_data[i].startswith("INFO:root:Namespace(model_name='SASRec_SAE', model_mode='')"):
			start_index = i
			break

	if start_index is not None:
		epoch_pattern = re.compile(r'Epoch\s+(\d+)')
		loss_pattern = re.compile(r'loss=([\d.]+)')
		dead_latent_pattern = re.compile(r'dead_latent=([\d.]+)')
		metrics_pattern = re.compile(r'\(HR@5:([\d.]+),NDCG@5:([\d.]+),HR@10:([\d.]+),NDCG@10:([\d.]+),HR@20:([\d.]+),NDCG@20:([\d.]+),HR@50:([\d.]+),NDCG@50:([\d.]+)')

		best_epoch_pattern = re.compile(r'Best Iter\(dev\)=\s*(\d+)')
		# 初始化列表来存储提取的数据
		epochs, losses, dead_latents, hr5s, ndcg5s, hr10s, ndcg10s, hr20s, ndcg20s, hr50s, ndcg50s = ([] for _ in range(11))

		# 从目标行开始解析后续的 epoch 数据
		for line in log_data[start_index:]:
			if 'Epoch' in line:
				epoch = epoch_pattern.search(line).group(1)
				loss = loss_pattern.search(line).group(1)
				dead_latent = dead_latent_pattern.search(line).group(1)

				epochs.append(int(epoch))
				losses.append(float(loss))
				dead_latents.append(float(dead_latent))

			if line.startswith('dev='):
				metrics = metrics_pattern.search(line)
				if metrics:
					hr5s.append(float(metrics.group(1)))
					ndcg5s.append(float(metrics.group(2)))
					hr10s.append(float(metrics.group(3)))
					ndcg10s.append(float(metrics.group(4)))
					hr20s.append(float(metrics.group(5)))
					ndcg20s.append(float(metrics.group(6)))
					hr50s.append(float(metrics.group(7)))
					ndcg50s.append(float(metrics.group(8)))

			if line.startswith("Best Iter(dev)= "):
				match = best_epoch_pattern.search(line)
				if match:
					best_epoch = int(match.group(1))

			if line.startswith('INFO:root:[Rec] Dev Before Training: '):
				metrics = metrics_pattern.search(line)
				if metrics:
					recmodel_ndcg5s = float(metrics.group(2))

			if line.startswith("Test After Training: "):
				metrics = metrics_pattern.search(line)
				if metrics:
					epochs.append('RecSAE_test')
					losses.append(0)
					dead_latents.append(0)
					hr5s.append(float(metrics.group(1)))
					ndcg5s.append(float(metrics.group(2)))
					hr10s.append(float(metrics.group(3)))
					ndcg10s.append(float(metrics.group(4)))
					hr20s.append(float(metrics.group(5)))
					ndcg20s.append(float(metrics.group(6)))
					hr50s.append(float(metrics.group(7)))
					ndcg50s.append(float(metrics.group(8)))

			if line.startswith("INFO:root:[Rec] Test Before Training:"):
				metrics = metrics_pattern.search(line)
				if metrics:
					epochs.append('SASRec_test')
					losses.append(0)
					dead_latents.append(0)
					hr5s.append(float(metrics.group(1)))
					ndcg5s.append(float(metrics.group(2)))
					hr10s.append(float(metrics.group(3)))
					ndcg10s.append(float(metrics.group(4)))
					hr20s.append(float(metrics.group(5)))
					ndcg20s.append(float(metrics.group(6)))
					hr50s.append(float(metrics.group(7)))
					ndcg50s.append(float(metrics.group(8)))




		# 将数据创建为 DataFrame 以便更好地查看
		data = {
			'Epoch': epochs,
			'Loss': losses,
			'Dead Latent': dead_latents,
			'HR@5': hr5s,
			'NDCG@5': ndcg5s,
			'HR@10': hr10s,
			'NDCG@10': ndcg10s,
			'HR@20': hr20s,
			'NDCG@20': ndcg20s,
			'HR@50': hr50s,
			'NDCG@50': ndcg50s
		}

		df_all = pd.DataFrame(data)
	else:
		print("未找到目标行")

	# 输出 DataFrame 供检查
	# print(df_all)

	file_path = f"../../log/SASRec_SAE/figs/SASRec_SAE__{dataset_name}__0__lr=0.0001__l2=1e-06__emb_size=64__num_layers=1__num_heads=1__sae_lr={sae_lr}__sae_k={sae_k}__sae_scale_size={sae_scale_size}__batch_size={batch_size}.png"
	drawFig(df_all,best_epoch,recmodel_ndcg5s,file_path)

	return df_all, best_epoch

	

def drawFig(df_all,best_epoch,recmodel_ndcg5s,file_path):
	df = df_all[1:-1]
	best_loss = df.loc[df['Epoch'] == best_epoch, 'Loss'].values[0]
	best_ndcg5 = df.loc[df['Epoch'] == best_epoch, 'NDCG@5'].values[0]


	fig, ax1 = plt.subplots(figsize=(8, 6))
	# 绘制 Loss 曲线
	ax1.plot(df['Epoch'], df['Loss'], marker='o', color='blue', label='RecSAE-Train Loss')
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Loss', color='blue')
	ax1.tick_params(axis='y', labelcolor='blue')
	ax1.set_ylim(0, 0.3)

		# 标注 best epoch 在 Loss 曲线上的位置
	ax1.annotate(f'Best Epoch {best_epoch}', 
				xy=(best_epoch, best_loss), 
				xytext=(best_epoch, best_loss + 0.08),
				arrowprops=dict(facecolor='blue', arrowstyle='->'),
				color='blue')

	# 创建右侧的 y 轴并绘制 NDCG@5 曲线
	ax2 = ax1.twinx()
	ax2.axhline(recmodel_ndcg5s, color='orange', linestyle='--', label='SASRec-Dev NDCG@5')

	ax2.plot(df['Epoch'], df['NDCG@5'], marker='o', color='orange', label='RecSAE-Dev NDCG@5')
	ax2.set_ylabel('NDCG@5', color='orange')
	ax2.tick_params(axis='y', labelcolor='orange')
	ax2.set_ylim(0.025, 0.045)
	ax2.annotate(f'Best Epoch {best_epoch}', 
             xy=(best_epoch, best_ndcg5), 
             xytext=(best_epoch, best_ndcg5 + 0.0015),
             arrowprops=dict(facecolor='orange', arrowstyle='->'),
             color='orange')
	
	
	rec_test = df_all.loc[df_all['Epoch'] == "SASRec_test", 'NDCG@5'].values[0]
	sae_test = df_all.loc[df_all['Epoch'] == "RecSAE_test", 'NDCG@5'].values[0]
	ax2.axhline(rec_test, color='green', linestyle='--', label='SASRec-Test NDCG@5')
	ax2.axhline(sae_test, color='green', label='RecSAE-Test NDCG@5')


	# fig.legend(ncol=3,loc="upper center") #loc="upper right", bbox_to_anchor=(0.9, 0.9)
	# fig.tight_layout()

	# 图表标题
	plt.title(f'SASRec - {sae_lr} - {batch_size}')

	plt.savefig(file_path)

import os
if __name__=='__main__':
	sae_lr_list = ['5e-05','0.0001','0.0003','0.0005','0.001']
	batch_size_list = [8,16,32,64,128]
	sae_lr = sae_lr_list[3]
	batch_size = batch_size_list[1]
	result, best_epoch = analysis(batch_size,sae_lr)
	print(result, best_epoch )
	# rec_result = pd.DataFrame(index = batch_size_list, columns = sae_lr_list)
	# loss_result = pd.DataFrame(index = batch_size_list, columns = sae_lr_list)
	
	# for sae_lr in sae_lr_list:
	# 	for batch_size in batch_size_list:
	# 		result, best_epoch = analysis(batch_size,sae_lr)
	# 		if result is None:
	# 			continue
	# 		print(batch_size, sae_lr)
	# 		result.set_index('Epoch',inplace=True)
	# 		rec_result.loc[batch_size,sae_lr] = result.loc['RecSAE_test','NDCG@5']
	# 		loss_result.loc[batch_size,sae_lr] = result.loc[best_epoch,'Loss']
	# rec_result.fillna(0,inplace=True)
	# loss_result.fillna(0,inplace=True)
	# print(rec_result)
	# print(loss_result)
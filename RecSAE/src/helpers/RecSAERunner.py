import torch
import torch.nn as nn
import os
from time import time
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from utils import utils
from models.BaseModel import BaseModel
import numpy as np
from scipy import stats
import json
from models.sae.sae import SAE
from typing import Dict, List
import pandas as pd
import gc

from helpers.BaseRunner import BaseRunner

def sig_test(x,y):
	t_stat, p_value = stats.ttest_ind(x, y)
	alpha = 0.05 
	if p_value < alpha:
		# print("两组数据之间存在显著性差异")
		return True
	else:
		# print("两组数据之间不存在显著性差异")
		return False
	
class RecSAERunner(BaseRunner):
	@staticmethod
	def parse_runner_args(parser):
		parser = BaseRunner.parse_runner_args(parser)
		parser.add_argument('--sae_lr', type=float, default=1e-4,
							help='SAE Learning rate.')
		parser.add_argument('--sae_train', type=int, default=0,
							help='train sae or evaluate RecSAE')
		parser.add_argument('--result_data_path', type=str, default="",
							help='base path to save prediction list and RecSAE activations')
		return parser
	
	
	
	def __init__(self, args):
		BaseRunner.__init__(self,args)
		self.learning_rate = args.sae_lr
		self.sae_train = args.sae_train
		self.result_data_path = args.result_data_path


	def train(self, data_dict: Dict[str, BaseModel.Dataset]):
		model = data_dict['train'].model
		model.eval()

		main_metric_results, dev_results = list(), list()
		self._check_time(start=True)

		# model.set_sae_mode("inference")
		# for key in ['dev','test']:
		# 	dev_result = self.evaluate(data_dict[key], self.topk[:1], self.metrics, prediction_label = "prediction")
		# 	dev_results.append(dev_result)
		# 	main_metric_results.append(dev_result[self.main_metric])
		# 	logging_str = '[Without SAE] {}=({})'.format(
		# 		key, utils.format_metric(dev_result))

		for epoch in range(self.epoch):
			self._check_time()
			gc.collect()
			torch.cuda.empty_cache()
			model.set_sae_mode("train")
			loss = self.fit(data_dict['train'], epoch=epoch + 1)
			if np.isnan(loss):
				logging.info("Loss is Nan. Stop training at %d."%(epoch+1))
				break
			training_time = self._check_time()
			dead_latent_ratio = model.get_dead_latent_ratio()
			logging_str = 'Epoch {:<5}loss={:<.4f}, dead_latent={:<.4f} [{:<3.1f} s]'.format(
				epoch + 1, loss, dead_latent_ratio, training_time)
			logging.info(logging_str)

			model.set_sae_mode("inference")
			# Record dev results
			dev_result = self.evaluate(data_dict['dev'], self.topk, self.metrics, prediction_label = "prediction_sae") # [self.main_topk]
			dev_results.append(dev_result)
			main_metric_results.append(dev_result[self.main_metric])
			dead_latent_ratio = model.get_dead_latent_ratio()
			logging_str = '[Dev] dead_latent={:<.4f}\ndev=({})'.format(
				 dead_latent_ratio, utils.format_metric(dev_result))

			# Test
			if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
				test_result = self.evaluate(data_dict['test'], self.topk, self.metrics, prediction_label = "prediction_sae")
				dead_latent_ratio = model.get_dead_latent_ratio()
				logging_str += '[Test] dead_latent={:<.4f}}\ntest=({})'.format(dead_latent_ratio, utils.format_metric(test_result))
			testing_time = self._check_time()
			logging_str += ' [{:<.1f} s]'.format(testing_time)

			try:
				# Проверяем, доступен ли путь для записи
				dir_path = os.path.dirname(model.recsae_model_path)
				if dir_path and not os.path.exists(dir_path):
					os.makedirs(dir_path, exist_ok=True)
					logging.info(f"Created directory: {dir_path}")
					
				# Проверяем длину пути - Windows имеет ограничение около 260 символов
				if len(model.recsae_model_path) > 240:
					logging.warning(f"Path too long, using shorter one")
					model.recsae_model_path = os.path.join(dir_path, f"model_{int(time())}.pt")
					
				# Пробуем использовать метод модели
				model.save_model(model.recsae_model_path)
				logging_str += ' *'
			except Exception as e:
				# Если метод модели вызвал ошибку, используем безопасное сохранение
				logging.warning(f"Error using model's save_model method: {str(e)}")
				if safe_save_model(model, model.recsae_model_path):
					logging_str += ' * (saved with backup method)'
				else:
					# Последний вариант - сохранить в текущей директории
					try:
						simple_path = f"model_backup_{int(time())}.pt"
						torch.save(model.state_dict(), simple_path)
						model.recsae_model_path = os.path.abspath(simple_path)
						logging.info(f"Saved model to simple path: {simple_path}")
						logging_str += ' * (saved to simple path)'
					except Exception as backup_e:
						logging.error(f"All save attempts failed: {str(backup_e)}")
						logging_str += ' * (failed to save)'

			# Выводим результаты в лог
			logging.info(logging_str)

			if self.early_stop > 0 and self.eval_termination(main_metric_results):
				logging.info("Early stop at %d based on dev result." % (epoch + 1))
				break
		
		# Find the best dev result across iterations
		best_epoch = main_metric_results.index(max(main_metric_results))
		logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
			best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
		model.load_model(model.recsae_model_path)

	def fit(self, dataset, epoch = -1):
		model = dataset.model
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)
		dataset.actions_before_epoch()
		model.sae_module.set_decoder_norm_to_unit_norm()
		
		loss_lst = list()
		dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
			batch = utils.batch_to_gpu(batch, model.device)
			# randomly shuffle the items to avoid models remembering the first item being the target
			item_ids = batch['item_id']
			# for each row (sample), get random indices and shuffle the original items
			indices = torch.argsort(torch.rand(*item_ids.shape), dim=-1)						
			batch['item_id'] = item_ids[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices]

			model.optimizer.zero_grad()
			out_dict = model(batch)

			if epoch == 1:
				loss = model.sae_module.fvu
			else:
				loss = model.sae_module.fvu + model.sae_module.auxk_loss/32

			loss.backward()
			model.optimizer.step()
			loss_lst.append(loss.detach().cpu().data.numpy())
		loss = np.mean(loss_lst).item()
		if np.isnan(loss):
			import ipdb;ipdb.set_trace()

		return loss
	

	def print_res(self, dataset: BaseModel.Dataset, prediction_label = "prediction", save_result = False) -> str:
		if save_result: # sae_train 0 and test set
			dataset.model.set_sae_mode("test")
		else:
			dataset.model.set_sae_mode("inference")
		result = BaseRunner.print_res(self,dataset, prediction_label=prediction_label)
		if save_result:
			model_path = self.result_data_path + "_activation.csv"
			dataset.model.save_epoch_result(dataset,path = model_path)
			logging.info(f'[RecSAE Runner] save activation data\n{model_path}')
		return result


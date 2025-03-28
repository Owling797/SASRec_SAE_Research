# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import pandas as pd
import torch
import hashlib
import glob
import time
import gc
import numpy as np

# Добавляем импорт для прямого доступа к модулям
import models.sequential

from helpers import *
from models.general import *
from models.sequential import *
from models.developing import *
from models.context import *
from models.context_seq import *
from models.reranker import *
from utils import utils


def parse_global_args(parser):
	parser.add_argument('--gpu', type=str, default='0',
						help='Set CUDA_VISIBLE_DEVICES, default for CPU only')
	parser.add_argument('--verbose', type=int, default=logging.INFO,
						help='Logging Level, 0, 10, ..., 50')
	parser.add_argument('--log_file', type=str, default='',
						help='Logging file path')
	parser.add_argument('--random_seed', type=int, default=0,
						help='Random seed of numpy and pytorch')
	parser.add_argument('--load', type=int, default=0,
						help='Whether load model and continue to train')
	parser.add_argument('--train', type=int, default=1,
						help='To train the model or not.')
	parser.add_argument('--save_final_results', type=int, default=1,
						help='To save the final validation and test results or not.')
	parser.add_argument('--regenerate', type=int, default=0,
						help='Whether to regenerate intermediate files')
	return parser

def train_sae(args, model, runner, data_dict):
	# Для сохранения метрик по эпохам
	dev_metrics_by_epoch = []
	
	# Проверка и исправление путей к моделям
	if not os.path.isabs(args.model_path):
		args.model_path = os.path.abspath(args.model_path)
		logging.info(f"Converted model_path to absolute: {args.model_path}")
		
	if not os.path.isabs(args.recsae_model_path):
		args.recsae_model_path = os.path.abspath(args.recsae_model_path)
		logging.info(f"Converted recsae_model_path to absolute: {args.recsae_model_path}")
		
	# Проверка существования директорий для модели
	os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
	os.makedirs(os.path.dirname(args.recsae_model_path), exist_ok=True)
	
	# Check if the base model exists before loading it
	if os.path.exists(args.model_path):
		model.load_model(args.model_path)
		logging.info(f'Base model loaded from {args.model_path}')
	else:
		# Поиск моделей в директории с "recsae_Attention"
		attention_pattern = os.path.join(os.path.abspath('../model/*Attention*'), '*.pt')
		attention_models = glob.glob(attention_pattern)
		
		# Если ничего не найдено, поищем в базовой директории моделей
		if not attention_models:
			base_pattern = os.path.join(os.path.abspath('../model/*/'), '*.pt')
			attention_models = glob.glob(base_pattern)
		
		# Если нашли модели, загрузим первую из них
		if attention_models:
			try:
				model_path = attention_models[0]
				logging.info(f'Found alternative model at {model_path}')
				model.load_model(model_path)
				logging.info(f'Loaded base model from alternative path: {model_path}')
			except Exception as e:
				logging.warning(f'Failed to load alternative model: {str(e)}')
				logging.warning(f'Training SAE from scratch without pretrained base model.')
		else:
			logging.warning(f'No suitable models found. Training SAE from scratch without pretrained base model.')
	
	logging.info('[Rec] Dev Before Training: ' + runner.print_res(data_dict['dev']))
	logging.info('[SAE] Dev Before Training: ' + runner.print_res(data_dict['dev'], prediction_label = 'prediction_sae'))

	logging.info('[Rec] Test Before Training: ' + runner.print_res(data_dict['test']))
	logging.info('[SAE] Test Before Training: ' + runner.print_res(data_dict['test'], prediction_label = 'prediction_sae'))

	if args.train > 0:
		# Модифицируем train метод, чтобы он возвращал метрики по эпохам
		epoch_metrics = []
		
		# Настраиваем метрики для отслеживания
		target_metrics = ['hr@5', 'ndcg@5','hr@10',  'ndcg@10','hr@20',  'ndcg@20','hr@50',  'ndcg@50']
		
		# Сохраняем оригинальный метод оценки для использования после каждой эпохи
		original_evaluate = runner.evaluate
		metrics_by_epoch_dev = []
		metrics_by_epoch_test = []
		
		# Переопределяем метод train для отслеживания метрик
		original_train = runner.train
		
		def train_with_metrics(data_dict_arg):
			model = data_dict_arg['train'].model
			model.eval()
			
			main_metric_results, dev_results = list(), list()
			runner._check_time(start=True)
			
			for epoch in range(runner.epoch):
				runner._check_time()
				gc.collect()
				torch.cuda.empty_cache()
				model.set_sae_mode("train")
				# Устанавливаем принудительно num_workers=0 в runner
				original_num_workers = runner.num_workers
				runner.num_workers = 0
				loss = runner.fit(data_dict_arg['train'], epoch=epoch + 1)
				# Восстанавливаем оригинальное значение
				runner.num_workers = original_num_workers
				if np.isnan(loss):
					logging.info("Loss is Nan. Stop training at %d."%(epoch+1))
					break
				training_time = runner._check_time()
				dead_latent_ratio = model.get_dead_latent_ratio()
				logging_str = 'Epoch {:<5}loss={:<.4f}, dead_latent={:<.4f} [{:<3.1f} s]'.format(
					epoch + 1, loss, dead_latent_ratio, training_time)
				logging.info(logging_str)
				
				model.set_sae_mode("inference")
				# Оцениваем на dev
				dev_result = original_evaluate(data_dict_arg['dev'], runner.topk, runner.metrics, prediction_label="prediction_sae")
				
				# Сохраняем метрики для dev
				epoch_dev_metrics = {k.lower(): v for k, v in dev_result.items()}
				metrics_by_epoch_dev.append(epoch_dev_metrics)
				
				# Оцениваем на test после каждой эпохи
				test_result = original_evaluate(data_dict_arg['test'], runner.topk, runner.metrics, prediction_label="prediction_sae")
				epoch_test_metrics = {k.lower(): v for k, v in test_result.items()}
				metrics_by_epoch_test.append(epoch_test_metrics)
				
				# Выводим метрики
				dev_metrics_str = ', '.join([f"{m}={epoch_dev_metrics.get(m, 0):.4f}" for m in target_metrics if m in epoch_dev_metrics])
				test_metrics_str = ', '.join([f"{m}={epoch_test_metrics.get(m, 0):.4f}" for m in target_metrics if m in epoch_test_metrics])
				
				logging.info(f"Epoch {epoch+1} Dev Metrics: {dev_metrics_str}")
				logging.info(f"Epoch {epoch+1} Test Metrics: {test_metrics_str}")
				
				# Выводим метрики в табличном виде для лучшей читаемости
				table_header = f"+{'-'*15}+{'-'*12}+{'-'*12}+"
				table_format = "| {:<13} | {:<10} | {:<10} |"
				
				logging.info(f"\nEpoch {epoch+1} Metrics:")
				logging.info(table_header)
				logging.info(table_format.format("Metric", "Dev", "Test"))
				logging.info(table_header)
				
				for metric in target_metrics:
					dev_value = epoch_dev_metrics.get(metric, 0)
					test_value = epoch_test_metrics.get(metric, 0)
					logging.info(table_format.format(metric, f"{dev_value:.4f}", f"{test_value:.4f}"))
				
				logging.info(table_header)
				
				# Продолжаем оригинальную логику train
				dev_results.append(dev_result)
				main_metric_results.append(dev_result[runner.main_metric])
				
				testing_time = runner._check_time()
				
				# Сохранение модели при необходимости
				try:
					if max(main_metric_results) == main_metric_results[-1]:
						model.save_model(model.recsae_model_path)
						logging.info("* Best model so far, saved")
				except Exception as e:
					logging.error(f"Error saving model: {str(e)}")
					
				if runner.early_stop > 0 and runner.eval_termination(main_metric_results):
					logging.info("Early stop at %d based on dev result." % (epoch + 1))
					break
					
			# Загружаем лучшую модель
			best_epoch = main_metric_results.index(max(main_metric_results))
			logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
				best_epoch + 1, utils.format_metric(dev_results[best_epoch]), runner.time[1] - runner.time[0]))
			model.load_model(model.recsae_model_path)
			
			# Сохраняем метрики непосредственно здесь, чтобы гарантировать, что они сохранены
			# даже если произойдет ранний выход из программы
			try:
				# Создаем директорию для сохранения метрик
				metrics_dir = os.path.join(os.path.dirname(args.result_data_path), "metrics")
				os.makedirs(metrics_dir, exist_ok=True)
				
				# Сохраняем метрики в файл CSV
				metrics_file = os.path.join(metrics_dir, f"{args.dataset}_{init_args.model_name}_metrics.csv")
				
				# Создаем DataFrame для сохранения метрик
				import pandas as pd
				
				# Создаем таблицу метрик: строки - метрики, столбцы - эпохи
				metrics_data = {'metric': target_metrics}
				
				# Добавляем столбцы для каждой эпохи (dev)
				for epoch_idx in range(len(metrics_by_epoch_dev)):
					metrics_data[f'dev_epoch_{epoch_idx+1}'] = [metrics_by_epoch_dev[epoch_idx].get(m, 0) for m in target_metrics]
				
				# Добавляем столбцы для каждой эпохи (test)
				for epoch_idx in range(len(metrics_by_epoch_test)):
					metrics_data[f'test_epoch_{epoch_idx+1}'] = [metrics_by_epoch_test[epoch_idx].get(m, 0) for m in target_metrics]
				
				# Создаем и сохраняем DataFrame
				metrics_df = pd.DataFrame(metrics_data)
				metrics_df.to_csv(metrics_file, index=False)
				logging.info(f"Metrics saved to {metrics_file}")
				
				# Также сохраняем метрики в формате, удобном для построения графиков:
				# для каждой метрики создаем отдельный файл с колонками epoch, dev, test
				for metric in target_metrics:
					metric_file = os.path.join(metrics_dir, f"{args.dataset}_{init_args.model_name}_{metric}.csv")
					metric_data = {
						'epoch': list(range(1, len(metrics_by_epoch_dev) + 1)),
						'dev': [metrics.get(metric, 0) for metrics in metrics_by_epoch_dev],
						'test': [metrics.get(metric, 0) for metrics in metrics_by_epoch_test]
					}
					pd.DataFrame(metric_data).to_csv(metric_file, index=False)
					logging.info(f"Metric {metric} saved to {metric_file}")
			except Exception as e:
				logging.error(f"Error saving metrics: {str(e)}")
		
		# Запоминаем оригинальный метод
		original_train_method = runner.train
		
		# Заменяем метод train нашей модифицированной версией
		runner.train = train_with_metrics
		
		# Вызываем обучение
		runner.train(data_dict)
		
		# Восстанавливаем оригинальный метод
		runner.train = original_train_method
		
	# Evaluate final results
	eval_res = runner.print_res(data_dict['dev'], prediction_label = 'prediction_sae')
	logging.info(os.linesep + 'Dev  After Training: ' + eval_res)
	eval_res = runner.print_res(data_dict['test'], prediction_label = 'prediction_sae')
	logging.info(os.linesep + 'Test After Training: ' + eval_res)
	
	# Вывод итоговой сводной таблицы метрик, если обучение проводилось
	if args.train > 0 and 'metrics_by_epoch_dev' in locals() and len(metrics_by_epoch_dev) > 0:
		best_epoch_idx = -1  # Используем последнюю эпоху по умолчанию
		
		# Найдем лучшую эпоху по метрике HR@5 на validation
		if 'hr@5' in metrics_by_epoch_dev[0]:
			hr5_values = [m.get('hr@5', 0) for m in metrics_by_epoch_dev]
			best_epoch_idx = hr5_values.index(max(hr5_values))
		
		# Вывод итоговой таблицы метрик для лучшей эпохи
		best_epoch = best_epoch_idx + 1
		logging.info(f"\n{'='*70}")
		logging.info(f"FINAL METRICS SUMMARY (Best Epoch: {best_epoch})")
		logging.info(f"{'='*70}")
		
		table_header = f"+{'-'*15}+{'-'*12}+{'-'*12}+"
		table_format = "| {:<13} | {:<10} | {:<10} |"
		
		logging.info(table_header)
		logging.info(table_format.format("Metric", "Dev", "Test"))
		logging.info(table_header)
		
		for metric in target_metrics:
			dev_value = metrics_by_epoch_dev[best_epoch_idx].get(metric, 0)
			test_value = metrics_by_epoch_test[best_epoch_idx].get(metric, 0)
			logging.info(table_format.format(metric, f"{dev_value:.4f}", f"{test_value:.4f}"))
		
		logging.info(table_header)
		logging.info(f"{'='*70}\n")

	model.actions_after_train()
	# Flush all logging output to ensure it's written
	for handler in logging.root.handlers:
		handler.flush()
	logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


def test_sae(args, model, runner, data_dict):
	# Проверка и исправление путей к моделям
	if not os.path.isabs(args.recsae_model_path):
		args.recsae_model_path = os.path.abspath(args.recsae_model_path)
		logging.info(f"Converted recsae_model_path to absolute: {args.recsae_model_path}")
		
	# Проверка существования директорий для результатов
	result_dir = os.path.dirname(args.result_data_path)
	if result_dir and not os.path.exists(result_dir):
		os.makedirs(result_dir, exist_ok=True)
		logging.info(f"Created directory for results: {result_dir}")
		
	# Check if the SAE model exists before loading it
	if os.path.exists(args.recsae_model_path):
		model.load_model(args.recsae_model_path)
		logging.info(f'SAE model loaded from {args.recsae_model_path}')
	else:
		logging.error(f'SAE model not found at {args.recsae_model_path}. Cannot test without a trained model.')
		logging.info('Please train the model first by setting sae_train=1 or provide a valid model path.')
		return
	
	eval_res = runner.print_res(data_dict['dev'], prediction_label = 'prediction_sae')
	logging.info(os.linesep + 'Dev  After Training: ' + eval_res)
	eval_res = runner.print_res(data_dict['test'], prediction_label = 'prediction_sae',save_result = True)
	logging.info(os.linesep + 'Test After Training: ' + eval_res)
	
	if args.save_final_results==1: # save the prediction results
		# save_rec_results(data_dict['dev'], runner, 100)
		save_rec_results(data_dict['test'], runner, 100)
	

def main():
	logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
	exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory', 'load',
			   'regenerate', 'sep', 'train', 'verbose', 'metric', 'test_epoch', 'buffer']
	logging.info(utils.format_arg_str(args, exclude_lst=exclude))

	# Random seed
	utils.init_seed(args.random_seed)

	# GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	args.device = torch.device('cpu')
	if args.gpu != '' and torch.cuda.is_available():
		args.device = torch.device('cuda')
	logging.info('Device: {}'.format(args.device))

	# Read data
	corpus_path = os.path.join(args.path, args.dataset, model_name.reader+args.data_appendix+ '.pkl')
	if not args.regenerate and os.path.exists(corpus_path):
		logging.info('Load corpus from {}'.format(corpus_path))
		corpus = pickle.load(open(corpus_path, 'rb'))
	else:
		corpus = reader_name(args)
		logging.info('Save corpus to {}'.format(corpus_path))
		pickle.dump(corpus, open(corpus_path, 'wb'))

	# Define model
	model = model_name(args, corpus).to(args.device)
	logging.info('#params: {}'.format(model.count_variables()))
	logging.info(model)

	# Define dataset
	data_dict = dict()
	for phase in ['train', 'dev', 'test']:
		data_dict[phase] = model_name.Dataset(model, corpus, phase)
		data_dict[phase].prepare()

	runner = runner_name(args)

	if args.sae_train:
		train_sae(args, model, runner, data_dict)
	else:
		test_sae(args, model, runner, data_dict)


def save_rec_results(dataset, runner, topk, predict_label = 'prediction_sae'):
	result_path = runner.result_data_path + '_prediction.csv'
	utils.check_dir(result_path)

	if init_args.model_mode == 'CTR': # CTR task 
		logging.info('Saving CTR prediction results to: {}'.format(result_path))
		predictions, labels = runner.predict(dataset)
		users, items= list(), list()
		for i in range(len(dataset)):
			info = dataset[i]
			users.append(info['user_id'])
			items.append(info['item_id'][0])
		rec_df = pd.DataFrame(columns=['user_id', 'item_id', 'pCTR', 'label'])
		rec_df['user_id'] = users
		rec_df['item_id'] = items
		rec_df['pCTR'] = predictions
		rec_df['label'] = labels
		rec_df.to_csv(result_path, sep=args.sep, index=False)
	elif init_args.model_mode in ['TopK','']: # TopK Ranking task
		logging.info('Saving top-{} recommendation results to: {}'.format(topk, result_path))
		predictions = runner.predict(dataset, prediction_label = predict_label)  # n_users, n_candidates
		users, rec_items, rec_predictions = list(), list(), list()
		for i in range(len(dataset)):
			info = dataset[i]
			users.append(info['user_id'])
			item_scores = zip(info['item_id'], predictions[i])
			sorted_lst = sorted(item_scores, key=lambda x: x[1], reverse=True)[:topk]
			rec_items.append([x[0] for x in sorted_lst])
			rec_predictions.append([x[1] for x in sorted_lst])
		rec_df = pd.DataFrame(columns=['user_id', 'rec_items', 'rec_predictions'])
		rec_df['user_id'] = users
		rec_df['rec_items'] = rec_items
		rec_df['rec_predictions'] = rec_predictions
		rec_df.to_csv(result_path, sep=args.sep, index=False)
	elif init_args.model_mode in ['Impression','General','Sequential']: # List-wise reranking task
		logging.info('Saving all recommendation results to: {}'.format(result_path))
		predictions = runner.predict(dataset)  # n_users, n_candidates
		users, pos_items, pos_predictions, neg_items, neg_predictions= list(), list(), list(), list(), list()
		for i in range(len(dataset)):
			info = dataset[i]
			users.append(info['user_id'])
			pos_items.append(info['pos_items'])
			neg_items.append(info['neg_items'])
			pos_predictions.append(predictions[i][:dataset.pos_len])
			neg_predictions.append(predictions[i][:dataset.neg_len])
		rec_df = pd.DataFrame(columns=['user_id', 'pos_items', 'pos_predictions', 'neg_items', 'neg_predictions'])
		rec_df['user_id'] = users
		rec_df['pos_items'] = pos_items
		rec_df['pos_predictions'] = pos_predictions
		rec_df['neg_items'] = neg_items
		rec_df['neg_predictions'] = neg_predictions
		rec_df.to_csv(result_path, sep=args.sep, index=False)
	else:
		return 0
	logging.info("{} Prediction results saved!".format(dataset.phase))

if __name__ == '__main__':
	init_parser = argparse.ArgumentParser(description='Model')
	init_parser.add_argument('--model_name', type=str, default='SASRec_SAE_Attention', help='Choose a model to run.')
	init_parser.add_argument('--model_mode', type=str, default='', 
							 help='Model mode(i.e., suffix), for context-aware models to select "CTR" or "TopK" Ranking task;\
            						for general/seq models to select Normal (no suffix, model_mode="") or "Impression" setting;\
                  					for rerankers to select "General" or "Sequential" Baseranker.')
	init_args, init_extras = init_parser.parse_known_args()
	
	# Используем полное имя модели вместо разделения на базовую модель
	model_name = eval('models.sequential.{0}{1}'.format(init_args.model_name, init_args.model_mode))
	reader_name = eval('{0}.{0}'.format(model_name.reader))  # model chooses the reader
	runner_name = eval('{0}.{0}'.format(model_name.runner))  # model chooses the runner

	# Args
	parser = argparse.ArgumentParser(description='')
	parser = parse_global_args(parser)
	parser = reader_name.parse_data_args(parser)
	parser = runner_name.parse_runner_args(parser)
	parser = model_name.parse_model_args(parser)
	args, extras = parser.parse_known_args()
	
	args.data_appendix = '' # save different version of data for, e.g., context-aware readers with different groups of context
	if 'Context' in model_name.reader:
		args.data_appendix = '_context%d%d%d'%(args.include_item_features,args.include_user_features,
										args.include_situation_features)

	# Используем базовую модель SASRec для имени файла лога
	base_model = init_args.model_name.split('_')[0]
	log_args = [base_model+init_args.model_mode, args.dataset+args.data_appendix, str(args.random_seed)]
	for arg in ['lr', 'l2'] + model_name.extra_log_args:
		log_args.append(arg + '=' + str(eval('args.' + arg)))
	log_file_name = '__'.join(log_args).replace(' ', '__')

	log_args = [init_args.model_name+init_args.model_mode, args.dataset+args.data_appendix, str(args.random_seed)]
	for arg in ['lr', 'l2'] + model_name.extra_log_args + model_name.sae_extra_params + ['batch_size']:
		log_args.append(arg + '=' + str(eval('args.' + arg)))
	log_file_name_all = '__'.join(log_args).replace(' ', '__')

	if args.log_file == '':
		args.log_file = '../log/{}/{}.txt'.format(init_args.model_name+init_args.model_mode, log_file_name_all)
	if args.model_path == '':
		# Правильно извлекаем базовую модель SASRec из имени модели
		base_model = init_args.model_name.split('_')[0]
		args.model_path = os.path.abspath('../model/{}/{}.pt'.format(base_model+init_args.model_mode, log_file_name))
	if args.recsae_model_path == "":
		# Создаем более короткое имя файла модели
		model_dir = os.path.abspath('../model/{}'.format(init_args.model_name+init_args.model_mode))
		# Используем хеш вместо полного имени файла
		hash_name = hashlib.md5(log_file_name_all.encode()).hexdigest()[:12]
		args.recsae_model_path = os.path.join(model_dir, f'{hash_name}.pt')
	if args.result_data_path == "":
		# Также сокращаем путь к результатам
		result_dir = os.path.abspath('../log/{}/results'.format(init_args.model_name+init_args.model_mode))
		# Используем тот же хеш для согласованности
		if 'hash_name' not in locals():
			hash_name = hashlib.md5(log_file_name_all.encode()).hexdigest()[:12]
		args.result_data_path = os.path.join(result_dir, hash_name)

	# Проверим все необходимые пути ещё раз и выведем их в лог
	logging.info(f"Model path: {args.model_path}")
	logging.info(f"RecSAE model path: {args.recsae_model_path}")
	logging.info(f"Result data path: {args.result_data_path}")
	logging.info(f"Log file: {args.log_file}")
	
	# Функция для создания всех родительских директорий
	def ensure_dir_exists(file_path):
		"""Создает все родительские директории для указанного пути к файлу"""
		dir_path = os.path.dirname(file_path)
		if dir_path and not os.path.exists(dir_path):
			try:
				os.makedirs(dir_path, exist_ok=True)
				logging.info(f"Created directory: {dir_path}")
				return True
			except Exception as e:
				logging.error(f"Failed to create directory {dir_path}: {str(e)}")
				return False
		return True
	
	# Создаем все необходимые директории
	for path in [args.model_path, args.recsae_model_path, args.result_data_path, args.log_file]:
		if path:
			ensure_dir_exists(path)
			
	# Проверяем, существуют ли директории, и если нет, пытаемся создать их с абсолютными путями
	model_dir = os.path.dirname(args.recsae_model_path)
	if not os.path.exists(model_dir):
		logging.warning(f"Directory for model still doesn't exist: {model_dir}")
		try:
			os.makedirs(model_dir, exist_ok=True)
			logging.info(f"Successfully created directory: {model_dir}")
		except Exception as e:
			logging.error(f"Failed to create directory {model_dir}: {str(e)}")
			
			# Крайний случай - сохраняем в текущую директорию
			backup_dir = os.path.join(os.getcwd(), "model_backup")
			args.recsae_model_path = os.path.join(backup_dir, os.path.basename(args.recsae_model_path))
			os.makedirs(backup_dir, exist_ok=True)
			logging.info(f"Will use backup path instead: {args.recsae_model_path}")

	# Настраиваем логгирование только в консоль
	logging.basicConfig(level=args.verbose, format='%(asctime)s - %(levelname)s - %(message)s')
	logging.info(init_args)
	
	# Создаем директорию для log_file, если она задана
	if args.log_file:
		log_dir = os.path.dirname(args.log_file)
		if log_dir and not os.path.exists(log_dir):
			os.makedirs(log_dir, exist_ok=True)
			
	try:
		# Добавляем обработку исключений для main()
		main()
	except Exception as e:
		logging.error(f"Error in main execution: {str(e)}")
		import traceback
		logging.error(traceback.format_exc())
	finally:
		# Убеждаемся, что все выводы записаны
		for handler in logging.root.handlers:
			handler.flush()
		logging.info("Program execution completed.") 
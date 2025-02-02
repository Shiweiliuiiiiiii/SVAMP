# coding: utf-8
import time
import torch.optim
from collections import OrderedDict
from attrdict import AttrDict
import pandas as pd
try:
	import cPickle as pickle
except ImportError:
	import pickle

from src.args import build_parser

from src.train_and_evaluate import *
from src.components.models import *
from src.components.contextual_embeddings import *
from src.utils.helper import *
from src.utils.logger import *
from src.utils.expressions_transfer import *
import sys
sys.path.append('../')
from sparse_core import Masking, CosineDecay
import copy
global log_folder
global model_folder
global result_folder
global data_path
global board_path

log_folder = 'logs'
model_folder = 'models'
board_path = './runs/'
outputs_folder = 'outputs'
result_folder = './out/'
data_path = '../../data/'

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def main():
	parser = build_parser()
	args = parser.parse_args()
	config = args
	if config.mode == 'train':
		is_train = True
	else:
		is_train = False

	''' Set seed for reproducibility'''
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	random.seed(config.seed)

	'''GPU initialization'''
	device = gpu_init_pytorch(config.gpu)

	if config.full_cv:
		global data_path
		data_name = config.dataset
		data_path = data_path + data_name + '/'

		if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)

		config.val_result_path = os.path.join(args.output_dir, result_folder, 'CV_results_{}.json'.format(data_name))

		fold_acc_score = 0.0
		folds_scores = []
		best_acc = []
		for z in range(5):
			run_name = config.run_name + '_fold' + str(z)
			config.dataset = 'fold' + str(z)
			config.log_path = os.path.join(args.output_dir, log_folder, run_name)
			config.model_path = os.path.join(args.output_dir, model_folder, run_name)
			config.model_load_path = os.path.join(config.pretrained_dir, model_folder, run_name)
			config.board_path = os.path.join(args.output_dir, board_path, run_name)
			config.outputs_path = os.path.join(args.output_dir, outputs_folder, run_name)

			for file in [config.log_path, config.model_path, config.board_path, config.outputs_path]:
				if not os.path.exists(file): os.makedirs(file)

			vocab1_path = os.path.join(config.model_path, 'vocab1.p')
			vocab2_path = os.path.join(config.model_path, 'vocab2.p')
			config_file = os.path.join(config.model_path, 'config.p')
			log_file = os.path.join(config.log_path, 'log.txt')

			if config.results:
				config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))


			logger = get_logger(run_name, log_file, logging.DEBUG)

			logger.info('Experiment Name: {}'.format(config.run_name))
			logger.debug('Created Relevant Directories')

			logger.info('Loading Data...')

			train_ls, dev_ls = load_raw_data(data_path, config.dataset)
			pairs_trained, pairs_tested, generate_nums, copy_nums = transfer_num(train_ls, dev_ls, config.challenge_disp)

			logger.debug('Data Loaded...')
			logger.debug('Number of Training Examples: {}'.format(len(pairs_trained)))
			logger.debug('Number of Testing Examples: {}'.format(len(pairs_tested)))
			logger.debug('Extra Numbers: {}'.format(generate_nums))
			logger.debug('Maximum Number of Numbers: {}'.format(copy_nums))

			# pairs: ([list of words in question], [list of infix Equation tokens incl brackets and N0, N1], [list of numbers], [list of indexes of numbers])
			# generate_nums: Unmentioned numbers used in eqns in atleast 5 examples ['1', '3.14']
			# copy_nums: Maximum number of numbers in a single sentence: 15

			# pairs: ([list of words in question], [list of prefix Equation tokens w/ metasymbols as N0, N1], [list of numbers], [list of indexes of numbers])

			logger.info('Creating Vocab...')
			input_lang = None
			output_lang = None

			input_lang, output_lang, train_pairs, test_pairs = prepare_data(config, logger, pairs_trained, pairs_tested, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)

			# checkpoint = get_latest_checkpoint(config.model_load_path, logger)

			logger.info('Initializing Models...')

			# Initialize models
			embedding = None
			if config.embedding == 'bert':
				embedding = BertEncoder(config.emb_name, device, config.freeze_emb)
			elif config.embedding == 'roberta':
				embedding = RobertaEncoder(config.emb_name, device, config.freeze_emb)
			else:
				embedding = Embedding(config, input_lang, input_size=input_lang.n_words, embedding_size=config.embedding_size, dropout=config.dropout)

			# encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=config.embedding_size, hidden_size=config.hidden_size, n_layers=config.depth, dropout=config.dropout)
			encoder = EncoderSeq(cell_type=config.cell_type, embedding_size=config.embedding_size, hidden_size=config.hidden_size, n_layers=config.depth, dropout=config.dropout)
			predict = Prediction(hidden_size=config.hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums), input_size=len(generate_nums), dropout=config.dropout)
			generate = GenerateNode(hidden_size=config.hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums), embedding_size=config.embedding_size, dropout=config.dropout)
			merge = Merge(hidden_size=config.hidden_size, embedding_size=config.embedding_size, dropout=config.dropout)
			# the embedding layer is only for generated number embeddings, operators, and paddings

			logger.debug('Models Initialized')

			logger.info('Loading Models on GPU {}...'.format(config.gpu))

			# Move models to GPU
			if USE_CUDA:
				embedding.to(device)
				encoder.to(device)
				predict.to(device)
				generate.to(device)
				merge.to(device)

			logger.debug('Models loaded on GPU {}'.format(config.gpu))

			generate_num_ids = []
			for num in generate_nums:
				generate_num_ids.append(output_lang.word2index[num])

			# load the pretrained initialization
			logger.info('loading pretrained models from {}.'.format(config.model_load_path))

			ckpt_path = os.path.join(config.model_load_path, 'model.pt')
			checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
			embedding.load_state_dict(checkpoint['embedding_state_dict'])
			encoder.load_state_dict(checkpoint['encoder_state_dict'])
			predict.load_state_dict(checkpoint['predict_state_dict'])
			generate.load_state_dict(checkpoint['generate_state_dict'])
			merge.load_state_dict(checkpoint['merge_state_dict'])

			modules = [embedding, encoder, predict, generate, merge]
			input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(
				train_pairs, config.batch_size)

			############## code for pruning #################

			# performing pruning before the first iteration
			decay = CosineDecay(args.prune_rate, int(args.epochs * len(output_lengths)))
			mask = Masking(None, prune_rate_decay=decay, prune_rate=args.prune_rate,
						   sparsity=args.sparsity, prune_mode=args.prune, growth_mode=args.growth,
						   redistribution_mode=args.redistribution, fp16=args.fp16, args=args)
			mask.add_module(modules)
			mask.init(model=modules, train_loader=None, device=mask.device, mode=mask.sparse_init,
					  density=(1 - args.sparsity))
			mask.apply_mask()
			mask.print_status()

			max_value_corr = 0
			len_total_eval = 0
			max_val_acc = 0.0
			max_train_acc = 0.0
			eq_acc = 0.0
			best_epoch = -1
			min_train_loss = float('inf')

			logger.info('Starting Validation')

			value_ac = 0
			equation_ac = 0
			eval_total = 0
			start = time.time()

			with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
				f_out.write('---------------------------------------\n')
				f_out.close()

			ex_num = 0
			for test_batch in test_pairs:
				# test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
				# 						 merge, output_lang, test_batch[5], beam_size=config.beam_size)
				test_res = evaluate_tree(config, test_batch[0], test_batch[1], generate_num_ids, embedding, encoder, predict, generate,
										 merge, input_lang, output_lang, test_batch[5], beam_size=config.beam_size)
				val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])

				cur_result = 0
				if val_ac:
					value_ac += 1
					cur_result = 1
				if equ_ac:
					equation_ac += 1
				eval_total += 1

				with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
					f_out.write('Example: ' + str(ex_num) + '\n')
					f_out.write('Source: ' + stack_to_string(sentence_from_indexes(input_lang, test_batch[0])) + '\n')
					f_out.write('Target: ' + stack_to_string(sentence_from_indexes(output_lang, test_batch[2])) + '\n')
					f_out.write('Generated: ' + stack_to_string(sentence_from_indexes(output_lang, test_res)) + '\n')
					if config.nums_disp:
						src_nums = len(test_batch[4])
						tgt_nums = 0
						pred_nums = 0
						for k_tgt in sentence_from_indexes(output_lang, test_batch[2]):
							if k_tgt not in ['+', '-', '*', '/']:
								tgt_nums += 1
						for k_pred in sentence_from_indexes(output_lang, test_res):
							if k_pred not in ['+', '-', '*', '/']:
								pred_nums += 1
						f_out.write('Numbers in question: ' + str(src_nums) + '\n')
						f_out.write('Numbers in Target Equation: ' + str(tgt_nums) + '\n')
						f_out.write('Numbers in Predicted Equation: ' + str(pred_nums) + '\n')
					f_out.write('Result: ' + str(cur_result) + '\n' + '\n')
					f_out.close()

				ex_num+=1

			if float(value_ac) / eval_total > max_val_acc:
				max_value_corr = value_ac
				len_total_eval = eval_total
				max_val_acc = float(value_ac) / eval_total
				eq_acc = float(equation_ac) / eval_total

			od = OrderedDict()
			od['val_acc_epoch'] = float(value_ac) / eval_total
			od['equation_acc_epoch'] = float(equation_ac) / eval_total
			od['max_val_acc'] = max_val_acc
			od['equation_acc'] = eq_acc
			print_log(logger, od)

			logger.debug('Validation Completed...\nTime Taken: {}'.format(time_since(time.time() - start)))

			best_acc.append((max_value_corr, len_total_eval))

		total_value_corr = 0
		total_len = 0
		for w in range(len(best_acc)):
			# folds_scores.append(float(best_acc[w][0])/best_acc[w][1])
			total_value_corr += best_acc[w][0]
			total_len += best_acc[w][1]
		fold_acc_score = float(total_value_corr)/total_len

		# store_val_results(config, fold_acc_score, folds_scores)
		logger.info('Final Val score: {}'.format(fold_acc_score))

	else:
		if not os.path.exists(os.path.join(args.output_dir, result_folder)):
			os.makedirs(os.path.join(args.output_dir, result_folder))

		run_name = config.run_name

		config.log_path = os.path.join(args.output_dir, log_folder, run_name)
		config.model_path = os.path.join(args.output_dir, model_folder, run_name)
		config.board_path = os.path.join(args.output_dir, board_path, run_name)
		config.outputs_path = os.path.join(args.output_dir, outputs_folder, run_name)
		config.model_load_path = os.path.join(config.pretrained_dir, model_folder, run_name)

		for file in [config.log_path, config.model_path, config.board_path, config.outputs_path]:
			if not os.path.exists(file): os.makedirs(file)

		vocab1_path = os.path.join(config.model_path, 'vocab1.p')
		vocab2_path = os.path.join(config.model_path, 'vocab2.p')
		config_file = os.path.join(config.model_path, 'config.p')
		log_file = os.path.join(config.log_path, 'log.txt')

		if config.results:
			config.result_path = os.path.join(args.output_dir, result_folder, 'val_results_{}.json'.format(config.dataset))

		if is_train:
			create_save_directories(config.log_path)
			create_save_directories(config.model_path)
			create_save_directories(config.outputs_path)
		else:
			create_save_directories(config.log_path)
			create_save_directories(config.result_path)

		logger = get_logger(run_name, log_file, logging.DEBUG)

		logger.info('Experiment Name: {}'.format(config.run_name))
		logger.debug('Created Relevant Directories')

		logger.info('Loading Data...')

		train_ls, dev_ls = load_raw_data(data_path, config.dataset, is_train)
		pairs_trained, pairs_tested, generate_nums, copy_nums = transfer_num(train_ls, dev_ls, config.challenge_disp)

		logger.debug('Data Loaded...')
		if is_train:
			logger.debug('Number of Training Examples: {}'.format(len(pairs_trained)))
		logger.debug('Number of Testing Examples: {}'.format(len(pairs_tested)))
		logger.debug('Extra Numbers: {}'.format(generate_nums))
		logger.debug('Maximum Number of Numbers: {}'.format(copy_nums))

		# pairs: ([list of words in question], [list of infix Equation tokens incl brackets and N0, N1], [list of numbers], [list of indexes of numbers])
		# generate_nums: Unmentioned numbers used in eqns in atleast 5 examples ['1', '3.14']
		# copy_nums: Maximum number of numbers in a single sentence: 15

		# pairs: ([list of words in question], [list of prefix Equation tokens w/ metasymbols as N0, N1], [list of numbers], [list of indexes of numbers])

		if is_train:
			logger.info('Creating Vocab...')
			input_lang = None
			output_lang = None
		else:
			logger.info('Loading Vocab File...')

			with open(vocab1_path, 'rb') as f:
				input_lang = pickle.load(f)
			with open(vocab2_path, 'rb') as f:
				output_lang = pickle.load(f)

			logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, input_lang.n_words))

		input_lang, output_lang, train_pairs, test_pairs = prepare_data(config, logger, pairs_trained, pairs_tested, config.trim_threshold, generate_nums, copy_nums, input_lang, output_lang, tree=True)

		checkpoint = get_latest_checkpoint(config.model_path, logger)


		with open(vocab1_path, 'wb') as f:
			pickle.dump(input_lang, f, protocol=pickle.HIGHEST_PROTOCOL)
		with open(vocab2_path, 'wb') as f:
			pickle.dump(output_lang, f, protocol=pickle.HIGHEST_PROTOCOL)

		logger.debug('Vocab saved at {}'.format(vocab1_path))

		generate_num_ids = []
		for num in generate_nums:
			generate_num_ids.append(output_lang.word2index[num])

		config.len_generate_nums = len(generate_nums)
		config.copy_nums = copy_nums

		with open(config_file, 'wb') as f:
			pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

		logger.debug('Config File Saved')

		# train_pairs: ([list of token ids of question], len(ques), [list of token ids of equation], len(equation), [list of numbers], [list of indexes of numbers], [number stack])

		logger.info('Initializing Models...')

		# Initialize models
		embedding = None
		if config.embedding == 'bert':
			embedding = BertEncoder(config.emb_name, device, config.freeze_emb)
		elif config.embedding == 'roberta':
			embedding = RobertaEncoder(config.emb_name, device, config.freeze_emb)
		else:
			embedding = Embedding(config, input_lang, input_size=input_lang.n_words, embedding_size=config.embedding_size, dropout=config.dropout)

		# encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=config.embedding_size, hidden_size=config.hidden_size, n_layers=config.depth, dropout=config.dropout)
		encoder = EncoderSeq(cell_type=config.cell_type, embedding_size=config.embedding_size, hidden_size=config.hidden_size, n_layers=config.depth, dropout=config.dropout)
		predict = Prediction(hidden_size=config.hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums), input_size=len(generate_nums), dropout=config.dropout)
		generate = GenerateNode(hidden_size=config.hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums), embedding_size=config.embedding_size, dropout=config.dropout)
		merge = Merge(hidden_size=config.hidden_size, embedding_size=config.embedding_size, dropout=config.dropout)
		# the embedding layer is only for generated number embeddings, operators, and paddings

		logger.debug('Models Initialized')

		# load the pretrained initialization
		logger.info('loading pretrained models from {}.'.format(config.model_load_path))

		ckpt_path = os.path.join(config.model_load_path, 'model.pt')
		checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
		embedding.load_state_dict(checkpoint['embedding_state_dict'])
		encoder.load_state_dict(checkpoint['encoder_state_dict'])
		predict.load_state_dict(checkpoint['predict_state_dict'])
		generate.load_state_dict(checkpoint['generate_state_dict'])
		merge.load_state_dict(checkpoint['merge_state_dict'])


		logger.info('Initializing Optimizers...')

		embedding_optimizer = torch.optim.Adam(embedding.parameters(), lr=config.emb_lr, weight_decay=config.weight_decay)
		encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay)
		predict_optimizer = torch.optim.Adam(predict.parameters(), lr=config.lr, weight_decay=config.weight_decay)
		generate_optimizer = torch.optim.Adam(generate.parameters(), lr=config.lr, weight_decay=config.weight_decay)
		merge_optimizer = torch.optim.Adam(merge.parameters(), lr=config.lr, weight_decay=config.weight_decay)

		logger.debug('Optimizers Initialized')
		logger.info('Initializing Schedulers...')

		embedding_scheduler = torch.optim.lr_scheduler.StepLR(embedding_optimizer, step_size=20, gamma=0.5)
		encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
		predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
		generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
		merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)

		logger.debug('Schedulers Initialized')

		logger.info('Loading Models on GPU {}...'.format(config.gpu))

		# Move models to GPU
		if USE_CUDA:
			embedding.to(device)
			encoder.to(device)
			predict.to(device)
			generate.to(device)
			merge.to(device)

		logger.debug('Models loaded on GPU {}'.format(config.gpu))

		# generate_num_ids = []
		# for num in generate_nums:
		# 	generate_num_ids.append(output_lang.word2index[num])


		############## code for sparse #################

		# performing pruning at the first epoch
		mask = None
		if args.sparse:

			modules = [embedding, encoder, predict, generate, merge]
			input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(
				train_pairs, config.batch_size)

			decay = CosineDecay(args.prune_rate, int(args.epochs * len(output_lengths)))
			mask = Masking(None, prune_rate_decay=decay, prune_rate=args.prune_rate,
						   sparsity=args.sparsity, prune_mode=args.prune, growth_mode=args.growth,
						   redistribution_mode=args.redistribution, fp16=args.fp16, args=args)
			mask.add_module(modules)

			if mask.sparse_init == 'snip':
				mask.init_growth_prune_and_redist()

				embedding_copy, encoder_copy, predict_copy, generate_copy, merge_copy = copy.deepcopy(
					embedding), copy.deepcopy(encoder), copy.deepcopy(
					predict), copy.deepcopy(generate), copy.deepcopy(merge)
				embedding_copy.train()
				encoder_copy.train()
				predict_copy.train()
				generate_copy.train()
				merge_copy.train()
				modules_copy = [embedding_copy, encoder_copy, predict_copy, generate_copy, merge_copy]

				idx = 0
				loss_snip = train_tree(
					config, input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
					num_stack_batches[idx], num_size_batches[idx], generate_num_ids, embedding_copy, encoder_copy,
					predict_copy, generate_copy, merge_copy,
					embedding_optimizer, encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,
					input_lang, output_lang,
					num_pos_batches[idx], mask=None)
				# torch.nn.utils.clip_grad_norm_(need_optimized_parameters, args.max_grad_norm)

				grads_abs = []
				for module in modules_copy:
					for name, weight in module.named_parameters():
						if name not in mask.masks: continue
						grads_abs.append(torch.abs(weight * weight.grad))

				# Gather all scores in a single vector and normalise
				all_scores = torch.cat([torch.flatten(x) for x in grads_abs])

				num_params_to_keep = int(len(all_scores) * (1 - mask.sparsity))
				threshold, _ = torch.topk(all_scores, num_params_to_keep + 1, sorted=True)
				acceptable_score = threshold[-1]

				snip_masks = []
				for i, g in enumerate(grads_abs):
					mask_ = (g > acceptable_score).float()
					snip_masks.append(mask_)

				for snip_mask, name in zip(snip_masks, mask.masks):
					mask.masks[name] = snip_mask
			else:
				mask.init(model=modules, train_loader=None, device=mask.device, mode=mask.sparse_init,
						  density=(1 - args.sparsity))
			mask.apply_mask()
			mask.print_status()

		############## code for sparse #################

		max_value_corr = 0
		len_total_eval = 0
		max_val_acc = 0.0
		max_train_acc = 0.0
		eq_acc = 0.0
		best_epoch = -1
		min_train_loss = float('inf')

		logger.info('Starting Validation')

		value_ac = 0
		equation_ac = 0
		eval_total = 0
		start = time.time()

		with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
			f_out.write('---------------------------------------\n')
			f_out.close()

		ex_num = 0
		for test_batch in test_pairs:
			# test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
			# 						 merge, output_lang, test_batch[5], beam_size=config.beam_size)
			test_res = evaluate_tree(config, test_batch[0], test_batch[1], generate_num_ids, embedding, encoder, predict, generate,
									 merge, input_lang, output_lang, test_batch[5], beam_size=config.beam_size)
			val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])

			cur_result = 0
			if val_ac:
				value_ac += 1
				cur_result = 1
			if equ_ac:
				equation_ac += 1
			eval_total += 1

			with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
				f_out.write('Example: ' + str(ex_num) + '\n')
				f_out.write('Source: ' + stack_to_string(sentence_from_indexes(input_lang, test_batch[0])) + '\n')
				f_out.write('Target: ' + stack_to_string(sentence_from_indexes(output_lang, test_batch[2])) + '\n')
				f_out.write('Generated: ' + stack_to_string(sentence_from_indexes(output_lang, test_res)) + '\n')
				if config.challenge_disp:
					f_out.write('Type: ' + test_batch[7] + '\n')
					f_out.write('Variation Type: ' + test_batch[8] + '\n')
					f_out.write('Annotator: ' + test_batch[9] + '\n')
					f_out.write('Alternate: ' + str(test_batch[10]) + '\n')
				if config.nums_disp:
					src_nums = len(test_batch[4])
					tgt_nums = 0
					pred_nums = 0
					for k_tgt in sentence_from_indexes(output_lang, test_batch[2]):
						if k_tgt not in ['+', '-', '*', '/']:
							tgt_nums += 1
					for k_pred in sentence_from_indexes(output_lang, test_res):
						if k_pred not in ['+', '-', '*', '/']:
							pred_nums += 1
					f_out.write('Numbers in question: ' + str(src_nums) + '\n')
					f_out.write('Numbers in Target Equation: ' + str(tgt_nums) + '\n')
					f_out.write('Numbers in Predicted Equation: ' + str(pred_nums) + '\n')
				f_out.write('Result: ' + str(cur_result) + '\n' + '\n')
				f_out.close()

			ex_num+=1

		if float(value_ac) / eval_total > max_val_acc:
			max_val_acc = float(value_ac) / eval_total
			eq_acc = float(equation_ac) / eval_total

		od = OrderedDict()
		od['val_acc_epoch'] = float(value_ac) / eval_total
		od['equation_acc_epoch'] = float(equation_ac) / eval_total
		od['max_val_acc'] = max_val_acc
		od['equation_acc'] = eq_acc
		print_log(logger, od)

		logger.info('Final Val score: {}'.format(max_val_acc))


if __name__ == '__main__':
	main()

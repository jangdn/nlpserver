import argparse
import copy, json, os

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from models.model import BiDAF
from models.data import SQuAD
from models.ema import EMA
import evaluate
import numpy as np

def get_vis(args,data):
	device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
	model = BiDAF(args,data.WORD.vocab.vectors).to(device)
	print ("load Pretrained model")
	load_path = torch.load(args.load_path)
	model.load_state_dict(load_path)
	ema = EMA(args.exp_decay_rate)
	criterion = nn.CrossEntropyLoss()
	model.save_mode = True

	model.eval()
	def save_vis_data(train=True):
		save_data = []

		iterator = data.train_iter if train else data.dev_iter
		mode = 'trainData'if train else 'testData'
		print ('Mode :{}'.format(mode))
		save_count = 0
		with torch.no_grad() :
			count =0
			for i,batch in enumerate(iterator):
				present_epoch = int(iterator.epoch)
				if present_epoch == 1:
					break;
				tmp = {}
				p1,p2 = model(batch)
				p1 = p1.unsqueeze(0)
				p2 = p2.unsqueeze(0)
				c_words = [data.WORD.vocab.itos[i.item()] for i in batch.c_word[0][0]]
				q_words = [data.WORD.vocab.itos[i.item()] for i in batch.q_word[0][0]]
				batch_loss = criterion(p1,batch.s_idx) + criterion(p2,batch.e_idx)
				batch_size, c_len = p1.size()

				ls = nn.LogSoftmax(dim=1)
				mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)

				score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
				scores = score
				score, s_idx = score.max(dim=1)
				score, e_idx = score.max(dim=1)
				s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

				tmp['context'] = c_words
				tmp['question'] = q_words
				tmp['gt_s_idx'] = batch.s_idx.cpu().numpy()
				tmp['gt_e_idx'] = batch.e_idx.cpu().numpy()
				tmp['save_data'] = model.save_data.copy()
				tmp['loss'] = batch_loss.cpu().numpy()
				tmp['prediction_s_idx'] = s_idx.cpu().numpy()
				tmp['prediction_e_idx'] = e_idx.cpu().numpy()
				tmp['prediction_scores'] = scores.cpu().numpy()
				save_data.append(tmp)
				if len(save_data)%2000 ==0:
					np.save('{}_{}'.format(count,mode),save_data)
					save_data = []
					count +=1
			np.save('{}_{}'.format(count,mode),save_data)
	save_vis_data()
	save_vis_data(False)




def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--char-dim', default=8, type=int)
	parser.add_argument('--char-channel-width', default=5, type=int)
	parser.add_argument('--char-channel-size', default=100, type=int)
	parser.add_argument('--context-threshold', default=400, type=int)
	parser.add_argument('--dev-batch-size', default=1, type=int)
	parser.add_argument('--dev-file', default='dev-v1.1.json')
	parser.add_argument('--dropout', default=0.2, type=float)
	parser.add_argument('--epoch', default=1, type=int)
	parser.add_argument('--exp-decay-rate', default=0.999, type=float)
	parser.add_argument('--gpu', default=0, type=int)
	parser.add_argument('--hidden-size', default=100, type=int)
	parser.add_argument('--learning-rate', default=0.5, type=float)
	parser.add_argument('--print-freq', default=250, type=int)
	parser.add_argument('--train-batch-size', default=1, type=int)
	parser.add_argument('--train-file', default='train-v1.1.json')
	parser.add_argument('--word-dim', default=100, type=int)

	parser.add_argument('--load_path',default=None ,type = str)
	args = parser.parse_args()

	print('loading SQuAD data...')
	data = SQuAD(args)
	setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
	setattr(args, 'word_vocab_size', len(data.WORD.vocab))
	setattr(args, 'dataset_file', f'./data/squad/{args.dev_file}')
	setattr(args, 'prediction_file', f'prediction{args.gpu}.out')
	setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
	print('data loading complete!')

	print('Get visualization start!')
	best_model = get_vis(args, data)


if __name__ == '__main__':
	main()

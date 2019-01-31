import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import LSTM,Linear

class BiDAF(nn.Module):
	def __init__(self,args,pretrained):
		super(BiDAF,self).__init__()
		self.args = args
		# 1. Character Embedding Layer
		self.char_emb = nn.Embedding(args.char_vocab_size,args.char_dim,padding_idx = 1)
		nn.init.uniform_(self.char_emb.weight,-0.001,0.001)

		self.char_conv = nn.Conv2d(1,args.char_channel_size ,(args.char_dim,args.char_channel_width))

		# 2.Word Embedding Layer
		# initilize word embedding with glove vector

		self.word_emb = nn.Embedding.from_pretrained(pretrained,freeze=True)

		assert self.args.hidden_size *2 == (self.args.char_channel_size + self.args.word_dim)

		# highway network
		for i in range(2):
			setattr(self,f'highway_linear{i}',
					nn.Sequential(Linear(args.hidden_size*2 , args.hidden_size *2),
								nn.ReLU()))
			setattr(self,f'highway_gate{i}',
					nn.Sequential(Linear(args.hidden_size*2,args.hidden_size*2),
								nn.Sigmoid()))

		# 3. Context Embedding Layer
		self.context_LSTM = LSTM(input_size=args.hidden_size*2,
								hidden_size=args.hidden_size,
								bidirectional=True,
								batch_first=True,
								dropout = args.dropout)
		# 4. Attention Flow Layer
		self.attn_weight_c = Linear(args.hidden_size*2,1)
		self.attn_weight_q = Linear(args.hidden_size*2,1)
		self.attn_weight_cq = Linear(args.hidden_size*2,1)

		# 5. Modeling Layer
		self.modelinng_LSTM1 = LSTM(input_size=args.hidden_size *8,
									hidden_size = args.hidden_size,
									bidirectional= True ,
									batch_first = True,
									dropout= args.dropout)
		self.modelinng_LSTM2 = LSTM(input_size = args.hidden_size *2,
									hidden_size=args.hidden_size,
									bidirectional= True,
									batch_first =True,
									dropout = args.dropout)
		self.p1_weight_g = Linear(args.hidden_size*8,1,dropout=args.dropout)
		self.p1_weight_m = Linear(args.hidden_size*2,1,dropout=args.dropout)
		self.p2_weight_g = Linear(args.hidden_size*8,1,dropout=args.dropout)
		self.p2_weight_m = Linear(args.hidden_size*2,1,dropout=args.dropout)

		self.output_LSTM =  LSTM(input_size=args.hidden_size*2,
								hidden_size=args.hidden_size,
								bidirectional=True ,
								batch_first=True ,
								dropout =args.dropout)
		self.dropout = nn.Dropout(p=args.dropout)
		self.save_mode = False
		self.save_data = {}

	def forward(self,batch):
		#print('c_word[0] type : ', type(batch.c_word[0]))
		#print('c_word type : ', type(batch.c_word))
		#print('c_word[0] size : ', batch.c_word[0].size())
		#print('q_word[0] size : ', batch.q_word[0].size())

		# charcter embedding_layer
		c_char = self.char_emb_layer(batch.c_char)
		q_char = self.char_emb_layer(batch.q_char)
		c_word = self.word_emb(batch.c_word[0])
		q_word = self.word_emb(batch.q_word[0])

		if self.save_mode:
			self.save_data['c_input'] = batch.c_char.cpu().numpy()
			self.save_data['q_input'] = batch.q_char.cpu().numpy()

			self.save_data['c_char'] = c_char.cpu().numpy()
			self.save_data['q_char'] = q_char.cpu().numpy()
			self.save_data['q_word'] = q_word.cpu().numpy()
			self.save_data['c_word'] = c_word.cpu().numpy()

		c_lens = batch.c_word[1]
		q_lens = batch.q_word[1]

		c = self.highway_network(c_char,c_word)
		q = self.highway_network(q_char,q_word)
		if self.save_mode:
			self.save_data['highway_c'] = c.cpu().numpy()
			self.save_data['highway_q'] = q.cpu().numpy()


		c = self.context_LSTM((c,c_lens))[0]
		q = self.context_LSTM((q,q_lens))[0]
		if self.save_mode:
			self.save_data['context_c'] = c.cpu().numpy()
			self.save_data['context_q'] = q.cpu().numpy()


		g = self.att_flow_layer(c,q)

		m = self.modelinng_LSTM2((self.modelinng_LSTM1((g,c_lens))[0],c_lens))[0]
		p1,p2 = self.output_layer(g,m,c_lens)
		if self.save_mode:
			self.save_data['g'] = g.cpu().numpy()
			self.save_data['m'] = m.cpu().numpy()
			self.save_data['p1'] = p1.cpu().numpy()
			self.save_data['p2'] = p2.cpu().numpy()

		return p1,p2

	def char_emb_layer(self,x):
		batch_size = x.size(0)
		# (b,seq_len,word_len,char_dim)
		x= self.dropout(self.char_emb(x))
		# (b*seq_len,1,char_dim,word_len)
		x= x.view(-1,self.args.char_dim,x.size(2)).unsqueeze(1)
		# (b*T,filter_nums,1,conv_len) -> (b*seq_len,filter_nums,conv_len)
		x = self.char_conv(x).squeeze(2)
		# (batch*seq_len,filter_nums,1) -> (batch*seq_len,filter_nums) stride is full

		x = F.max_pool1d(x,x.size(2)).squeeze(2)
		x = x.view(batch_size,-1,x.size(1))
		return x
	def highway_network(self,x1,x2):

		"""
			x1 : char embedding
			x2 : glove embedding

		"""
		x = torch.cat([x1,x2],dim=-1)
		for i in range(2):
			h = getattr(self,f'highway_linear{i}')(x)
			g = getattr(self,f'highway_gate{i}')(x)
			x= g*h + (1-g)*x

		return x
	def att_flow_layer(self,c,q):
		"""
			inputs
				c: (batch,c_len , hidden_size *2)
				q: (batch,q_len,hidden_size *2)
			returns
				(batch,c_len,q_len)
		"""
		c_len = c.size(1)
		q_len = q.size(1)
		cq = []
		for i in range(q_len):
			#(batch_size , 1,hidden_size *2)
			qi = q.select(1,i).unsqueeze(1)
			#(batch_size,c_len,1)
			ci = self.attn_weight_cq(c*qi).squeeze()
			cq.append(ci)
		# (batch,c_len,q_len)
		cq = torch.stack(cq,dim=-1)

		# (batch,c_len,q_len)
		s = self.attn_weight_c(c).expand(-1,-1,q_len) +\
			self.attn_weight_q(q).permute(0,2,1).expand(-1,c_len,-1) + \
			cq


		# (batch,c_len,q_len)
		a = F.softmax(s,dim=2) # attention score

		c2q_attn = torch.bmm(a,q) # (batch,c_len,q_len) -> (batch,c_len,hidden *2)

		# (batch,1,c_len)

		b = F.softmax(torch.max(s,dim=2)[0],dim=1).unsqueeze(1)
		# (batch,1,c_len) * (batch,c_len,hidden_size) -> (batch,)
		q2c_att = torch.bmm(b,c).squeeze(1)
		q2c_att = q2c_att.unsqueeze(1).expand(-1,c_len,-1)

		if self.save_mode:
			self.save_data['c2q_attn'] = a.cpu().numpy()
			self.save_data['q2c_attn'] = b.cpu().numpy()
		x = torch.cat([c,c2q_attn,c*c2q_attn,c*q2c_att],dim=-1)
		return x

	def output_layer(self,g,m,l):
		"""
		inputs:
			g : (batch,c_len,hidden_size * 8)
			m : (batch,c_len,hidden_size * 2)
		returns:
			p1 (batch,c_len)
			p2 (batch,c_len)

		"""
		# (batch,c_len)
		p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
		# (batch,c_len )
		m2 = self.output_LSTM((m,l))[0]

		p2 = (self.p2_weight_g(g)+ self.p2_weight_m(m2)).squeeze()

		return p1,p2

import numpy as np
import os
import sys
from pkl import load_training, load_testing
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import itertools

XVALID = False

np.random.seed(1234)

class Sigmoid(object):
	@staticmethod
	def f(x):
		return 1/(1+np.exp(-x))
	@staticmethod
	def d(x):
		return x*(1-x)

class Tanh(object):
	@staticmethod
	def f(x):
		return np.tanh(x)

	@staticmethod
	def d(x):
		return 1 - x**2

class RELU(object):
	@staticmethod
	def f(x):
		return np.max([0, x])
	def d(x):
		return (x + np.abs(x))/2*x

class MLP(object):
	def __init__(self, input_size=None, 
						layer_size=None, 
						loss = "xent", 
						activation_fxn = "sigmoid", 
						output_size=None, 
						n_layers=None, 
						alpha=None, 
						epochs=None, 
						batch_size=None, 
						gamma=None,
						add_noise=False,
						clip=None):
		super(MLP, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.layer_size = layer_size
		self.n_layers = n_layers
		self.alpha = alpha
		self.epochs = epochs
		self.batch_size = batch_size
		self.loss = loss
		self.clip=clip

		self.gamma = gamma
		self.add_noise=add_noise
		if activation_fxn == "sigmoid":
			self.activation = Sigmoid
		elif activation_fxn == "tanh":
			self.activation = Tanh
		elif activation_fxn == "relu":
			self.activation = RELU
		else:
			print("Error: invalid activation fxn")
			sys.exit()

		self.vec_f = np.vectorize(self.activation.f)
		self.vec_d = np.vectorize(self.activation.d)

		self.weights, self.biases= [], []

		W1 = np.random.normal(size=(self.input_size, self.layer_size), loc=0, scale=0.05)
		self.weights.append(W1)
		b1 = np.zeros((1, self.layer_size))
		self.biases.append(b1)

		for i in range(n_layers-1):
			Wn = np.random.normal(size=(self.layer_size, self.layer_size), loc=0, scale=0.05)
			self.weights.append(Wn)
			bn = np.zeros((1, self.layer_size))
			self.biases.append(bn)


		Wout = np.random.normal(size=(self.layer_size, self.output_size), loc=0, scale=0.05)
		self.weights.append(Wout)
		bout = np.zeros((1, self.output_size))
		self.biases.append(bout)


	def to_single(self, y):
		new_y =[np.argmax(row) for row in y]
		return new_y

	def forward(self, x):
		outputs, derivatives = [],[]
		Z1 = np.dot(x, self.weights[0])  + self.biases[0]
		
		H1 = self.vec_f(Z1)
		dH1 = self.vec_d(H1)
		outputs.append(H1)
		derivatives.append(dH1)
		for i in range(self.n_layers):
			Zn = np.dot(outputs[-1], self.weights[i+1])  + self.biases[i+1]
			Hn = self.vec_f(Zn) 
			dHn = self.vec_d(Zn)
			outputs.append(Hn)
			derivatives.append(dHn)
		# softmax output 

		exp_out = np.exp(outputs[-1])
		norm = np.sum(exp_out, axis=1, keepdims=True)

		softmax = exp_out/norm
		outputs[-1] = softmax
		return outputs, derivatives

	def backprop(self, x, y, outputs, derivatives):
		w_grads, b_grads, deltas = [], [], []
		softmax = outputs[-1]
		outputs = [x] + outputs
		H1 = outputs[1]
		dH1 = derivatives[0]

		delta_y = softmax - y
		deltas.append(delta_y)
		dE_dOut = np.dot(H1.T, delta_y)
		w_grads.append(dE_dOut.T)
		dE_dB2 = np.sum(delta_y, axis=0)/x.shape[0]
		b_grads.append(dE_dB2)

		for i in range(self.n_layers-1, -1, -1):
			delta_i = np.multiply(np.dot( deltas[-1], self.weights[i+1].T), derivatives[i]) 
			dE_dWi = np.dot(delta_i.T, outputs[i])
			dE_dBi = np.sum(delta_i, axis=0)/x.shape[0]

			w_grads.append(dE_dWi.T)
			b_grads.append(dE_dBi)
			deltas.append(delta_i)

		w_grads.reverse()
		b_grads.reverse()

		# clip
		if self.clip is not None:
			for i in range(len(w_grads)):
				w_grads[i][w_grads[i]>self.clip[1]]=self.clip[1]
				b_grads[i][b_grads[i]>self.clip[1]]=self.clip[1]
				w_grads[i][w_grads[i]<self.clip[0]]=self.clip[0]
				b_grads[i][b_grads[i]<self.clip[0]]=self.clip[0]
		return w_grads, b_grads

	def train(self, x_train, y_train, x_valid, y_valid):
		for i in range(self.epochs):
			x_batches, y_batches = [],[]
			num_batch = int(x_train.shape[0]/128)+1
			idx = 0
			for i in range(num_batch):
				x_curr_batch, y_curr_batch = [], []
				for j in range(128):
					try:
						x_curr_batch.append(x_train[idx+j])
						y_curr_batch.append(y_train[idx+j])
					except IndexError:
						break
				idx+=128
				if len(x_curr_batch) > 0:
					x_batches.append(np.array(x_curr_batch))
					y_batches.append(np.array(y_curr_batch))

			x_batches = [xb for xb in x_batches if len(xb) >0]
			y_batches = [yb for yb in y_batches if len(yb) > 0]
			zipped_batches = list(zip(x_batches, y_batches))
			for j in tqdm(range(len(x_batches))):
				x, y = zipped_batches[j]

				outputs, derivatives = self.forward(x)

				w_grads, b_grads = self.backprop(x, y, outputs, derivatives)
				for i in range(self.n_layers):

					self.weights[i] -= self.alpha*w_grads[i]
					self.biases[i] -= self.alpha*b_grads[i]

			v_outputs, _ = self.forward(x_valid)
			v_smax = v_outputs[-1]
			valid_acc = accuracy_score(self.to_single(y_valid), self.to_single(v_smax))
			print("valid_acc:", valid_acc)

x_full, y_full = load_training("../../data", False) 
x_for_test = load_testing("../../data")

x_full, y_full = x_full[:50000], y_full[:50000]
x_full= x_full.reshape(-1, 4096)

y_full_vec = np.zeros((y_full.shape[0], 10))
for i, y in enumerate(y_full):
	y_full_vec[i][y] += 1

y_full = y_full_vec

full_zip = list(zip(x_full, y_full))
np.random.shuffle(full_zip)
x_full, y_full = zip(*full_zip)
x_full = np.array(x_full)
y_full = np.array(y_full)
x_for_test = np.array(x_for_test).reshape(-1, 4096)

if not XVALID:
	print("TESTING MLP")
	x_train = x_full[:int(.7*len(x_full))]
	y_train = y_full[:int(.7*len(y_full))]
	print("x_train: ", x_train.shape)
	print("y ", y_train.shape)
	x_valid = x_full[int(.7*len(x_full)):int(.8*len(x_full)):]
	y_valid = y_full[int(.7*len(y_full)):int(.8*len(y_full)):]
	x_test = x_full[int(.8*len(x_full)):]
	y_test = y_full[int(.8*len(y_full)):]

	mlp = MLP( input_size=x_train.shape[1], 
				layer_size=512, 
				loss = "xent", 
				activation_fxn = "tanh", 
				output_size=y_train.shape[1], 
				n_layers=1, 
				alpha=.01, 
				epochs=10, 
				batch_size=128, 
				gamma=None,
				add_noise=False)

	mlp.train(x_train, y_train, x_valid, y_valid)
	output, _ =mlp.forward(x_test)
	prediction = output[-1]
	test_acc = accuracy_score(mlp.to_single(y_test), mlp.to_single(prediction))
	print("TEST ACCURACY:", test_acc)

	with open("../../output/test_predictions_ffnn.csv", "w") as f1:
		f1.write("ID,LABEL\n")
		test_output, _ = mlp.forward(x_for_test)
		test_pred = output[-1]
		for i,p in enumerate(test_pred):
			idx = np.argmax(p)
			f1.write("{},{}\n".format(i,idx))


if XVALID:
	x_train, y_train = x_full, y_full
	layer_sizes = [128, 256, 512]
	alphas = [0.001, 0.01, 0.1]
	layer_nums = [1,2,3]
	for layer_size, alpha, lay_num in itertools.product(layer_sizes, alphas, layer_nums): 
		print("running layer_size: {}, alpha: {}, layer_num: {}".format(layer_size, alpha, lay_num))
		valid_accs = []
		with open("{}-lay-acc.txt".format(lay_num), 'a') as f1:
			for cv_fold in range(5):
				if lay_num > 1:
					clip = [-100, 100]
				else:
					clip = None
				mlp = MLP( input_size=x_train.shape[1], 
										layer_size=layer_size, 
										loss = "xent", 
										activation_fxn = "tanh", 
										output_size=y_train.shape[1], 
										n_layers=lay_num, 
										alpha=alpha, 
										epochs=10, 
										batch_size=128, 
										gamma=None,
										add_noise=False,
										clip=clip)

				full_len = x_train.shape[0]
				valid_mask = np.arange(int(cv_fold*.2*full_len), int((cv_fold+1)*.2*full_len), 1, dtype=np.int)
				train_mask = np.array([x for x in np.arange(0,full_len,1, dtype=np.int) if x < valid_mask[0] or x > valid_mask[-1]])
				
				X = x_train[train_mask]
				Y = y_train[train_mask]
				x_valid = x_train[valid_mask]
				y_valid = y_train[valid_mask]

				mlp.train(X, Y, x_valid, y_valid)
				output, _ =mlp.forward(x_valid)
				prediction = output[-1]
				valid_acc = accuracy_score(mlp.to_single(y_valid), mlp.to_single(prediction))
				valid_accs.append(valid_acc)

			# average validation accs:
			v_acc_avg = np.sum(valid_accs)/len(valid_accs)
			f1.write("{},{},{}\n".format(layer_size, alpha, v_acc_avg))



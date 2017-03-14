import torch
from torch.autograd import Variable
import scipy.io as sio

def loadSavedMatFile(x):
	data = sio.loadmat(x)
	# populate each column of array 	#convert from numpy to torchTensor
	base_in   = Variable(torch.from_numpy(data['base_in']))
	base_out  = Variable(torch.from_numpy(data['base_out']))
	left_in   = Variable(torch.from_numpy(data['left_in']))
	left_out  = Variable(torch.from_numpy(data['left_out']))
	right_in  = Variable(torch.from_numpy(data['right_out']))
	right_out = Variable(torch.from_numpy(data['right_out']))
	x     	  = Variable(torch.from_numpy(data['x']))
	y     	  = Variable(torch.from_numpy(data['y']))
	z     	  = Variable(torch.from_numpy(data['z']))
	pitch     = Variable(torch.from_numpy(data['pitch']))
	yaw   	  = Variable(torch.from_numpy(data['yaw']))

	return 	base_in, base_out, left_in, left_out, right_in, right_out, x, y, z, pitch, yaw

def split_data(x):
	base_in, base_out, left_in, left_out, right_in, right_out, x, y, z, pitch, yaw = loadSavedMatFile(x)
	inputs = torch.cat((base_in, base_out, left_in, 
						left_out, right_in, right_out, 
						z, pitch, yaw), 1)
	outputs = torch.cat((z, pitch, yaw), 1)

	N = int(inputs.size(0))

	nTrain = int(N*(1.-0.1))
	nTest  = N - nTrain
	# print('outputs: \n', base_in[0:int(k)])
	train_in = inputs[:nTrain]
	train_out = outputs[:nTrain]
	test_in = inputs[nTrain:]
	test_out = inputs[nTrain:]

	base_idx = torch.LongTensor([1])
	# print(inputs.narrow(0, 0, base_in.size(0)))

	return train_in, train_out, test_in, test_out
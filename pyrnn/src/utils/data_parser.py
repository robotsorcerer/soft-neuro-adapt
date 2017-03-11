import torch
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
	inputs = torch.cat((base_in, base_out, left_in, left_out, right_in, right_out), 0)
	outputs = torch.cat((z, pitch, yaw), 0)

	k = int(0.8*base_in.size(0))
	# print('outputs: \n', base_in[0:int(k)])
	train_in = torch.cat((base_in[0:k], base_out[0:k], left_in[0:k], left_out[0:k], right_in[0:k], right_out[0:k]), 0)
	train_out = torch.cat((z[0:k], pitch[0:k], yaw[0:k]), 0)

	test_in = torch.cat((base_in[k:], base_out[k:], left_in[k:], left_out[k:], right_in[k:], right_out[k:]), 0)
	train_out = torch.cat((z[k:], pitch[k:], yaw[k:]), 0)

	return train_in, train_out, test_in, test_out
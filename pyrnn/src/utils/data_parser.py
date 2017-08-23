import torch
import time
from torch.autograd import Variable
import scipy.io as sio
import gzip
import bz2
import csv
from random import  shuffle

torch.set_default_tensor_type('torch.DoubleTensor')

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

def loadSavedCSVFile(x):
	inputs = { 'ro': list(),
			   'lo': list(),
			   'li': list(),
			   'ri': list(),
			   'bi': list(),
			   'bo': list(),
			}

	outputs = {'roll': list(),
				'z': list(),
				'pitch': list(),
		   	}

	with gzip.GzipFile(x) as f:
		reader = csv.reader(f.readlines(), delimiter="\t")
		for row in reader:
			inputs['ro'].append(float(row[0]))
			inputs['lo'].append(float(row[1]))
			inputs['li'].append(float(row[2]))
			inputs['ri'].append(float(row[3]))
			inputs['bi'].append(float(row[4]))
			inputs['bo'].append(float(row[5]))
			outputs['roll'].append(float(row[7]))
			outputs['z'].append(float(row[8]))
			outputs['pitch'].append(float(row[9]))
	return inputs, outputs

def split_csv_data(x):

	# dictionaries to return

	train = {
		'in': None,
		'out': None,
	}

	test = {
		'in': None,
		'out': None,
	}

	in_dict, out_dict = loadSavedCSVFile(x)

	li = in_dict['li'];		lo = in_dict['lo'];		bi = in_dict['bi']
	bo = in_dict['bo'];		ri = in_dict['ri'];		ro = in_dict['ro']

	roll = out_dict['roll'];	z = out_dict['z'];	pitch = out_dict['pitch']

	'''
	past inputs will be observed at x time steps in the future
	so we observe the outputs by x time steps later. I chose 2 based on the
	observation during the experiment
	'''
	data_length = len(in_dict['ro'])
	############################################################
	#remove spurious data from last x time steps
	delay = 10
	###########################################################
	# neglect last delayed obeservations and controls
	idx_stop = data_length - delay
	temp = list(zip(li[delay:idx_stop], lo[delay:idx_stop], bi[delay:idx_stop],
			   bo[delay:idx_stop], ri[delay:idx_stop], ro[delay:idx_stop],
			   roll[delay:idx_stop], z[delay:idx_stop], pitch[delay:idx_stop])	)
	shuffle(temp)

	li_tuple, lo_tuple, bi_tuple, bo_tuple, ri_tuple, ro_tuple, roll_tuple, z_tuple, pitch_tuple = zip(*temp)
	del temp[:]
	li = list(li_tuple);		lo = list(lo_tuple);		bi   = list(bi_tuple);			bo    = list(bo_tuple)
	ri = list(ri_tuple);		ro = list(ro_tuple);		roll = list(roll_tuple); 		pitch = list(pitch_tuple)
	z  = list(z_tuple);

	li_tensor = Variable(torch.Tensor(li).unsqueeze(0).t())
	lo_tensor = Variable(torch.Tensor(lo).unsqueeze(0).t())
	bi_tensor = Variable(torch.Tensor(bi).unsqueeze(0).t())
	bo_tensor = Variable(torch.Tensor(bo).unsqueeze(0).t())
	ri_tensor = Variable(torch.Tensor(ri).unsqueeze(0).t())
	ro_tensor = Variable(torch.Tensor(ro).unsqueeze(0).t())

	roll_tensor	=  Variable(torch.Tensor(roll).unsqueeze(0).t())
	z_tensor		=  Variable(torch.Tensor(z).unsqueeze(0).t())
	pitch_tensor	=  Variable(torch.Tensor(pitch).unsqueeze(0).t())

	# find train size
	train_size = int(0.8*li_tensor.size(0))
	test_size = 1 - train_size

	"""
	Input to neural network is of the following sort
	X = [ y(t-1)	y(t) 	y(t-2)	y(t-3) u(t-1) 	u(t)	u(t-2)	u(t-3) ]
		[ y(t-2)	y(t-1)	y(t) 	y(t-3) u(t-2)	u(t)	u(t-1)	u(t-3) ]
		[ y(t)		y(t-3)	y(t-2)	y(t-1) u(t-3)	u(t)	u(t-2)	u(t-1) ]
	Inputs will be of size 54,285 x 36
	Outputs will be of size 54,285 x 3
	"""

	inputs = torch.cat((
						li_tensor, lo_tensor, bi_tensor,
						bo_tensor, ri_tensor, ro_tensor,
						roll_tensor, z_tensor, pitch_tensor,
						 ), 1)

	# outputs = torch.cat((roll_tensor[:min_length], z_tensor[:min_length], pitch_tensor[:min_length]), 1)
	outputs = torch.cat((
						li_tensor, lo_tensor, bi_tensor,
						bo_tensor, ri_tensor, ro_tensor), 1)

	# print(inputs.size(), outputs.size())
	train['in'] = inputs[:train_size]
	train['out'] = outputs[:train_size]

	test['in'] = inputs[train_size:]
	test['out'] = outputs[train_size:]

	return train, test#, inputs, outputs


def split_mat_data(x):
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

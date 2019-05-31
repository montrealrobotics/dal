from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import Config
import numpy as np
import pdb
from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli
from a2c_ppo_acktr.utils import init
from utils import init_normc_


def normalized_columns_initializer(weights, std=1.0):
	out = torch.randn(weights.size())
	out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
	return out

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		weight_shape = list(m.weight.data.size())
		fan_in = np.prod(weight_shape[1:4])
		fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
		w_bound = np.sqrt(6. / (fan_in + fan_out))
		m.weight.data.uniform_(-w_bound, w_bound)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		weight_shape = list(m.weight.data.size())
		fan_in = weight_shape[1]
		fan_out = weight_shape[0]
		w_bound = np.sqrt(6. / (fan_in + fan_out))
		m.weight.data.uniform_(-w_bound, w_bound)
		m.bias.data.fill_(0)

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)


class perceptual_conv_l0(nn.Module):
    def __init__(self, layers):
        super(perceptual_conv_l0, self).__init__()
        
        self.layers = layers

        n_channel_in1 = 5
        n_channel_out1 = 8
        n_pool1 = 2

        n_channel_in2 = 8
        n_channel_out2 = 16
        n_pool2 = 2

        n_pool3 = 2

        if self.layers == 3:
            n_channel_in3 = 16
            n_channel_out3 = 4

        elif self.layers == 4:
            n_channel_in3 = 16
            n_channel_out3 = 8
            n_channel_in4 = 8
            n_channel_out4 = 4

        elif self.layers == 5:
            n_channel_in2 = 8
            n_channel_out2 = 32
            n_channel_in3 = 32
            n_channel_out3 = 16
            n_channel_in4 = 16
            n_channel_out4 = 8
            n_channel_in5 = 8
            n_channel_out5 = 4 
        
        else:
            print("invalid layers")

        kernel_size = 3
        stride = 1
        padding = 1

        self.conv1 = nn.Conv2d(n_channel_in1, n_channel_out1, kernel_size, stride, padding) #Should I add bias=False?
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = n_pool1)

        self.conv2 = nn.Conv2d(n_channel_in2, n_channel_out2, kernel_size, stride, padding)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = n_pool2)

        self.conv3 = nn.Conv2d(n_channel_in3, n_channel_out3, kernel_size, stride, padding)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = n_pool3)

        if self.layers >= 4:
            self.conv4 = nn.Conv2d(n_channel_in4, n_channel_out4, kernel_size, stride, padding)
            self.relu4 = nn.ReLU()

        if self.layers == 5:
            self.conv5 = nn.Conv2d(n_channel_in5, n_channel_out5, kernel_size, stride, padding)
            self.relu5 = nn.ReLU()
        
        # self.act1 = nn.Tanh()
        # self.act2 = nn.Softmax(dim=2)

    def forward(self, x):

        x = Variable(x.float())
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        if self.layers == 3:
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.pool3(x)
        
        elif self.layers == 4:
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.pool3(x)
            x = self.conv4(x)
            x = self.relu4(x)
        
        elif self.layers == 5:
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.pool3(x)
            x = self.conv4(x)
            x = self.relu4(x)
            x = self.conv5(x)
            x = self.relu5(x)
        
        else:
            print("invalid layers")


        return x


class perceptual_conv_real_l1(nn.Module):
    def __init__(self, layers):
        super(perceptual_conv_real_l1, self).__init__()

        self.layers = layers

        n_channel_in1 = 5
        n_channel_out1 = 8
        n_pool1 = 2

        n_channel_in2 = 8
        n_channel_out2 = 16
        n_pool2 = 2
        n_pool3 = 2

        if self.layers == 3:
            n_channel_in3 = 16
            n_channel_out3 = 1
            

        if self.layers == 4:
            n_channel_in3 = 16
            n_channel_out3 = 8

            n_channel_in4 = 8
            n_channel_out4 = 1
            
        if self.layers == 5:
            n_channel_in3 = 16
            n_channel_out3 = 32

            n_channel_in4 = 32
            n_channel_out4 = 8

            n_channel_in5 = 8
            n_channel_out5 = 1

        
        kernel_size = 3
        stride = 1
        padding = 1

        self.conv1 = nn.Conv2d(n_channel_in1, n_channel_out1, kernel_size, stride, padding)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = n_pool1)

        self.conv2 = nn.Conv2d(n_channel_in2, n_channel_out2, kernel_size, stride, padding)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = n_pool2)

        self.conv3 = nn.Conv2d(n_channel_in3, n_channel_out3, kernel_size, stride, padding)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = n_pool3)

        if self.layers == 4:
            self.conv4 = nn.Conv2d(n_channel_in4, n_channel_out4, kernel_size, stride, padding)
            self.relu4 = nn.ReLU()

        if self.layers == 5:
            self.conv3 = nn.Conv2d(n_channel_in3, n_channel_out3, kernel_size, stride, padding)
            self.relu3 = nn.ReLU()
            self.conv4 = nn.Conv2d(n_channel_in4, n_channel_out4, kernel_size, stride, padding)
            self.relu4 = nn.ReLU()
            self.conv5 = nn.Conv2d(n_channel_in5, n_channel_out5, kernel_size, stride, padding)
            self.relu5 = nn.ReLU()

    def forward(self, x):

        x = Variable(x.float())
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        # x = self.pool3(x)

        if self.layers >= 4:
            x = self.conv4(x)
            x = self.relu4(x)

        if self.layers == 5:
            x = self.conv5(x)
            x = self.relu5(x)
 
        return x

class perceptual_laserscan_fc(nn.Module):
	def __init__(self,
			d_in = Config.map_width * Config.map_height + Config.laser_scan_dim,
			h1 = Config.h1,
			h2 = Config.h2,
			h3 = Config.h3,
			h4 = Config.h4,
			d_out = Config.num_orientations * Config.map_width * Config.map_height):

		super(perceptual_laserscan_fc, self).__init__()

		self.linear1 = nn.Linear(d_in, h1)
		self.linear2 = nn.Linear(h1, h2)
		self.linear3 = nn.Linear(h2, h3)
		self.linear4 = nn.Linear(h3, h4)
		self.linear5 = nn.Linear(h4, d_out)

	def forward(self, x):
		x = Variable(x.float())
		h1_relu = F.relu(self.linear1(x))
		h2_relu = F.relu(self.linear2(h1_relu))
		h3_relu = F.relu(self.linear3(h2_relu))
		h4_relu = F.relu(self.linear4(h3_relu))
		likli_out = self.linear5(h4_relu)       ## May be, use Tanh activation here, or a sigmoid.
		return likli_out


class perceptual_laserscan_conv(nn.Module):
	def __init__(self):
		super(perceptual_laserscan_conv, self).__init__()
		n_conv0_filters = 2
		n_conv1_filters = 8
		n_conv2_filters = 16
		n_conv3_filters = 32
		n_conv4_filters = 4
		kernel_size = 3
		stride = 1
		padding = 1

		self.conv1 = nn.Conv2d(n_conv0_filters, n_conv1_filters, kernel_size, stride, padding) #Should I add bias=False?
		self.conv2 = nn.Conv2d(n_conv1_filters, n_conv2_filters, kernel_size, stride, padding)
		self.conv3 = nn.Conv2d(n_conv2_filters, n_conv3_filters, kernel_size, stride, padding)
		self.conv4 = nn.Conv2d(n_conv3_filters, n_conv4_filters, kernel_size, stride, padding)

	def forward(self, x):
		#x = x.unsqueeze(0)
		x = Variable(x.float())
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		return F.softmax(x)     #Should we add dim=0/1? I think no. check. #or, should it be logsoftmax?

class perceptual_conv_fc(nn.Module):
	def __init__(self, map_size, n_fc1_out = 8, kernel_size = 3, use_softmax = True):
		super(perceptual_conv_fc, self).__init__()
		# input image : 176 x 176
		n_channel_in1 = 2
		n_channel_out1 = 4
		n_pool1 = 2
		n_channel_in2 = 4
		n_channel_out2 = 8
		n_pool2 = 2
		n_channel_in3 = 8
		n_channel_out3 = 16
		n_pool3 = 2
		n_channel_in4 = 16
		n_channel_out4 = 32

		n_pool4 = 2

		#n_fc1_out = 8
		n_fc2_out = 4
		#kernel_size = 3
		stride = 1
		padding = (kernel_size-1)/2
		#padding = 1

		self.map_size = map_size
		self.n_channel_out4=n_channel_out4

		self.conv1 = nn.Conv2d(n_channel_in1, n_channel_out1, kernel_size, stride, padding) #Should I add bias=False?
		self.bn1 = nn.BatchNorm2d(n_channel_out1)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size = n_pool1)

		self.conv2 = nn.Conv2d(n_channel_in2, n_channel_out2, kernel_size, stride, padding)
		self.bn2 = nn.BatchNorm2d(n_channel_out2)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(kernel_size = n_pool2)

		self.conv3 = nn.Conv2d(n_channel_in3, n_channel_out3, kernel_size, stride, padding) #Should I add bias=False?
		self.bn3 = nn.BatchNorm2d(n_channel_out3)
		self.relu3 = nn.ReLU()
		self.pool3 = nn.MaxPool2d(kernel_size = n_pool3)

		self.conv4 = nn.Conv2d(n_channel_in4, n_channel_out4, kernel_size, stride, padding)
		self.bn4 = nn.BatchNorm2d(n_channel_out4)        
		self.relu4 = nn.ReLU()
		self.pool4 = nn.MaxPool2d(kernel_size = n_pool4)
		
		self.fc1 = nn.Linear(in_features = map_size * map_size * n_channel_out4, out_features = n_fc1_out * map_size * map_size )
		self.dropout = nn.Dropout(0.5)
		self.fc2 = nn.Linear(in_features = map_size * map_size * n_fc1_out, out_features = n_fc2_out * map_size * map_size )
		
		self.act1 = nn.Tanh()
		self.use_softmax = use_softmax
		
		if use_softmax:
			self.act2 = nn.Softmax(dim=1)
		else:
			self.act2 = nn.Tanh()

	def forward(self, x):
		#x = x.unsqueeze(0)
		# x = Variable(x.float())
		x = x.float().detach()
		
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.pool1(x)

		#x.register_hook(print)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu2(x)
		x = self.pool2(x)

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu3(x)
		x = self.pool3(x)

		x = self.conv4(x)
		x = self.bn4(x)
		x = self.relu4(x)
		x = self.pool4(x)

		x = x.view(-1,self.map_size*self.map_size*self.n_channel_out4)
		x = self.fc1(x)
		x = self.dropout(x)
		x = self.act1(x)
		x = self.fc2(x)

		x = self.act2(x)
		if not self.use_softmax:
			x = x.add(1.0)
			x = x.mul(0.5)
		return x

		
class policy_A3C(torch.nn.Module):
	def __init__(self, img_size, n_channel_input=5,num_actions = 3, add_raw_map_scan = False, img_size2 = 224, n_channel_input2=2):
		super(policy_A3C, self).__init__()

		self.two_path = add_raw_map_scan

		if add_raw_map_scan == False:
			n_policy_conv1_filters = 16
			n_policy_conv2_filters = 16
			size_policy_conv1_filters = 3
			size_policy_conv2_filters = 3
			conv_out_height = (((img_size - size_policy_conv1_filters) + 1) - size_policy_conv2_filters) + 1
			conv_out_width = (((img_size - size_policy_conv1_filters) + 1) - size_policy_conv2_filters) + 1
			
			
			self.policy_conv1 = nn.Conv2d(n_channel_input, n_policy_conv1_filters, size_policy_conv1_filters, stride=1)  #put stride = 1?
			self.policy_conv2 = nn.Conv2d(n_policy_conv1_filters, n_policy_conv2_filters, size_policy_conv2_filters)
			#self.proj_layer = nn.Linear(n_policy_conv2_filters * conv_out_height * conv_out_width, 256)
			self.n_lstm_state = n_policy_conv2_filters*conv_out_height*conv_out_width
			self.lstm = nn.LSTMCell(self.n_lstm_state, 256)
		else:
			n_policy_conv1_filters = 16
			n_policy_conv2_filters = 16
			n_policy_conv1_filters2 = 32
			n_policy_conv2_filters2 = 32
			n_pool1 = 2
			n_pool2 = 2

			size_policy_conv1_filters = 3
			size_policy_conv2_filters = 3
			size_policy_conv1_filters2 = 3
			size_policy_conv2_filters2 = 3
			
			conv_out_height = (((img_size - size_policy_conv1_filters) + 1) - size_policy_conv2_filters) + 1
			conv_out_width = (((img_size - size_policy_conv1_filters) + 1) - size_policy_conv2_filters) + 1
			conv_out_height2 = ((((img_size2 - size_policy_conv1_filters2) + 1)/n_pool1 - size_policy_conv2_filters2) + 1)/n_pool2
			conv_out_width2 =  ((((img_size2 - size_policy_conv1_filters2) + 1)/n_pool1 - size_policy_conv2_filters2) + 1)/n_pool2

			self.pool1 = nn.MaxPool2d(kernel_size = n_pool1)
			self.pool2 = nn.MaxPool2d(kernel_size = n_pool2)
			
			self.policy_conv1 = nn.Conv2d(n_channel_input, n_policy_conv1_filters, size_policy_conv1_filters, stride=1)  #put stride = 1?
			self.policy_conv2 = nn.Conv2d(n_policy_conv1_filters, n_policy_conv2_filters, size_policy_conv2_filters)

			self.policy_conv1_2 = nn.Conv2d(n_channel_input2, n_policy_conv1_filters2, size_policy_conv1_filters2, stride=1)  #put stride = 1?
			self.policy_conv2_2 = nn.Conv2d(n_policy_conv1_filters2, n_policy_conv2_filters2, size_policy_conv2_filters2)
			self.n_lstm_state1 = n_policy_conv2_filters * conv_out_height * conv_out_width
			self.n_lstm_state2 = n_policy_conv2_filters2 * conv_out_height2 * conv_out_width2
			self.lstm = nn.LSTMCell(self.n_lstm_state1 + self.n_lstm_state2, 256)


		self.critic_linear = nn.Linear(256,1)
		self.actor_linear = nn.Linear(256, num_actions)

		self.apply(weights_init)
		self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
		self.actor_linear.bias.data.fill_(0)
		self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
		self.critic_linear.bias.data.fill_(0)

		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)

		# self.train()

	def forward(self, inputs):
		if self.two_path==False:
			inputs, (hx, cx) = inputs
			x = F.elu(self.policy_conv1(inputs))
			x = F.elu(self.policy_conv2(x))
			x = x.view(-1, self.n_lstm_state)
			hx, cx = self.lstm(x, (hx, cx))
			x = hx
		else:
			inputs1, inputs2, (hx, cx) = inputs
			x1 = F.elu(self.policy_conv1(inputs1))
			x1 = F.elu(self.policy_conv2(x1))
			x1 = x1.view(-1, self.n_lstm_state1)

			x2 = F.elu(self.policy_conv1_2(inputs2))
			x2 = self.pool1(x2)
			x2 = F.elu(self.policy_conv2_2(x2))
			x2 = self.pool2(x2)
			x2 = x2.view(-1, self.n_lstm_state2)

			x = torch.cat((x1,x2), dim=1) # (1, 13456) and (1, 93312) --> (1, 106768)
			hx, cx = self.lstm(x, (hx, cx)) 
			x = hx 

		return self.critic_linear(x), self.actor_linear(x), (hx, cx)

class intrinsic_model(torch.nn.Module):
	def __init__(self,map_size=11):#, args):
		super(intrinsic_model, self).__init__()

		#self.map_size = args.map_size
		self.map_size = map_size
		num_orientations = 4
		n_policy_conv1_filters = 16
		n_policy_conv2_filters = 16
		size_policy_conv1_filters = 3
		size_policy_conv2_filters = 3
		conv_out_height = (((self.map_size - size_policy_conv1_filters) + 1) - size_policy_conv2_filters) + 1
		conv_out_width = (((self.map_size - size_policy_conv1_filters) + 1) - size_policy_conv2_filters) + 1
		self.policy_conv1 = nn.Conv2d(3, n_policy_conv1_filters, size_policy_conv1_filters, stride = 1)
		self.policy_conv2 = nn.Conv2d(n_policy_conv1_filters, n_policy_conv2_filters, size_policy_conv2_filters, stride = 1)
		self.linear1 = nn.Linear(n_policy_conv2_filters * conv_out_height * conv_out_width, 256)
		self.linear2 = nn.Linear(257, 1)
		self.apply(weights_init)
		self.linear2.weight.data = normalized_columns_initializer(self.linear2.weight.data, 0.01)
		self.linear2.bias.data.fill_(0)
		self.train()

	def forward(self, state, action):
		x = state
		x = x.unsqueeze(0)
		x = F.elu(self.policy_conv1(x))
		x = F.elu(self.policy_conv2(x))
		x = x.view(x.size(0), -1)
		x = self.linear1(x)
		action_variable = Variable(action).float()
		x = torch.cat((action_variable, x), dim=1)
		x = self.linear2(x)
		x = torch.clamp(x, min=-2, max=2)
		return x


class Policy(nn.Module):
	def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
		super(Policy, self).__init__()
		if base_kwargs is None:
			base_kwargs = {}
		if base is None:
			if len(obs_shape) == 3:
				base = CNNBase
			elif len(obs_shape) == 1:
				base = MLPBase
			else:
				raise NotImplementedError

		self.base = base(obs_shape[0], **base_kwargs)


		print(action_space)
		print(action_space.__class__.__name__)
		if action_space.__class__.__name__ == "Discrete":
			num_outputs = action_space.n
			self.dist = Categorical(self.base.output_size, num_outputs)
		elif action_space.__class__.__name__ == "Box":
			num_outputs = action_space.shape[0]
			self.dist = DiagGaussian(self.base.output_size, num_outputs)
		elif action_space.__class__.__name__ == "MultiBinary":
			num_outputs = action_space.shape[0]
			self.dist = Bernoulli(self.base.output_size, num_outputs)
		else:
			raise NotImplementedError

	@property
	def is_recurrent(self):
		return self.base.is_recurrent

	@property
	def recurrent_hidden_state_size(self):
		"""Size of rnn_hx."""
		return self.base.recurrent_hidden_state_size

	def forward(self, inputs, rnn_hxs, masks):
		raise NotImplementedError

	def act(self, inputs, rnn_hxs, masks, deterministic=False):
		value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
		dist = self.dist(actor_features)

		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()

		action_log_probs = dist.log_probs(action)
		dist_entropy = dist.entropy().mean()

		return value, action, action_log_probs, rnn_hxs

	def get_value(self, inputs, rnn_hxs, masks):
		value, _, _ = self.base(inputs, rnn_hxs, masks)
		return value

	def evaluate_actions(self, inputs, rnn_hxs, masks, action):
		value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
		dist = self.dist(actor_features)

		action_log_probs = dist.log_probs(action)
		dist_entropy = dist.entropy().mean()

		return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):

	def __init__(self, recurrent, recurrent_input_size, hidden_size):
		super(NNBase, self).__init__()

		self._hidden_size = hidden_size
		self._recurrent = recurrent

		if recurrent:
			self.gru = nn.GRU(recurrent_input_size, hidden_size)
			for name, param in self.gru.named_parameters():
				if 'bias' in name:
					nn.init.constant_(param, 0)
				elif 'weight' in name:
					nn.init.orthogonal_(param)

	@property
	def is_recurrent(self):
		return self._recurrent

	@property
	def recurrent_hidden_state_size(self):
		if self._recurrent:
			return self._hidden_size
		return 1

	@property
	def output_size(self):
		return self._hidden_size

	def _forward_gru(self, x, hxs, masks):
		if x.size(0) == hxs.size(0):
			x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
			x = x.squeeze(0)
			hxs = hxs.squeeze(0)
		else:
			# x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
			N = hxs.size(0)
			T = int(x.size(0) / N)

			# unflatten
			x = x.view(T, N, x.size(1))

			# Same deal with masks
			masks = masks.view(T, N)

			# Let's figure out which steps in the sequence have a zero for any agent
			# We will always assume t=0 has a zero in it as that makes the logic cleaner
			has_zeros = ((masks[1:] == 0.0) \
							.any(dim=-1)
							.nonzero()
							.squeeze()
							.cpu())


			# +1 to correct the masks[1:]
			if has_zeros.dim() == 0:
				# Deal with scalar
				has_zeros = [has_zeros.item() + 1]
			else:
				has_zeros = (has_zeros + 1).numpy().tolist()

			# add t=0 and t=T to the list
			has_zeros = [0] + has_zeros + [T]


			hxs = hxs.unsqueeze(0)
			outputs = []
			for i in range(len(has_zeros) - 1):
				# We can now process steps that don't have any zeros in masks together!
				# This is much faster
				start_idx = has_zeros[i]
				end_idx = has_zeros[i + 1]

				rnn_scores, hxs = self.gru(
					x[start_idx:end_idx],
					hxs * masks[start_idx].view(1, -1, 1)
				)

				outputs.append(rnn_scores)

			# assert len(outputs) == T
			# x is a (T, N, -1) tensor
			x = torch.cat(outputs, dim=0)
			# flatten
			x = x.view(T * N, -1)
			hxs = hxs.squeeze(0)

		return x, hxs


class CNNBase(NNBase):
	def __init__(self, num_inputs, recurrent=False, hidden_size=512):
		super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

		init_ = lambda m: init(m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0),
			nn.init.calculate_gain('relu'))

		num_inputs = 6
		# hidden_size = 100
		# self.main = nn.Sequential(
		#     init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
		#     nn.ReLU(),
		#     init_(nn.Conv2d(32, 64, 4, stride=2)),
		#     nn.ReLU(),
		#     init_(nn.Conv2d(64, 32, 3, stride=1)),
		#     nn.ReLU(),
		#     Flatten(),
		#     init_(nn.Linear(32 * 7 * 7, hidden_size)),
		#     nn.ReLU()
		# )

		self.main = nn.Sequential(
			init_(nn.Conv2d(num_inputs, 16, 3,padding=1, stride=1)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2),
			init_(nn.Conv2d(16, 32, 3,padding=1, stride=1)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2),
			init_(nn.Conv2d(32, 32, 3,padding=1, stride=1)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2),
			init_(nn.Conv2d(32, 16, 3,padding=1, stride=1)),
			nn.ReLU(),
			# nn.MaxPool2d(kernel_size = 2),
			Flatten(),
			init_(nn.Linear(16 * 11 * 11, hidden_size)),
			nn.ReLU()
		)

		init_ = lambda m: init(m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0))

		self.critic_linear = init_(nn.Linear(hidden_size, 1))

		self.train()

	def forward(self, inputs, rnn_hxs, masks):
		# inputs = inputs.permute(0,3,1,2)
		# x = self.main(inputs / 255.0)
		print("inputs shape is ",inputs.shape)
		x = self.main(inputs)

		if self.is_recurrent:
			x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

		return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
	def __init__(self, num_inputs, recurrent=False, hidden_size=64):
		super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

		if recurrent:
			num_inputs = hidden_size

		init_ = lambda m: init(m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0),
			np.sqrt(2))

		self.actor = nn.Sequential(
			init_(nn.Linear(num_inputs, hidden_size)),
			nn.Tanh(),
			init_(nn.Linear(hidden_size, hidden_size)),
			nn.Tanh()
		)

		self.critic = nn.Sequential(
			init_(nn.Linear(num_inputs, hidden_size)),
			nn.Tanh(),
			init_(nn.Linear(hidden_size, hidden_size)),
			nn.Tanh()
		)

		self.critic_linear = init_(nn.Linear(hidden_size, 1))

		self.train()

	def forward(self, inputs, rnn_hxs, masks):
		x = inputs

		if self.is_recurrent:
			x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

		hidden_critic = self.critic(x)
		hidden_actor = self.actor(x)

		return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs



class perceptual_conv_real_224_l0(nn.Module):
	def __init__(self, layers):
		super(perceptual_conv_real_224_l0, self).__init__()
		
		self.layers = layers

		n_channel_in1 = 5
		n_channel_out1 = 8
		n_pool1 = 2

		n_channel_in2 = 8
		n_channel_out2 = 16
		n_pool2 = 2

		n_pool3 = 2

		if self.layers == 3:
			n_channel_in3 = 16
			n_channel_out3 = 4

		elif self.layers == 4:
			n_channel_in3 = 16
			n_channel_out3 = 8
			n_channel_in4 = 8
			n_channel_out4 = 4

		elif self.layers == 5:
			n_channel_in2 = 8
			n_channel_out2 = 32
			n_channel_in3 = 32
			n_channel_out3 = 16
			n_channel_in4 = 16
			n_channel_out4 = 8
			n_channel_in5 = 8
			n_channel_out5 = 4 
		
		else:
			print("invalid layers")

		kernel_size = 3
		stride = 1
		padding1 = 1
		padding0 = 0

		self.conv1 = nn.Conv2d(n_channel_in1, n_channel_out1, kernel_size, stride, padding0) #Should I add bias=False?
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size = n_pool1)

		self.conv2 = nn.Conv2d(n_channel_in2, n_channel_out2, kernel_size, stride, padding0)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(kernel_size = n_pool2)

		self.conv3 = nn.Conv2d(n_channel_in3, n_channel_out3, kernel_size, stride, padding0)
		self.relu3 = nn.ReLU()
		self.pool3 = nn.MaxPool2d(kernel_size = n_pool3)

		if self.layers >= 4:
			self.conv4 = nn.Conv2d(n_channel_in4, n_channel_out4, kernel_size, stride, padding1)
			self.relu4 = nn.ReLU()
			self.pool4 = nn.MaxPool2d(kernel_size=2)

		if self.layers == 5:
			self.conv5 = nn.Conv2d(n_channel_in5, n_channel_out5, kernel_size, stride, padding0)
			self.relu5 = nn.ReLU()

	def forward(self, x):

		x = Variable(x.float())
		x = self.conv1(x)
		# print(x.shape)
		x = self.relu1(x)
		x = self.pool1(x)
		# print(x.shape)

		x = self.conv2(x)
		# print(x.shape)
		x = self.relu2(x)
		x = self.pool2(x)
		# print(x.shape)

		if self.layers == 3:
			x = self.conv3(x)
			x = self.relu3(x)
			x = self.pool3(x)
		
		elif self.layers == 4:
			x = self.conv3(x)
			x = self.relu3(x)
			x = self.pool3(x)
			x = self.conv4(x)
			x = self.relu4(x)
		
		elif self.layers == 5:
			x = self.conv3(x)
			# print(x.shape)
			x = self.relu3(x)
			x = self.pool3(x)
			# print(x.shape)
			x = self.conv4(x)
			# print(x.shape)
			x = self.relu4(x)
			x = self.pool4(x)
			# print(x.shape)
			x = self.conv5(x)
			# print(x.shape)
			x = self.relu5(x)
		
		else:
			print("invalid layers")


		return x


class perceptual_conv_real_224_l1(nn.Module):
	def __init__(self, layers):
		super(perceptual_conv_real_224_l1, self).__init__()

		self.layers = layers

		n_channel_in1 = 5
		n_channel_out1 = 8
		n_pool1 = 2

		n_channel_in2 = 8
		n_channel_out2 = 16
		n_pool2 = 2
		n_pool3 = 2

		if self.layers == 3:
			n_channel_in3 = 16
			n_channel_out3 = 1
			

		if self.layers == 4:
			n_channel_in3 = 16
			n_channel_out3 = 8

			n_channel_in4 = 8
			n_channel_out4 = 4
			
		if self.layers == 5:
			n_channel_in3 = 16
			n_channel_out3 = 32

			n_channel_in4 = 32
			n_channel_out4 = 8

			n_channel_in5 = 8
			n_channel_out5 = 4

		
		kernel_size = 3
		stride = 1
		padding = 1

		self.conv1 = nn.Conv2d(n_channel_in1, n_channel_out1, kernel_size, stride, padding)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size = n_pool1)

		self.conv2 = nn.Conv2d(n_channel_in2, n_channel_out2, kernel_size, stride, padding)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(kernel_size = n_pool2)

		self.conv3 = nn.Conv2d(n_channel_in3, n_channel_out3, kernel_size, stride, padding)
		self.relu3 = nn.ReLU()
		self.pool3 = nn.MaxPool2d(kernel_size = n_pool3)

		if self.layers == 4:
			self.conv4 = nn.Conv2d(n_channel_in4, n_channel_out4, kernel_size, stride, padding)
			self.relu4 = nn.ReLU()

		if self.layers == 5:
			self.conv3 = nn.Conv2d(n_channel_in3, n_channel_out3, kernel_size, stride, padding)
			self.relu3 = nn.ReLU()
			self.conv4 = nn.Conv2d(n_channel_in4, n_channel_out4, kernel_size, stride, padding)
			self.relu4 = nn.ReLU()
			self.conv5 = nn.Conv2d(n_channel_in5, n_channel_out5, kernel_size, stride, padding)
			self.relu5 = nn.ReLU()

	def forward(self, x):

		x = Variable(x.float())
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.pool1(x)

		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pool2(x)

		x = self.conv3(x)
		x = self.relu3(x)
		x = self.pool3(x)

		if self.layers >= 4:
			x = self.conv4(x)
			x = self.relu4(x)

		if self.layers == 5:
			x = self.conv5(x)
			x = self.relu5(x)
 
		return x

class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, h1, h2):
        super().__init__()
        input_dim = state_dim + action_dim

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.fc = nn.Sequential(
            init_(nn.Linear(input_dim, h1)),
            nn.LeakyReLU(),
            init_(nn.Linear(h1, h2)),
            nn.LeakyReLU(),
            init_(nn.Linear(h2, 1)),
            nn.LeakyReLU()
        )

    def forward(self, obs, action):
        # action = action.type(torch.cuda.FloatTensor)
        # action = action.float()
        obs = obs.view(1, -1)
        print("shape of obs, action = ", obs.shape, action.shape)
        x = self.fc(torch.cat([obs, action], dim=1))
        return x

#!/usr/bin/env python
# for stage-4 world
from __future__ import print_function
from enum import IntEnum

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from scipy import ndimage, interpolate
import pdb
import glob
import os
import errno
import re
import time
import random
import cv2
from recordtype import recordtype

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import transforms
# from logger import Logger

from copy import deepcopy

import argparse

import copy
import math
import argparse
from datetime import datetime
from gym_dal.maze import generate_map
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from a2c_ppo_acktr import arguments

from resnet_pm import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.resnet import resnet18 as resnet18s
from torchvision.models.resnet import resnet34 as resnet34s
from torchvision.models.resnet import resnet50 as resnet50s
from torchvision.models.resnet import resnet101 as resnet101s
from torchvision.models.resnet import resnet152 as resnet152s
from torchvision.models.densenet import densenet121, densenet169, densenet201, densenet161

from networks import intrinsic_model
from networks import policy_A3C

def shift(grid, d, axis=None, fill = 0.5):
	grid = np.roll(grid, d, axis=axis)
	if axis == 0:
		if d > 0:
			grid[:d,:] = fill
		elif d < 0:
			grid[d:,:] = fill
	elif axis == 1:
		if d > 0:
			grid[:,:d] = fill
		elif d < 0:
			grid[:,d:] = fill
	return grid

def softmax(w, t = 1.0):
	e = np.exp(np.array(w) / t)
	dist = e / np.sum(e)
	return dist

def softermax(w, t = 1.0):
	w = np.array(w)
	w = w - w.min() + np.exp(1)
	e = np.log(w)
	dist = e / np.sum(e)
	return dist


def to_index(a, N, mm):
	a_max = mm[1]
	a_min = mm[0]
	return int(np.floor(N*(a_max-a)/(a_max-a_min)))

def to_real(i, mm, n):
	u = (mm[1]-mm[0])/n
	u0 = u/2
	return mm[1]-u*i - u0

def wrap(phase):
	# wrap into [-pi, pi]
	phase = ( phase + np.pi) % (2 * np.pi ) - np.pi
	return phase


def normalize(x):
	if x.min() == x.max():
		return 0.0*x
	x = x-x.min()
	x = x/x.max()
	return x



Pose2d = recordtype("Pose2d", "theta x y")
Grid = recordtype("Grid", "head row col")

class Lidar():
	def __init__(self, ranges, angle_min, angle_max,
				 range_min, range_max):
		# self.ranges = np.clip(ranges, range_min, range_max)
		self.ranges = ranges
		self.angle_min = angle_min
		self.angle_max = angle_max
		num_data = len(self.ranges)
		self.angle_increment = (self.angle_max-self.angle_min)/num_data #math.increment


class DalEnv(gym.Env):

	class Actions(IntEnum):
		left = 0
		right = 1
		forward = 2
		hold = 3

	def __init__(self):

		seed = 1337
		self.args = arguments.get_args_iko()
		self.rl_test = False
		self.start_time = time.time()

		self.actions = DalEnv.Actions
		self.action_space = spaces.Discrete(len(self.actions))

		if (self.args.use_gpu) > 0 and torch.cuda.is_available():
			self.device = torch.device("cuda" )
			torch.set_default_tensor_type(torch.cuda.FloatTensor)
		else:
			self.device = torch.device("cpu")
			torch.set_default_tensor_type(torch.FloatTensor)

		self.init_fig = False
		self.n_maze_grids = None

		self.grid_rows = self.args.map_size * self.args.sub_resolution
		self.grid_cols = self.args.map_size * self.args.sub_resolution
		self.grid_dirs = self.args.n_headings
		self.collision_radius = self.args.collision_radius

		num_dirs = 1

		num_classes = self.args.n_lm_grids ** 2 * num_dirs
		final_num_classes = num_classes
		  
		if self.args.n_pre_classes is not None:
			num_classes = self.args.n_pre_classes
		else:
			num_classes = final_num_classes

		self.map_rows, self.map_cols = 224, 224
		if self.args.pm_net == "none":
			self.perceptual_model = None
		elif self.args.pm_net == "densenet121":
			self.perceptual_model = densenet121(pretrained = self.args.use_pretrained, drop_rate = self.args.drop_rate)
			num_ftrs = self.perceptual_model.classifier.in_features # 1024
			self.perceptual_model.classifier = nn.Linear(num_ftrs, num_classes)
		elif self.args.pm_net == "densenet169":
			self.perceptual_model = densenet169(pretrained = self.args.use_pretrained, drop_rate = self.args.drop_rate)
			num_ftrs = self.perceptual_model.classifier.in_features # 1664
			self.perceptual_model.classifier = nn.Linear(num_ftrs, num_classes)
		elif self.args.pm_net == "densenet201":
			self.perceptual_model = densenet201(pretrained = self.args.use_pretrained, drop_rate = self.args.drop_rate)
			num_ftrs = self.perceptual_model.classifier.in_features # 1920
			self.perceptual_model.classifier = nn.Linear(num_ftrs, num_classes)
		elif self.args.pm_net == "densenet161":
			self.perceptual_model = densenet161(pretrained = self.args.use_pretrained, drop_rate = self.args.drop_rate)
			num_ftrs = self.perceptual_model.classifier.in_features # 2208
			self.perceptual_model.classifier = nn.Linear(num_ftrs, num_classes)
		elif self.args.pm_net == "resnet18s":
			self.perceptual_model = resnet18s(pretrained=self.args.use_pretrained)
			num_ftrs = self.perceptual_model.fc.in_features
			self.perceptual_model.fc = nn.Linear(num_ftrs, num_classes)
		elif self.args.pm_net == "resnet34s":
			self.perceptual_model = resnet34s(pretrained=self.args.use_pretrained)
			num_ftrs = self.perceptual_model.fc.in_features
			self.perceptual_model.fc = nn.Linear(num_ftrs, num_classes)
		elif self.args.pm_net == "resnet50s":
			self.perceptual_model = resnet50s(pretrained=self.args.use_pretrained)
			num_ftrs = self.perceptual_model.fc.in_features
			self.perceptual_model.fc = nn.Linear(num_ftrs, num_classes)
		elif self.args.pm_net == "resnet101s":
			self.perceptual_model = resnet101s(pretrained=self.args.use_pretrained)
			num_ftrs = self.perceptual_model.fc.in_features
			self.perceptual_model.fc = nn.Linear(num_ftrs, num_classes)
		elif self.args.pm_net == "resnet152s":
			self.perceptual_model = resnet152s(pretrained=self.args.use_pretrained)
			num_ftrs = self.perceptual_model.fc.in_features
			self.perceptual_model.fc = nn.Linear(num_ftrs, num_classes)
		elif self.args.pm_net == "resnet18":
			self.perceptual_model = resnet18(num_classes = num_classes)
			num_ftrs = self.perceptual_model.fc.in_features
		elif self.args.pm_net == "resnet34":
			self.perceptual_model = resnet34(num_classes = num_classes)
			num_ftrs = self.perceptual_model.fc.in_features
		elif self.args.pm_net == "resnet50":
			self.perceptual_model = resnet50(num_classes = num_classes)
			num_ftrs = self.perceptual_model.fc.in_features
		elif self.args.pm_net == "resnet101":
			self.perceptual_model = resnet101(num_classes = num_classes)
			num_ftrs = self.perceptual_model.fc.in_features
		elif self.args.pm_net == "resnet152":
			self.perceptual_model = resnet152(num_classes = num_classes)
			num_ftrs = self.perceptual_model.fc.in_features # 2048
		else:
			raise Exception('pm-net required: resnet or densenet')


		# if self.args.RL_type == 0:
		# 	self.policy_model = policy_A3C(self.args.n_state_grids, 2+self.args.n_state_dirs, num_actions = self.args.num_actions)
		# elif self.args.RL_type == 1:
		# 	self.policy_model = policy_A3C(self.args.n_state_grids, 1+self.args.n_state_dirs, num_actions = self.args.num_actions)
		# elif self.args.RL_type == 2:
		# 	self.policy_model = policy_A3C(self.args.n_state_grids, 2*self.args.n_state_dirs, num_actions = self.args.num_actions, add_raw_map_scan = True)

		self.intri_model = intrinsic_model(self.grid_rows)

		
		
		self.max_scan_range = 3.5
		self.min_scan_range = 0.1

		self.manhattans = []
		self.manhattan = 0
		self.rewards = []
		self.reward = 0

		self.done = 0

		self.step_count = 0
		self.step_max = self.args.num[2]
	   
		self.map_2d = None
		# self.laser_1d = None
		self.xlim = (-3.0, 3.0)
		self.ylim = (-3.0, 3.0)

		if self.args.thickness == 0.0:
			self.radius = 0.5*(self.xlim[1]-self.xlim[0])/self.args.map_size/2*0.9
		else:
			self.radius = (self.xlim[1]-self.xlim[0])/self.args.map_size/2*self.args.thickness
		
		self.longest = float(self.grid_dirs/2 + self.grid_rows-1 + self.grid_cols-1)  #longest possible manhattan distance

		self.cell_size = (self.xlim[1]-self.xlim[0])/self.grid_rows
		self.heading_resol = 2*np.pi/self.grid_dirs
		self.fwd_step = self.cell_size*self.args.fwd_step
		self.collision = False
		self.sigma_xy = self.cell_size * 0.1
		self.sigma_theta = self.heading_resol * 0.1
			

		self.scans_over_map = np.zeros((self.grid_rows,self.grid_cols,360))
		self.scans_over_map_high = np.zeros((self.map_rows, self.map_cols, 360))

		self.scan_2d = np.zeros((self.map_rows, self.map_cols))
		self.scan_2d_low = np.zeros((self.grid_rows, self.grid_cols))

		self.map_2d = np.zeros((self.map_rows, self.map_cols))
		self.map_design = np.zeros((self.grid_rows, self.grid_cols),dtype='float')
		self.map_design_tensor = torch.zeros((1,self.grid_rows, self.grid_cols),device=torch.device(self.device))

		self.data_cnt = 0

		self.bel_ent = np.log(1.0/(self.grid_dirs*self.grid_rows*self.grid_cols))

		self.likelihood = torch.ones((self.grid_dirs,self.grid_rows, self.grid_cols),device=torch.device(self.device))
		self.likelihood = self.likelihood / self.likelihood.sum()   
		
		# self.gt_likelihood_high = np.ones((self.grid_dirs, self.map_rows, self.map_cols))
		self.gt_likelihood_high = np.ones((self.grid_dirs, self.grid_rows, self.grid_cols))
		self.gt_likelihood_high = self.gt_likelihood_high / self.gt_likelihood_high.sum()
		self.gt_likelihood_unnormalized = np.ones((self.grid_dirs,self.grid_rows,self.grid_cols)) 
		self.gt_likelihood_unnormalized_high = np.ones((self.grid_dirs, self.grid_rows, self.grid_cols))

		# self.belief = torch.ones((self.grid_dirs,self.map_rows, self.map_cols),device=torch.device(self.device))
		self.belief = torch.ones((self.grid_dirs, self.grid_rows, self.grid_cols), device=torch.device(self.device))
		self.belief = self.belief / self.belief.sum()

		self.loss_policy = 0
		self.loss_value = 0
		
		# self.turtle_loc = np.zeros((self.map_rows,self.map_cols))
		self.turtle_loc = np.zeros((self.grid_rows, self.grid_cols))


		# what to do
		# current pose: where the robot really is. motion incurs errors in pose
		self.current_pose = Pose2d(0,0,0)
		self.goal_pose = Pose2d(0,0,0)
		self.last_pose = Pose2d(0,0,0)
		self.perturbed_goal_pose = Pose2d(0,0,0)        
		self.start_pose = Pose2d(0,0,0)
		#grid pose
		self.true_grid = Grid(head=0,row=0,col=0)
		self.bel_grid = Grid(head=0,row=0,col=0)


		self.reward_block_penalty = 0
		self.reward_bel_gt = 0
		self.reward_bel_gt_nonlog = 0
		self.reward_infogain = 0
		self.reward_bel_ent = 0
		self.reward_hit = 0
		self.reward_dist = 0
		self.reward_inv_dist = 0

		self.actions_space = list(("turn_left", "turn_right", "go_fwd", "hold"))
		self.action_name = 'none'
		self.current_state = "new_env_pose"

		self.state = np.zeros((6, self.grid_rows, self.grid_cols), dtype=np.float32)

		self.observation_space = spaces.Box(
			low=0,
			high=1,
			shape=self.state.shape,
			dtype='float32'
		)
		# Initialize the RNG
		self.seed(seed=seed)

		# Initialize the state
		self.reset()

		#end of init

	def seed(self, seed=None):
		self.np_random, _ = seeding.np_random(seed)
		return [seed]

	def reset(self):
		self.clear_objects()
		self.set_walls()
		self.place_turtle()
		self.get_lidar()
		self.get_scan_2d()
		self.step_count = 0
		self.gt_likelihood_high = np.ones((self.grid_dirs, self.grid_rows, self.grid_cols))
		self.gt_likelihood_high = self.gt_likelihood_high / self.gt_likelihood_high.sum()
		self.gt_likelihood_unnormalized = np.ones((self.grid_dirs,self.grid_rows,self.grid_cols)) 
		self.gt_likelihood_unnormalized_high = np.ones((self.grid_dirs, self.grid_rows, self.grid_cols))

		self.rewards = []
		self.manhattans=[]
		self.reward = 0
		self.explored_space = np.zeros((self.grid_dirs,self.grid_rows, self.grid_cols),dtype='float')
		self.bel_list = []

		self.state[0,:,:] = self.map_design
		ding = self.belief.detach().cpu().numpy()
		self.state[1:5,:,:] = ding
		self.state[5,:,:] = self.scan_2d_low

		return self.state

	@property
	def steps_remaining(self):
		return self.max_steps - self.step_count
	

	def reset_pose(self):
		self.place_turtle()
		self.step_count = 0
		self.bel_ent = np.log(1.0/(self.grid_dirs*self.grid_rows*self.grid_cols))
		self.step_count = 0

		# reset belief too
		self.belief[:,:,:]=1.0
		self.belief /= self.belief.sum()#np.sum(self.belief, dtype=float)

		done = False

	def step(self, action):

		# start = time.time()
		done = False
		reward = 0
		if self.step_count == 0:
			self.get_synth_scan()
		self.step_count = self.step_count + 1

		self.action = action
		self.action_name = self.actions_space[action]
		self.update_target_pose()
		self.collision_check()
		self.execute_action_teleport()
		self.transit_belief()
		if self.collision == False:
			self.update_true_grid()

		self.get_lidar()
		self.update_explored()
		self.scan_2d, self.scan_2d_low = self.get_scan_2d_n_headings()

		# self.get_scan_2d()
		# self.generate_map_trans()
		self.compute_gtl()
		# self.likelihood = self.update_likelihood_rotate(self.map_2d, self.scan_2d)
		self.likelihood = self.gt_likelihood_high
		self.likelihood = self.likelihood/(torch.sum(self.likelihood))
		if self.args.mask:
			self.mask_likelihood()
		self.product_belief() # likelihood x belief
		self.update_bel_list()
		self.get_reward()

		self.state[0,:,:] = self.map_design
		ding = self.belief.detach().cpu().numpy()
		self.state[1:5,:,:] = ding
		self.state[5,:,:] = self.scan_2d_low


		if self.step_count >= self.step_max:
			done = True

		obs = self.state
		reward = self.reward

		return obs, reward, done, {'reward_block_penalty': self.reward_block_penalty,
									'reward_bel_gt': self.reward_bel_gt,
									'reward_bel_gt_nonlog': self.reward_bel_gt_nonlog,
									'reward_infogain': self.reward_infogain,
									'reward_bel_ent': self.reward_bel_ent,
									'reward_hit': self.reward_hit,
									'reward_dist': self.reward_dist,
									'reward_inv_dist': self.reward_inv_dist }

	def update_explored(self):
		if self.explored_space[self.true_grid.head,self.true_grid.row, self.true_grid.col] == 0.0:
			self.new_pose = True
		else:
			self.new_pose = False
		self.explored_space[self.true_grid.head,self.true_grid.row, self.true_grid.col] = 1.0
		return

	def compute_gtl(self):
		if self.args.gtl_src == 'hd-corr':
			self.get_gt_likelihood_corr(clip=0)
		elif self.args.gtl_src == 'hd-corr-clip':
			self.get_gt_likelihood_corr(clip=1)
		elif self.args.gtl_src == 'hd-cos':
			self.get_gt_likelihood_cossim()
		else:
			raise Exception('GTL source required: --gtl-src= [low-dim-map, high-dim-map]')
		self.normalize_gtl()

	def normalize_gtl(self):
		if type(self.gt_likelihood_high).__name__ == 'torch.CudaTensor':
			gt_high = self.gt_likelihood_high.cpu().numpy()
		elif type(self.gt_likelihood_high).__name__ == 'Tensor':
			gt_high = self.gt_likelihood_high.cpu().numpy()
		else:
			gt_high = self.gt_likelihood_high
		#self.gt_likelihood_unnormalized = np.copy(self.gt_likelihood)
		if self.args.gtl_output == "softmax":
			gt_high = softmax(gt_high, self.args.temperature)
		elif self.args.gtl_output == "softermax":
			gt_high = softermax(gt_high.cpu())
		elif self.args.gtl_output == "linear":
			gt_high = np.clip(gt_high.cpu(), 1e-5, 1.0)
			gt_high = gt_high / gt_high.sum()
		self.gt_likelihood_high = torch.tensor(gt_high).float().to(self.device)

	def get_gt_likelihood_cossim(self):
		offset = 360/self.grid_dirs
		y= np.array(self.scan_data_at_unperturbed.ranges)[::self.args.pm_scan_step]
		# y= np.array(self.scan_data.ranges)[::self.args.pm_scan_step]
		y = np.clip(y, self.min_scan_range, self.max_scan_range)
		# y[y==np.inf]= self.max_scan_range

		for heading in range(self.grid_dirs):  ## that is, each direction
			#compute cosine similarity at each loc
			X = np.roll(self.scans_over_map_high, int(-offset*heading),axis=2)[:,:,::self.args.pm_scan_step]
			for i_ld in range(self.grid_rows):
				for j_ld in range(self.grid_cols):
					if (i_ld*self.map_cols+j_ld == self.taken).any():
						self.gt_likelihood_high[heading,i_ld,j_ld]= 0.0
					else:
						x = X[i_ld,j_ld,:]
						x = np.clip(x, self.min_scan_range, self.max_scan_range)

						self.gt_likelihood_high[heading,i_ld,j_ld]= self.get_cosine_sim(x,y)

	def get_cosine_sim(self,x,y):
		# numpy arrays.
		return sum(x*y)/np.linalg.norm(y,2)/np.linalg.norm(x,2)

	def get_gt_likelihood_corr(self):
		offset = 360/self.grid_dirs
		y= np.array(self.scan_data_at_unperturbed.ranges)[::self.args.pm_scan_step]
		# y= np.array(self.scan_data.ranges)[::self.args.pm_scan_step]
		y = np.clip(y, self.min_scan_range, self.max_scan_range)
		# y[y==np.inf]= self.max_scan_range
		for heading in range(self.grid_dirs):  ## that is, each direction
			#compute cosine similarity at each loc
			X = np.roll(self.scans_over_map, -offset*heading,axis=2)[:,:,::self.args.pm_scan_step]
			for i_ld in range(self.grid_rows):
				for j_ld in range(self.grid_cols):
					if (i_ld*self.grid_cols+j_ld == self.taken).any():
						self.gt_likelihood[heading,i_ld,j_ld]= 0.0
					else:
						x = X[i_ld,j_ld,:]
						x = np.clip(x, self.min_scan_range, self.max_scan_range)
						self.gt_likelihood[heading,i_ld,j_ld]= self.get_corr(x,y)

	def get_synth_scan(self):     
		# place sensor at a location, then reach out in 360 rays all around it and record when each ray gets hit.
		n_places=self.grid_rows * self.grid_cols

		for i_place in range(n_places):
			row_ld = i_place // self.grid_cols
			col_ld = i_place %  self.grid_cols
			x_real = to_real(row_ld, self.xlim, self.grid_rows ) # from low-dim location to real
			y_real = to_real(col_ld, self.ylim, self.grid_cols ) # from low-dim location to real
			scan = self.get_a_scan(x_real, y_real,scan_step=self.args.pm_scan_step)
			self.scans_over_map[row_ld, col_ld,:] = np.clip(scan, 1e-10, self.max_scan_range)
			# if i_place%10==0: print ('.')

		## Uncomment the following if you want scans_over_map at high resolution.
		# n_places = self.map_rows * self.map_cols
		# for i_place in range(n_places):
		#     row_ld = i_place // self.map_rows
		#     col_ld = i_place % self.map_cols
		#     x_real = to_real(row_ld, self.xlim, self.map_rows ) # from low-dim location to real
		#     y_real = to_real(col_ld, self.ylim, self.map_cols ) # from low-dim location to real
		#     scan = self.get_a_scan(x_real, y_real,scan_step=self.args.pm_scan_step)
		#     self.scans_over_map_high[row_ld, col_ld,:] = np.clip(scan, 1e-10, self.max_scan_range)
		#     if i_place%100==0: print ('.')




	def update_true_grid(self):
		self.true_grid.row=to_index(self.current_pose.x, self.grid_rows, self.xlim)
		self.true_grid.col=to_index(self.current_pose.y, self.grid_cols, self.ylim)
		heading = self.current_pose.theta
		
		self.true_grid.head = self.grid_dirs * wrap(heading + np.pi/self.grid_dirs) / 2.0 / np.pi
		self.true_grid.head = int(self.true_grid.head % self.grid_dirs)


	def teleport_turtle(self):
		# if self.args.perturb > 0:
		self.current_pose.x = self.perturbed_goal_pose.x
		self.current_pose.y = self.perturbed_goal_pose.y
		self.current_pose.theta = self.perturbed_goal_pose.theta

			
	def set_walls(self):
		if self.args.test_mode:
			map_file = os.path.join(self.args.test_data_path, "map-design-%05d.npy"%self.env_count)
			maze = np.load(map_file)

		else:            
			if self.args.random_rm_cells[1]>0:
				low=self.args.random_rm_cells[0]
				high=self.args.random_rm_cells[1]
				num_cells_to_delete = np.random.randint(low, high)
			else:
				num_cells_to_delete = self.args.rm_cells
			maze = generate_map(self.args.map_size, num_cells_to_delete)

		self.map_design = np.zeros((self.grid_rows, self.grid_cols))

		for i in range(self.args.map_size):
			for j in range(self.args.map_size):
				if i < self.args.map_size-1:
					if maze[i,j]==1 and maze[i+1,j]==1:
						#place vertical
						self.set_a_wall([i,j],[i+1,j],self.args.map_size,horizontal=False)
				if j < self.args.map_size-1:
					if maze[i,j]==1 and maze[i,j+1] ==1:
						#place horizontal wall
						self.set_a_wall([i,j],[i,j+1],self.args.map_size,horizontal=True)
				if i>0 and i<self.args.map_size-1 and j>0 and j<self.args.map_size-1:
					if maze[i,j]==1 and maze[i-1,j] == 0 and maze[i+1,j]==0 and maze[i,j-1]==0 and maze[i,j+1]==0:
						self.set_a_pillar([i,j], self.args.map_size)

		# self.map_design = maze
		self.map_design_tensor[0,:,:] = torch.tensor(self.map_design).float().to(self.device)
		self.taken = np.arange(self.map_design.size)[self.map_design.flatten()==1]
		

	def clear_objects(self):
		self.map_2d = np.zeros((self.map_rows, self.map_cols))
		self.map_design = np.zeros((self.grid_rows, self.grid_cols),dtype='float')
		self.map_design_tensor = torch.zeros((1,self.grid_rows, self.grid_cols),device=torch.device(self.device))

								
	def set_a_pillar(self, a, grids):
		x=to_real(a[0], self.xlim, grids)
		y=to_real(a[1], self.ylim, grids)

		#rad = self.radius
		if self.args.backward_compatible_maps:
			rad = 0.15
		elif self.args.random_thickness:
			rad = np.random.normal(loc=self.radius, scale=self.radius*0.1)
			rad = np.clip(rad, 0, self.radius*.95)
		else:
			rad = self.radius

		corner0 = [x+rad,y+rad]
		corner1 = [x-rad,y-rad]
		x0 = to_index(corner0[0], self.map_rows, self.xlim)
		y0 = to_index(corner0[1], self.map_cols, self.ylim)
		x1 = to_index(corner1[0], self.map_rows, self.xlim)
		y1 = to_index(corner1[1], self.map_cols, self.ylim)
		for ir in range(x0,x1+1):
			for ic in range(y0,y1+1):
				dx = to_real(ir, self.xlim, self.map_rows) - x
				dy = to_real(ic, self.ylim, self.map_cols) - y
				dist = np.sqrt(dx**2+dy**2)
				if dist <= rad:
					self.map_2d[ir,ic]=1.0

		x0 = to_index(corner0[0], self.grid_rows, self.xlim)
		y0 = to_index(corner0[1], self.grid_cols, self.ylim)
		x1 = to_index(corner1[0], self.grid_rows, self.xlim)
		y1 = to_index(corner1[1], self.grid_cols, self.ylim)

		corners = [(0,0), (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
		half_cell = 0.5*(self.xlim[1]-self.xlim[0])/self.grid_rows
		for ir in range(x0,x1+1):
			for ic in range(y0,y1+1):
				for con in corners:            
					dx = to_real(ir, self.xlim, self.grid_rows) + con[0]*half_cell - x
					dy = to_real(ic, self.ylim, self.grid_cols) + con[1]*half_cell - y
					dist = np.sqrt(dx**2+dy**2)
					if dist <= rad:
						self.map_design[ir,ic]=1.0
						break
		
						
	def set_a_wall(self,a,b,grids,horizontal=True):

		ax = to_real(a[0], self.xlim, grids)
		ay = to_real(a[1], self.ylim, grids)
		bx = to_real(b[0], self.xlim, grids)
		by = to_real(b[1], self.ylim, grids)

		if self.args.backward_compatible_maps:
			rad = 0.1*np.ones(4)
		elif self.args.random_thickness:
			rad = np.random.normal(loc=self.radius, scale=self.radius*0.1, size=4)
			rad = np.clip(rad, 0, self.radius*0.95)
		else:
			rad = self.radius*np.ones(4)

		corner0 = [ax+rad[0],ay+rad[1]]
		corner1 = [bx-rad[2],by-rad[3]]

		x0 = to_index(corner0[0], self.map_rows, self.xlim)
		y0 = to_index(corner0[1], self.map_cols, self.ylim)

		if self.args.backward_compatible_maps:
			x1 = to_index(corner1[0], self.map_rows, self.xlim)
			y1 = to_index(corner1[1], self.map_cols, self.ylim)
		else:
			x1 = to_index(corner1[0], self.map_rows, self.xlim)+1
			y1 = to_index(corner1[1], self.map_cols, self.ylim)+1

		self.map_2d[x0:x1, y0:y1]=1.0

		x0 = to_index(corner0[0], self.grid_rows, self.xlim)
		y0 = to_index(corner0[1], self.grid_cols, self.ylim)
		x1 = to_index(corner1[0], self.grid_rows, self.xlim)+1
		y1 = to_index(corner1[1], self.grid_cols, self.ylim)+1

		self.map_design[x0:x1, y0:y1]=1.0

	
	def place_turtle(self):
		# new turtle location (random)
		turtle_can = [i for i in range(self.grid_rows*self.grid_cols) if i not in self.taken]
		turtle_bin = np.random.choice(turtle_can,1)

		self.true_grid.row = turtle_bin//self.grid_cols
		self.true_grid.col = turtle_bin% self.grid_cols
		self.true_grid.head = np.random.randint(self.grid_dirs)
		self.goal_pose.x = to_real(self.true_grid.row, self.xlim, self.grid_rows)
		self.goal_pose.y = to_real(self.true_grid.col, self.ylim, self.grid_cols)
		self.goal_pose.theta = wrap(self.true_grid.head*self.heading_resol)
		# if self.args.perturb>0:
		if self.args.init_error == "XY" or self.args.init_error == "BOTH":
			delta_x = (0.5-np.random.rand())*(self.xlim[1]-self.xlim[0])/self.grid_rows
			delta_y = (0.5-np.random.rand())*(self.ylim[1]-self.ylim[0])/self.grid_cols
		else:
			delta_x=0
			delta_y=0
		if self.args.init_error == "THETA" or self.args.init_error == "BOTH":
			delta_theta =  (0.5-np.random.rand())*self.heading_resol
		else:
			delta_theta=0
		
		self.perturbed_goal_pose.x = self.goal_pose.x+delta_x
		self.perturbed_goal_pose.y = self.goal_pose.y+delta_y
		self.perturbed_goal_pose.theta = self.goal_pose.theta+delta_theta

		if self.args.test_mode:
			pg_pose_file = os.path.join(self.args.test_data_path, "pg-pose-%05d.npy"%self.env_count)
			g_pose_file = os.path.join(self.args.test_data_path, "g-pose-%05d.npy"%self.env_count)
			pg_pose = np.load(pg_pose_file)
			g_pose = np.load(g_pose_file)
			self.goal_pose.theta = g_pose[0]
			self.goal_pose.x = g_pose[1]
			self.goal_pose.y = g_pose[2]
			if self.args.init_error == "XY" or self.args.init_error == "BOTH":
				self.perturbed_goal_pose.x = pg_pose[1]
				self.perturbed_goal_pose.y = pg_pose[2]
			else:
				self.perturbed_goal_pose.x = g_pose[1]
				self.perturbed_goal_pose.y = g_pose[2]
			if self.args.init_error == "THETA" or self.args.init_error == "BOTH":
				self.perturbed_goal_pose.theta = pg_pose[0]
			else:
				self.perturbed_goal_pose.theta = g_pose[0]
	 
		self.teleport_turtle()
		self.update_true_grid()
		# self.update_current_pose()

		
	def generate_map_trans(self):
		self.grid_center = ((self.grid_rows-1)/2, (self.grid_cols-1)/2)
		self.map_trans = self.map_design
		self.map_trans = shift(self.map_trans, int(self.grid_center[0]-self.true_grid.row), axis = 0, fill=1.0)
		self.map_trans = shift(self.map_trans, int(self.grid_center[1]-self.true_grid.col), axis = 1, fill=1.0)
		self.map_trans = np.rot90(self.map_trans, -self.true_grid.head)

		
	def get_scan_2d(self):
			  
		data = self.scan_data
		if self.map_rows == None :
			return
		if self.map_cols == None:
			return

		N=self.map_rows
		M=self.map_cols
		self.scan_2d = np.zeros(shape=(N,M))

		x_max = self.xlim[1] # map height/2 in meters
		x_min = self.xlim[0]
		y_max = self.ylim[1]# map width/2 in meters
		y_min = self.ylim[0]

		resol0=min((x_max-x_min)/N,(y_max-y_min)/M)

		angle = data.angle_min
		for i,dist in enumerate(data.ranges):
			resol=resol0
			if ~np.isinf(dist):
				over = 0
				while True:
					x = (dist+over)*np.cos(angle)
					y = (dist+over)*np.sin(angle)
					n = to_index(x, N, self.xlim)
					m = to_index(y, M, self.ylim)
					if not (n>=0 and n<N and m>0 and m<M): break
					self.scan_2d[n,m] = 1.0
					over += resol
			angle += data.angle_increment



	def get_a_scan(self, x_real, y_real, offset=0, scan_step=1, noise=0, sigma=0.05):
		#class member variables: map_rows, map_cols, xlim, ylim, min_scan_range, max_scan_range, map_2d
		
		row_hd = to_index(x_real, self.map_rows, self.xlim)  # from real to hd
		col_hd = to_index(y_real, self.map_cols, self.ylim)  # from real to hd
		scan = np.zeros(360)
		missing = np.random.choice(360, noise, replace=False)
		gaussian_noise = np.random.normal(scale=sigma, size=360)
		for i_ray in range(0,360, scan_step):
			theta = math.radians(i_ray)+offset
			if i_ray in missing:
				dist = np.inf
			else:
				dist = self.min_scan_range
			while True:
				if dist >= self.max_scan_range: 
					dist = np.inf
					break
				x_probe = x_real + dist * np.cos(theta)
				y_probe = y_real + dist * np.sin(theta)
				# see if there's something
				i_hd_prb = to_index(x_probe, self.map_rows, self.xlim)
				j_hd_prb = to_index(y_probe, self.map_cols, self.ylim)
				if i_hd_prb < 0 or i_hd_prb >= self.map_rows: 
					dist = np.inf
					break
				if j_hd_prb < 0 or j_hd_prb >= self.map_cols: 
					dist = np.inf
					break
				if self.map_2d[i_hd_prb, j_hd_prb] >= 0.5: 
					break
				dist += 0.01+0.01*(np.random.rand())
			scan[i_ray]=dist+gaussian_noise[i_ray]
		return scan
		
						
	def get_scan_2d_n_headings(self):
		
		data = self.scan_data
		if self.map_rows == None :
			return
		if self.map_cols == None:
			return

		O=self.grid_dirs
		N=self.map_rows
		M=self.map_cols
		self.scan_2d = np.zeros(shape=(O,N,M))
		# self.scan_2d_rotate = np.zeros(shape=(O,N,M))
		angles = np.linspace(data.angle_min, data.angle_max, data.ranges.size, endpoint=False)

		for i,dist in enumerate(data.ranges):
			for rotate in range(O):
				offset = 2*np.pi/O*rotate
				angle = offset + angles[i]
				if angle > math.radians(self.args.fov[0]) and angle < math.radians(self.args.fov[1]):
					continue
				if ~np.isinf(dist):
					x = (dist)*np.cos(angle)
					y = (dist)*np.sin(angle)
					n = to_index(x, N, self.xlim)
					m = to_index(y, M, self.ylim)
					if n>=0 and n<N and m>0 and m<M:
						self.scan_2d[rotate,n,m] = 1.0

		rows1 = self.args.n_state_grids
		cols1 = self.args.n_state_grids
		rows2 = self.args.n_local_grids
		cols2 = rows2

		center=self.args.n_local_grids/2

		if self.args.binary_scan:
			self.scan_2d_low = np.ceil(normalize(cv2.resize(self.scan_2d[0,:,:], (rows1, cols1),interpolation=cv2.INTER_AREA)))
		else:
			self.scan_2d_low = normalize(cv2.resize(self.scan_2d[0,:,:], (rows1, cols1),interpolation=cv2.INTER_AREA))

		return self.scan_2d, self.scan_2d_low



	def get_scan_2d_noshade(self):
		
		data = self.scan_data
		if self.map_rows == None :
			return
		if self.map_cols == None:
			return

		N=self.map_rows
		M=self.map_cols
		self.scan_2d = np.zeros(shape=(N,M))

		angle = data.angle_min
		for i,dist in enumerate(data.ranges):
			if angle > math.radians(self.args.fov[0]) and angle < math.radians(self.args.fov[1]):
				angle += data.angle_increment
				continue
			if ~np.isinf(dist):
				x = (dist)*np.cos(angle)
				y = (dist)*np.sin(angle)
				n = to_index(x, N, self.xlim)
				m = to_index(y, M, self.ylim)
				if n>=0 and n<N and m>0 and m<M:
					self.scan_2d[n,m] = 1.0
			angle += data.angle_increment
		

	def mask_likelihood(self):
		the_mask = torch.tensor(np.ones([self.grid_dirs, self.grid_rows, self.grid_cols])).float().to(self.device)
		for i in range(self.grid_rows):
			for j in range(self.grid_cols):
				if (i*self.grid_cols+j==self.taken).any():
					the_mask[:,i,j]=0.0
		self.likelihood = self.likelihood * the_mask
		self.likelihood = self.likelihood/self.likelihood.sum()

		
	def product_belief(self):
		if type(self.belief) is np.ndarray:
			self.belief = torch.from_numpy(self.belief).float().to(self.device)
		self.belief = self.belief * (self.likelihood)

		#normalize belief
		self.belief /= self.belief.sum()
		#update bel_grid
		guess = np.unravel_index(np.argmax(self.belief.cpu().detach().numpy(), axis=None), self.belief.shape)
		self.bel_grid = Grid(head=guess[0],row=guess[1],col=guess[2])
	   
		
	def update_target_pose(self):
		self.last_pose.x = self.perturbed_goal_pose.x
		self.last_pose.y = self.perturbed_goal_pose.y
		self.last_pose.theta = self.perturbed_goal_pose.theta

		self.start_pose.x = self.perturbed_goal_pose.x
		self.start_pose.y = self.perturbed_goal_pose.y
		self.start_pose.theta = self.perturbed_goal_pose.theta

		offset = self.heading_resol*self.args.rot_step        
		if self.action_name == "turn_right":
			self.goal_pose.theta = wrap(self.start_pose.theta-offset)
			self.goal_pose.x = self.start_pose.x
			self.goal_pose.y = self.start_pose.y
		elif self.action_name == "turn_left":
			self.goal_pose.theta = wrap(self.start_pose.theta+offset)
			self.goal_pose.x = self.start_pose.x
			self.goal_pose.y = self.start_pose.y
		elif self.action_name == "go_fwd":
			
			self.goal_pose.x = self.start_pose.x + math.cos(self.start_pose.theta)*self.fwd_step
			self.goal_pose.y = self.start_pose.y + math.sin(self.start_pose.theta)*self.fwd_step
			self.goal_pose.theta = self.start_pose.theta
		elif self.action_name == "hold":
			return
		else:
			print('undefined action name %s'%self.action_name)
			exit()
			
		delta_x, delta_y = 0,0
		delta_theta = 0
		if self.args.process_error:
			delta_x, delta_y = np.random.normal(scale=self.args.process_error[0],size=2)
			delta_theta =  np.random.normal(scale=self.args.process_error[1])

		self.perturbed_goal_pose.x = self.goal_pose.x+delta_x
		self.perturbed_goal_pose.y = self.goal_pose.y+delta_y
		self.perturbed_goal_pose.theta = wrap(self.goal_pose.theta+delta_theta)


	def collision_check(self):
		row=to_index(self.perturbed_goal_pose.x, self.grid_rows, self.xlim)
		col=to_index(self.perturbed_goal_pose.y, self.grid_cols, self.ylim)

		x = self.perturbed_goal_pose.x
		y = self.perturbed_goal_pose.y
		rad = self.collision_radius

		if self.args.collision_from == "scan" and self.action_str == "go_fwd":
			self.collision = self.collision_fnc(0, 0, 0, self.scan_2d_slide)
		elif self.args.collision_from == "map":
			self.collision = self.collision_fnc(x,y,rad, self.map_2d)
		else:
			self.collision = False


		if self.collision:
			#undo update target
			self.perturbed_goal_pose.x = self.last_pose.x
			self.perturbed_goal_pose.y = self.last_pose.y
			self.perturbed_goal_pose.theta = self.last_pose.theta


	def collision_fnc(self, x, y, rad, img):
		corner0 = [x+rad,y+rad]
		corner1 = [x-rad,y-rad]
		x0 = to_index(corner0[0], self.map_rows, self.xlim)
		y0 = to_index(corner0[1], self.map_cols, self.ylim)
		x1 = to_index(corner1[0], self.map_rows, self.xlim)
		y1 = to_index(corner1[1], self.map_cols, self.ylim)

		if x0 < 0 :
			return True
		if y0 < 0:
			return True
		if x1 >= self.map_rows:
			return True
		if y1 >= self.map_cols:
			return True
		
		if rad == 0:
			if img[x0, y0] > 0.5 :
				return True
			else:
				return False
		else:
			pass

		for ir in range(x0,x1+1):
			for ic in range(y0,y1+1):
				dx = to_real(ir, self.xlim, self.map_rows) - x
				dy = to_real(ic, self.ylim, self.map_cols) - y
				dist = np.sqrt(dx**2+dy**2)
				if dist <= rad and img[ir,ic]==1.0:
					return True
		return False


	def get_lidar(self):
		ranges = self.get_a_scan(self.current_pose.x, self.current_pose.y, 
								 offset=self.current_pose.theta, 
								 noise=self.args.lidar_noise,
								 sigma=self.args.lidar_sigma)
		bearing_deg = np.arange(360.0)
		mindeg=0
		maxdeg=359
		incrementdeg=1

		params = {'ranges': ranges,
				  'angle_min': math.radians(mindeg),
				  'angle_max': math.radians(maxdeg),
				  'range_min': self.min_scan_range,
				  'range_max': self.max_scan_range}
				  
		self.scan_data = Lidar(**params)
		## scan_data @ unperturbed pose
		x = to_real(self.true_grid.row, self.xlim, self.grid_rows)
		y = to_real(self.true_grid.col, self.ylim, self.grid_cols)
		offset = self.heading_resol*self.true_grid.head
		# ranges = self.get_a_scan(x, y, offset=offset, noise=0, sigma=0)
		ranges = self.get_a_scan(x, y, offset=offset, noise=0, sigma=0.03)
		params = {'ranges': ranges,
				  'angle_min': math.radians(mindeg),
				  'angle_max': math.radians(maxdeg),
				  'range_min': self.min_scan_range,
				  'range_max': self.max_scan_range}
				  
		self.scan_data_at_unperturbed = Lidar(**params)


		
	def fwd_clear(self):
		robot_width = 0.3
		angles=math.degrees(np.arctan2(0.5*robot_width, self.fwd_step))
		ranges = self.scan_data.ranges
			
		if min(ranges[0:int(angles)]) < 1.5*self.fwd_step or min(ranges[-int(angles):]) < 1.5*self.fwd_step:
			return False
		else:
			return True


	def execute_action_teleport(self):
		if self.collision: 
			return False
		self.teleport_turtle()

		return True

	def transit_belief(self):
		self.belief = self.belief.cpu().detach().numpy()
		if self.collision == True:
			self.prior = np.copy(self.belief)
			self.belief = torch.from_numpy(self.belief).float().to(self.device)
			return

		self.belief = self.trans_bel()
		self.belief = torch.from_numpy(self.belief).float().to(self.device)#$ requires_grad=True)
		if type(self.belief).__name__ == 'torch.CudaTensor':
			self.belief = self.belief.cpu().numpy()
		elif type(self.belief).__name__ == 'Tensor':
			self.belief = self.belief.cpu().numpy()
		else:
			self.belief = self.belief
		self.prior = np.copy(self.belief)

	def trans_bel(self):
		

		rotation_step = self.args.rot_step

		if self.action_name == "turn_right":
			self.belief=np.roll(self.belief,-rotation_step, axis=0)
		elif self.action_name == "turn_left":
			self.belief=np.roll(self.belief, rotation_step, axis = 0)
		elif self.action_name == "go_fwd":
			if self.args.trans_belief == "roll":
				self.belief[0,:,:]=np.roll(self.belief[0,:,:], -1, axis=0)
				self.belief[1,:,:]=np.roll(self.belief[1,:,:], -1, axis=1)
				self.belief[2,:,:]=np.roll(self.belief[2,:,:], 1, axis=0)
				self.belief[3,:,:]=np.roll(self.belief[3,:,:], 1, axis=1)

			elif self.args.trans_belief == "stoch-shift" or self.args.trans_belief == "shift":
				prior = self.belief.min()
				for i in range(self.grid_dirs):
					theta = i * self.heading_resol
					fwd_dist = self.args.fwd_step
					dx = fwd_dist*np.cos(theta+np.pi)
					dy = fwd_dist*np.sin(theta+np.pi)
					# simpler way:
					DX = np.round(dx)
					DY = np.round(dy)
					shft_hrz = shift(self.belief[i,:,:], int(DY), axis=1, fill=prior)
					self.belief[i,:,:]=shift(shft_hrz, int(DX), axis=0, fill=prior)
					

		if self.args.trans_belief == "stoch-shift" and self.action_name != "hold":
			for ch in range(self.grid_dirs):
				self.belief[ch,:,:] = ndimage.gaussian_filter(self.belief[ch,:,:], sigma=self.sigma_xy)
			
			n_dir = self.grid_dirs//4
			p_roll = 0.20
			roll_n = []
			roll_p = []
			for r in range(1, n_dir):
				if roll_n == [] and roll_p == []:
					roll_n.append(p_roll*np.roll(bel,-1,axis=0))
					roll_p.append(p_roll*np.roll(bel, 1,axis=0))
				else:
					roll_n.append(p_roll*np.roll(roll_n[-1],-1,axis=0))
					roll_p.append(p_roll*np.roll(roll_p[-1], 1,axis=0))
			self.belief = sum(roll_n + roll_p)+self.belief    
			
		self.belief /= np.sum(self.belief)
		return self.belief

	
	def get_reward(self):
		self.manhattan = self.get_manhattan(self.belief.cpu().detach().numpy()) #manhattan distance between gt and belief.
		self.manhattans.append(self.manhattan)
		self.reward = 0.0
		# if self.args.penalty_for_block and self.action_name == "go_fwd_blocked":

		if self.collision == True:
			self.reward_block_penalty = -1.0
		else:
			self.reward_block_penalty = 0.0

		self.reward_bel_gt = torch.log(self.belief[self.true_grid.head,self.true_grid.row,self.true_grid.col]).cpu().detach().numpy()
		
		self.reward_bel_gt_nonlog = self.belief[self.true_grid.head,self.true_grid.row,self.true_grid.col].cpu().detach().numpy()
		
		bel = torch.clamp(self.belief, 1e-9, 1.0)
		# info gain = p*log(p) - q*log(q)
		infogain = (bel * torch.log(bel)).sum().detach() - self.bel_ent
		self.bel_ent = (bel * torch.log(bel)).sum().detach()
		self.reward_infogain = infogain.cpu().detach().numpy()
		
		bel=self.belief
		self.reward_bel_ent = (bel * torch.log(bel)).sum().cpu().detach().numpy()
		
		if self.manhattan == 0:
			self.reward_hit = 1
		else:
			self.reward_hit = 0
		
		self.reward_dist = (self.longest-self.manhattan)/self.longest
		
		self.reward_inv_dist = 1.0/(self.manhattan+1.0)

		if self.args.penalty_for_block:
			self.reward += self.reward_block_penalty
		if self.args.rew_bel_gt:
			self.reward += self.reward_bel_gt
		if self.args.rew_bel_gt_nonlog:
			self.reward += self.reward_bel_gt_nonlog
		if self.args.rew_infogain:
			self.reward += self.reward_infogain
		if self.args.rew_bel_ent:
			self.reward += self.reward_bel_ent
		if self.args.rew_hit:
			self.reward += self.reward_hit
		if self.args.rew_dist:
			self.reward += self.reward_dist
		if self.args.rew_inv_dist:
			self.reward += self.reward_inv_dist

		self.rewards.append(self.reward)
		

	def get_manhattan(self, bel):
		guess = (self.bel_grid.head, self.bel_grid.row, self.bel_grid.col)
		e_dir = abs(guess[0]-self.true_grid.head)
		e_dir = min(4-e_dir, e_dir)
		return float(e_dir+abs(guess[1]-self.true_grid.row)+abs(guess[2]-self.true_grid.col))


	def update_bel_list(self):
		guess = self.bel_grid
		if guess not in self.bel_list:
			self.new_bel = True
			self.bel_list.append(guess)
		else:
			self.new_bel = False
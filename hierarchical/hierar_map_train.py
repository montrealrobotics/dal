import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from networks import perceptual_conv_real_224_l0, perceptual_conv_real_224_l1
# from logger import Logger
from datetime import datetime
import argparse
import os
import random
import glob
from tensorboardX import SummaryWriter
import time
from datetime import datetime
from arguments import get_args
from copy import deepcopy
from recordtype import recordtype
import math

current_time = datetime.now().strftime('%b%d_%H-%M-%S')

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


class LocalizationNode:
    def __init__(self):
        args = get_args()

        ## For storing informative tensor board logs, and trained models
        self.writer = SummaryWriter(log_dir='runs3a_224/'
            + 'lrpm0_' + str(args.lrpm0)+'_' 
            + 'lrpm1_' + str(args.lrpm1)+'_'
            + 'layers0_' + str(args.layers0)+'_'
            + 'layers1_' + str(args.layers1)+'_'
            + 'bs_' + str(args.batch_size)+'_'
            + 'epochs_' + str(args.epochs)+'_'
            + 'cells' + str(args.cells)+'_'
            + args.criti 
            +'_'+current_time)

        self.modelfile='likelihood_models/'+'model224_3a' + 'lrpm0_' + str(args.lrpm0)+'_' \
        + 'lrpm1_' + str(args.lrpm1)+'_' \
        + 'layers0_' + str(args.layers0)+'_' \
        + 'layers1_' + str(args.layers1)+'_' \
        + 'bs_' + str(args.batch_size)+'_' \
        + 'cells_' + str(args.cells)+'_' \
        + args.criti \
        +'_'+current_time

        self.batch_size = args.batch_size
        self.grid_rows = 11
        self.map_rows = 224
        self.thresh = Variable(torch.Tensor([args.thresh]))
        self.layers0 = args.layers0
        self.layers1 = args.layers1
        self.epochs = args.epochs
        self.criti = args.criti
        self.cells = args.cells

        self.likeli_model_l0 = perceptual_conv_real_224_l0(self.layers0)
        self.likeli_model_l1 = perceptual_conv_real_224_l1(self.layers1)
       
        args.gpu = True
        # use GPU if available
        if args.gpu == True and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            self.device = torch.device("cpu")
            torch.set_default_tensor_type(torch.FloatTensor)

        if self.device == torch.device("cuda") and torch.cuda.device_count() > 1:
            print ("Use", torch.cuda.device_count(), 'GPUs')
            self.likeli_model_l0 = nn.DataParallel(self.likeli_model_l0).cuda()
            self.likeli_model_l1 = nn.DataParallel(self.likeli_model_l1).cuda()
        else:
            print ("Use CPU")

        self.optimizer0 = torch.optim.Adam(list(self.likeli_model_l0.parameters()), lr=args.lrpm0)

        ## you can choose to update level-0 parameters in the same optimizer. 
        self.optimizer1 = torch.optim.Adam(list(self.likeli_model_l1.parameters())+list(self.likeli_model_l0.parameters()), lr=args.lrpm1)
        
        ## loss function. 
        if self.criti == 'mse':
            self.criterion = nn.MSELoss()
        elif self.criti == 'kld':
            self.criterion = nn.KLDivLoss()

        self.args = args
        self.start_time = time.time()

        self.grid_rows, self.grid_cols, self.grid_dirs = 11, 11, 4
        
        self.map_rows, self.map_cols = 224, 224 

        self.max_scan_range = 3.5
        self.min_scan_range = 0.1
        
        self.map_2d = np.load('/home/sai/tb3-anl/dal/env_map.npy') #Load your map here
        
        self.map_2d = (255 - self.map_2d)/255 # You can comment out the following line depending on how you saved your map, or do the required preprocessing
        

        self.taken = np.arange(self.map_2d.size)[self.map_2d.flatten()==1]
        
        self.xlim = (-self.args.xlim, self.args.xlim)
        self.ylim = (-self.args.xlim, self.args.xlim)
        
        self.longest = float(self.grid_dirs/2 + self.grid_rows-1 + self.grid_cols-1)  #longest possible manhattan distance

        self.cell_size = (self.xlim[1]-self.xlim[0])/self.grid_rows
        self.heading_resol = 2*np.pi/self.grid_dirs
        
        self.collision = False
        
        self.scans_over_map = np.zeros((self.grid_rows,self.grid_cols,360))
        self.scans_over_map_high = np.zeros((self.map_rows, self.map_cols, 360))

        self.scan_2d = np.zeros((self.map_rows, self.map_cols))
        
        
        self.likelihood = torch.ones((self.grid_dirs,self.grid_rows, self.grid_cols),device=torch.device(self.device))
        self.likelihood = self.likelihood / self.likelihood.sum()

        self.gt_likelihood = np.ones((self.grid_dirs,self.grid_rows,self.grid_cols))
        self.gt_likelihood_high = np.ones((self.grid_dirs, self.map_rows, self.map_cols))
        self.gt_likelihood_unnormalized = np.ones((self.grid_dirs,self.grid_rows,self.grid_cols)) 
        self.gt_likelihood_unnormalized_high = np.ones((self.grid_dirs, self.grid_rows, self.grid_cols))       
        
        
        self.turtle_loc = np.zeros((self.map_rows,self.map_cols))

        # current pose: where the robot really is. motion incurs errors in pose
        self.current_pose = Pose2d(0,0,0)
        self.goal_pose = Pose2d(0,0,0)
        self.last_pose = Pose2d(0,0,0)
        self.perturbed_goal_pose = Pose2d(0,0,0)        
        self.start_pose = Pose2d(0,0,0)
        #grid pose
        self.true_grid = Grid(head=0,row=0,col=0)
        self.bel_grid = Grid(head=0,row=0,col=0)


    def loop(self): 
        
        self.map_2d = np.load('/home/sai/tb3-anl/dal/env_map.npy')
        ## Uncomment the following 2 lines if you haven't saved the scans_over_map at high resolution before
        # self.get_synth_scan()
        # np.save('/home/sai/montreal_synth_scan.npy', self.scans_over_map_high)
        self.scans_over_map_high = np.load('/home/sai/montreal_synth_scan.npy')

        ## getting locations where turtle can be placed
        turtle_can = [ltc for ltc in range(self.map_rows*self.map_cols) if ltc not in self.taken]

        ## Perumtation of possible locations(so that we access random locations for each map
        turtle_can = np.random.permutation(turtle_can)
        
        for btc in range(100): #epochs        

            ## Get first n random locations and save the six variables
            my_input = np.zeros((self.batch_size, 5, 224, 224))
            gt_likli_low_train = np.zeros((self.batch_size, 4, 11, 11))
            gt_likli_high_train = np.zeros((self.batch_size, 4, 224, 224))
            for j in range(int(turtle_can.shape[0]/self.batch_size)):
                for i in range(self.batch_size): #batch_size

                    # new turtle location (random)
                    turtle_bin = turtle_can[j*self.batch_size +  i]

                    self.true_grid.row = turtle_bin//self.map_rows
                    self.true_grid.col = turtle_bin% self.map_cols
                    self.true_grid.head = np.random.randint(self.grid_dirs)
                    self.goal_pose.x = to_real(self.true_grid.row, self.xlim, self.map_rows)
                    self.goal_pose.y = to_real(self.true_grid.col, self.ylim, self.map_cols)
                    self.goal_pose.theta = wrap(self.true_grid.head*self.heading_resol)
               
                    self.perturbed_goal_pose.x = self.goal_pose.x
                    self.perturbed_goal_pose.y = self.goal_pose.y
                    self.perturbed_goal_pose.theta = self.goal_pose.theta

                    self.teleport_turtle()
                
                    self.get_lidar()       
                # if self.args.shade:
                #     self.get_scan_2d()
                # else:
                    #self.get_scan_2d_noshade()
                    self.get_scan_2d_n_headings()
                ### 2. update likelihood from observation
                    self.generate_map_trans()
                
                # if self.args.dirac_delta_gtl:
                #     self.get_gt_likelihood_dirac()
                    if self.args.gtl_src == 'ld':
                        self.get_gt_likelihood()
                    elif self.args.gtl_src == 'hd-corr':
                        self.get_gt_likelihood_corr()
                    elif self.args.gtl_src == 'hd-cos':
                        self.get_gt_likelihood_cossim()
                    else:
                        raise Exception('GTL source required: --gtl-src= [low-dim-map, high-dim-map]')
                #self.normalize_gtl()
                    my_input[i,0,:,:] = self.map_2d
                    my_input[i,1:5,:,:] = self.scan_2d
                    gt_likli_low_train[i,:,:,:] = self.gt_likelihood
                    gt_likli_high_train[i,:,:,:] = self.gt_likelihood_high
                self.do_train(my_input, gt_likli_low_train, gt_likli_high_train, btc*turtle_can.shape[0] + j * self.batch_size + i)
                
            self.writer.close()    

            ## Getting the new environment
            print("Done with the for loop.")
            self.current_state = "new_env_pose"
            return

        else:
            print("undefined state name %s"%self.current_state)
            self.current_state = None
            exit()

        return

    def teleport_turtle(self):
        self.current_pose.x = self.perturbed_goal_pose.x
        self.current_pose.y = self.perturbed_goal_pose.y
        self.current_pose.theta = self.perturbed_goal_pose.theta


    def normalize_gtl(self):
        gt = self.gt_likelihood
        gt_high = self.gt_likelihood_high
        self.gt_likelihood_unnormalized = np.copy(self.gt_likelihood)
        self.gt_likelihood_unnormalized_high = np.copy(self.gt_likelihood_high)
        if self.args.gtl_output == "softmax":
            gt = softmax(gt, self.args.temperature)
            gt_high = softmax(gt_high, self.args.temperature)
        elif self.args.gtl_output == "softermax":
            gt = softermax(gt)
            gt_high = softermax(gt_high)
        elif self.args.gtl_output == "linear":
            gt = np.clip(gt, 1e-5, 1.0)
            gt_high = np.clip(gt_high, 1e-5, 1.0)
            gt=gt/gt.sum()
            gt_high = gt_high/gt_high.sum()
        self.gt_likelihood = torch.tensor(gt).float().to(self.device)
        self.gt_likelihood_high = torch.tensor(gt_high).float().to(self.device)
        

    def get_gt_likelihood_cossim(self):
        offset = 360/self.grid_dirs
        y= np.array(self.scan_data_at_unperturbed.ranges)[::self.args.pm_scan_step]
        # y= np.array(self.scan_data.ranges)[::self.args.pm_scan_step]
        y = np.clip(y, self.min_scan_range, self.max_scan_range)
        # y[y==np.inf]= self.max_scan_range
        for heading in range(self.grid_dirs):  ## that is, each direction
            #compute cosine similarity at each loc
            X = np.roll(self.scans_over_map, int(-offset*heading),axis=2)[:,:,::self.args.pm_scan_step]
            for i_ld in range(self.grid_rows):
                for j_ld in range(self.grid_cols):
                    if (i_ld*self.grid_cols+j_ld == self.taken).any():
                        self.gt_likelihood[heading,i_ld,j_ld]= 0.0
                    else:
                        x = X[i_ld,j_ld,:]
                        x = np.clip(x, self.min_scan_range, self.max_scan_range)

                        self.gt_likelihood[heading,i_ld,j_ld]= self.get_cosine_sim(x,y)

        for heading in range(self.grid_dirs):  ## that is, each direction
            #compute cosine similarity at each loc
            X = np.roll(self.scans_over_map_high, int(-offset*heading),axis=2)[:,:,::self.args.pm_scan_step]
            for i_ld in range(self.map_rows):
                for j_ld in range(self.map_cols):
                    if (i_ld*self.map_cols+j_ld == self.taken).any():
                        self.gt_likelihood_high[heading,i_ld,j_ld]= 0.0
                    else:
                        x = X[i_ld,j_ld,:]
                        x = np.clip(x, self.min_scan_range, self.max_scan_range)

                        self.gt_likelihood_high[heading,i_ld,j_ld]= self.get_cosine_sim(x,y)


    def get_gt_likelihood_corr(self):
        offset = 360/self.grid_dirs
        # y= np.array(self.scan_data_at_unperturbed.ranges)[::self.args.pm_scan_step]
        y= np.array(self.scan_data.ranges)[::self.args.pm_scan_step]
        y = np.clip(y, self.min_scan_range, self.max_scan_range)
        # y[y==np.inf]= self.max_scan_range
        for heading in range(self.grid_dirs):  ## that is, each direction
            #compute cosine similarity at each loc
            X = np.roll(self.scans_over_map, int(-offset*heading),axis=2)[:,:,::self.args.pm_scan_step]
            for i_ld in range(self.grid_rows):
                for j_ld in range(self.grid_cols):
                    if (i_ld*self.grid_cols+j_ld == self.taken).any():
                        self.gt_likelihood[heading,i_ld,j_ld]= 0.0
                    else:
                        x = X[i_ld,j_ld,:]
                        x = np.clip(x, self.min_scan_range, self.max_scan_range)
                        self.gt_likelihood[heading,i_ld,j_ld]= self.get_corr(x,y)

        for heading in range(self.grid_dirs):  ## that is, each direction
            #compute cosine similarity at each loc
            X = np.roll(self.scans_over_map_high, int(-offset*heading),axis=2)[:,:,::self.args.pm_scan_step]
            for i_ld in range(self.map_rows):
                for j_ld in range(self.map_cols):
                    if (i_ld*self.map_cols+j_ld == self.taken).any():
                        self.gt_likelihood_high[heading,i_ld,j_ld]= 0.0
                    else:
                        x = X[i_ld,j_ld,:]
                        x = np.clip(x, self.min_scan_range, self.max_scan_range)

                        self.gt_likelihood_high[heading,i_ld,j_ld]= self.get_corr(x,y)


        
    def generate_map_trans(self):
        self.grid_center = ((self.grid_rows-1)/2, (self.grid_cols-1)/2)
        self.map_trans = self.map_2d
        self.map_trans = shift(self.map_trans, int(self.grid_center[0]-self.true_grid.row), axis = 0, fill=1.0)
        self.map_trans = shift(self.map_trans, int(self.grid_center[1]-self.true_grid.col), axis = 1, fill=1.0)
        self.map_trans = np.rot90(self.map_trans, -self.true_grid.head)

    def get_cosine_sim(self,x,y):
        # numpy arrays.
         return sum(x*y)/np.linalg.norm(y,2)/np.linalg.norm(x,2)

    def get_corr(self,x,y):
        mx=np.mean(x)
        my=np.mean(y)
        corr=sum((x-mx)*(y-my))/np.linalg.norm(y-my,2)/np.linalg.norm(x-mx,2)
        return 0.5*(corr+1.0)
        # return np.clip(corr, 1e-10, 1.0)
        
    def get_gt_likelihood(self):
        # self.true_grid.head, self.true_grid.row, self.true_grid.col --> virtual scan input from map
        # self.map_design shift as row,col and rotate as head

        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                for o in range(self.grid_dirs):
                    cand = self.map_2d
                    cand = shift(cand, int(self.grid_center[0]-i), axis = 0, fill=-1.0)
                    cand = shift(cand, int(self.grid_center[1]-j), axis = 1, fill=-1.0)
                    cand = np.rot90(cand, -o)
                    match = (1-np.fabs(cand-self.map_trans))*np.array(cand>=0.0,dtype=float)
                    if np.sum(np.array(cand>=0.0,dtype=float),dtype=float) == 0:
                        prob = 0
                    else:
                        prob=np.sum(match,dtype=float)/np.sum(np.array(cand>=0.0,dtype=float),dtype=float)
                    self.gt_likelihood[o,i,j]=prob

        #self.gt_likelihood = softmax(self.gt_likelihood)

        
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

        self.scan_2d_low = np.zeros((self.grid_rows,self.grid_cols),dtype=int)
        row_ticks = np.linspace(start=0,stop=self.map_rows,num=self.grid_rows+1,dtype=int)
        col_ticks = np.linspace(start=0,stop=self.map_cols,num=self.grid_cols+1,dtype=int)
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                patch = self.scan_2d[row_ticks[i]:row_ticks[i+1],col_ticks[j]:col_ticks[j+1]]
                self.scan_2d_low[i,j] = 1 if np.mean(patch) > 0.5 else 0

    def get_a_scan(self, x_real, y_real, offset=0, scan_step=1):
        row_hd = to_index(x_real, self.map_rows, self.xlim)  # from real to hd
        col_hd = to_index(y_real, self.map_cols, self.ylim)  # from real to hd
        scan = np.zeros(360)
        for i_ray in range(0,360, scan_step):
            theta = math.radians(i_ray)+offset
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
            scan[i_ray]=dist
        return scan
        

    def get_synth_scan(self):

        n_places = self.map_rows * self.map_cols
        for i_place in range(n_places):
            row_ld = i_place // self.map_rows
            col_ld = i_place % self.map_cols
            x_real = to_real(row_ld, self.xlim, self.map_rows ) # from low-dim location to real
            y_real = to_real(col_ld, self.ylim, self.map_cols ) # from low-dim location to real
            scan = self.get_a_scan(x_real, y_real,scan_step=self.args.pm_scan_step)
            self.scans_over_map_high[row_ld, col_ld,:] = np.clip(scan, 1e-10, self.max_scan_range)
            if i_place%10==0: print (i_place)
                
        
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
                #print("stuck_here? ", angle)
                if angle > math.radians(self.args.fov[0]) and angle < math.radians(self.args.fov[1]):
                    continue
                if ~np.isinf(dist):
                    x = (dist)*np.cos(angle)
                    y = (dist)*np.sin(angle)
                    n = to_index(x, N, self.xlim)
                    m = to_index(y, M, self.ylim)
                    if n>=0 and n<N and m>0 and m<M:
                        self.scan_2d[rotate,n,m] = 1.0

        return

               
    def mask_likelihood(self):
        the_mask = torch.tensor(np.ones([self.grid_dirs, self.grid_rows, self.grid_cols])).float().to(self.device)
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                if (i*self.grid_cols+j==self.taken).any():
                    the_mask[:,i,j]=0.0
        self.likelihood = self.likelihood * the_mask
        #self.likelihood = torch.clamp(self.likelihood, 1e-9, 1.0)
        self.likelihood = self.likelihood/self.likelihood.sum()

    def get_lidar(self):
        
        ## scan data from current pose (possibly perturbed)
        ranges = self.get_a_scan(self.current_pose.x, self.current_pose.y, offset=self.current_pose.theta)
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
        x = to_real(self.true_grid.row, self.xlim, self.map_rows)
        y = to_real(self.true_grid.col, self.ylim, self.map_cols)
        print("x, y = ", x, y)
        offset = self.heading_resol*self.true_grid.head
        ranges = self.get_a_scan(x, y, offset=offset)
        params = {'ranges': ranges,
                  'angle_min': math.radians(mindeg),
                  'angle_max': math.radians(maxdeg),
                  'range_min': self.min_scan_range,
                  'range_max': self.max_scan_range}
                  
        self.scan_data_at_unperturbed = Lidar(**params)

    
    def do_train(self, input_batch0, target_batch0, target_batch1, i):
        self.likeli_model_l0.train()
        self.likeli_model_l1.train()
        self.optimizer0.zero_grad()
        self.optimizer1.zero_grad()

        input_batch0 = Variable(torch.FloatTensor(input_batch0))
        input_batch0 = input_batch0.to(self.device)
        target_batch0 = Variable(torch.FloatTensor(target_batch0))
        target_batch0 = target_batch0.to(self.device)
        target_batch1 = Variable(torch.FloatTensor(target_batch1))
        target_batch1 = target_batch1.to(self.device)

        
        output0 = self.likeli_model_l0(input_batch0)

        bs, a,b,c = output0.shape # get the output shape
        u = torch.reshape(output0, (-1, a*b*c))
        _, idx = torch.topk(output0.view(output0.shape[0], -1), dim=-1, k=self.cells) 
        x = idx/(b*c)
        y = (idx%(b*c))/b
        z = (idx%(b*c))%c

        output1 = torch.zeros((bs, 4, 224, 224))
        for btc in range(self.cells):
            scan_cut = torch.zeros((bs, 4, 160, 160))
            map_cut = torch.zeros((bs, 160, 160))
            for eth in range(bs):
                dire, row, col = x[eth,btc], y[eth,btc], z[eth,btc]
                
                ## Cut a square patch of size 160x160 around (row, col). If the patch is going beyond map size, then cut it at the boundaries. 
                if row*20 - 80 >= 0:
                    row_min = row*20 - 80
                else:
                    row_min = 0
                if row*20 + 80 <= 224:
                    row_max = row*20 + 80
                else:
                    row_max = 224

                if col*20 - 80 >= 0:
                    col_min = col*20 - 80
                else:
                    col_min = 0
                if col*20 + 80 <= 224:
                    col_max = col*20 + 80
                else:
                    col_max = 224

                scan_cut[eth, :, 0:row_max-row_min, 0:col_max-col_min] = input_batch0[eth, 1:5, row_min:row_max, col_min:col_max]
                map_cut[eth, 0:row_max-row_min, 0:col_max-col_min] = input_batch0[eth, 0, row_min:row_max, col_min:col_max]

            input_batch1 = torch.zeros((bs, 5, 160, 160))
            input_batch1[:,0,:,:] = map_cut
            input_batch1[:,1:5,:,:] = scan_cut
            output_cut = self.likeli_model_l1(input_batch1)
            weight = output0[:,dire,row,col].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            output1[:, :, row*20:(row+1)*20, col*20:(col+1)*20] = weight *output_cut

        ## loss for level-0
        loss0 = self.criterion(output0, target_batch0)
        if self.criti == 'kld':
            loss0 = abs(loss0)
        loss0.backward(retain_graph=True)
        self.optimizer0.step()

        ## loss for level-1
        loss1 = self.criterion(output1, target_batch1)
        if self.criti == 'kld':
            loss1 = abs(loss1)
        loss1.backward()
        self.optimizer1.step()

        ## Tensor board logging
        self.writer.add_scalar('loss0', loss0, i)
        self.writer.add_scalar('loss1', loss1, i)
        print(loss0, loss1)


        str_date_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.logfile = 'anl_output-%s.txt'%str_date_time
        self.pm_filepath0='/home/sai/hierar/hierarchical/scripts/montreal_hierar/' + self.modelfile+'_0.model'
        self.pm_filepath1='/home/sai/hierar/hierarchical/scripts/montreal_hierar/' + self.modelfile+'_1.model'
        torch.save(self.likeli_model_l0, self.pm_filepath0)
        torch.save(self.likeli_model_l1, self.pm_filepath1)

        print("backprop done")
        

lol = LocalizationNode()
lol.loop()
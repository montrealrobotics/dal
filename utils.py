import torch
import torch.nn as nn

import gym
from gym import error, spaces
import numpy as np
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
# from a2c_ppo_acktr.envs import VecNormalize
import numpy as np
import matplotlib.pyplot as plt
import cv2



## gym environment utils
class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class WarpObs(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 88
        self.height = 88
        self.observation_space = spaces.Box(low=0, high=1,
            shape=(6, self.width, self.height), dtype=np.float32)

    def observation(self, frame):
        return frame


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


## DAL utils
def to_real(i, mm, n):
    u = (mm[1]-mm[0])/n
    u0 = u/2
    return mm[1]-u*i - u0


def to_index(a, N, mm):
    a_max = mm[1]
    a_min = mm[0]
    return int(np.floor(N*(a_max-a)/(a_max-a_min)))


def create_circular_mask(h, w, center=None, radius=None, angle=None, thick=0):
    # img = np.random.randint(0,2,(4,224,224))
    # print (img.shape)
    # mask = create_circular_mask(224,224, center = (100,100), radius = 20)
    # img[3,~mask] = 0
    # plt.imshow(img[3,:,:])
    # plt.show()

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    if angle is None:
        mask = (dist_from_center <= radius)
    else:
        angle_from_center = np.arctan2(Y-center[1],X-center[0])
        mask = (dist_from_center <= radius) & (np.abs(np.unwrap(angle_from_center-angle)) * dist_from_center <= thick)
        #(angle_from_center < angle+0.15/(dist_from_center+0.01)) & (angle_from_center > angle-0.15/(dist_from_center+0.01))
    return mask

def square_clock(x, n):
    width = x.shape[2]
    height = x.shape[1]
    quater = n/4-1

    #even/odd
    even = 1 - quater % 2
    side = quater+2+even
    N = side*max(width,height)
    img = np.zeros((N,N))
        
    for i in range(n):
        s = (i+n/8)%n
        if s < n/4:
            org = (0, n/4-s)
        elif s < n/2:
            org = (s-n/4+even, 0)
        elif s < 3*n/4:
            org = (n/4+even, s-n/2+even)
        else:
            org = (n/4-(s-3*n/4), n/4+even)
        ox = org[0]*height
        oy = org[1]*width
        img[ox:ox+height, oy:oy+width] = x[i,:,:]
    del x
    return img, side

def square_array(img, n):
    output = np.zeros((n*img.shape[1], n*img.shape[2]),np.float32)

    for i in range(n):
        for j in range(n):
            output[i*n:(i+1)*n,j*n:(j+1)*n]=img[i*n+j]
    return output

def wrap(phase):
    # wrap into [-pi, pi]
    phase = ( phase + np.pi) % (2 * np.pi ) - np.pi
    return phase

def wrap_2pi(phase):
    # wrap into [0, 2*pi]
    phase = ( phase) % (2 * np.pi )
    return phase

def distort_map(img, rows, cols):
    # done making a map for LM : self.map_for_LM
    # try erode and dilate
    
    kernel = np.array([[0,1,1,1,0],
                       [1,1,1,1,1],
                       [1,1,1,1,1],
                       [1,1,1,1,1],
                       [0,1,1,1,0]], np.uint8)

    # expand img
    boarder = img-cv2.erode(img,np.ones((3,3),np.uint8),iterations=1)

    img1 = np.clip(boarder,0,1)
    # leave portion
    img1 = np.clip(img1 * (np.random.rand(rows,cols) > 0.5).astype(int) + img-boarder, 0, 1)
    img1 = cv2.dilate(img1, kernel, iterations = 1)

    # shrink
    img2 = cv2.erode(img1, kernel, iterations = 1)
    # img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
    img3 = img2
    n = np.random.randint(5)
    for _ in range(n):
        img3 = img3 * (np.random.rand(rows,cols) > 0.33).astype(int)
        #img3 = img2*np.random.randint(2,size=[224,224])
        img3 = cv2.dilate(img3, kernel, iterations = 2)
        img3 = cv2.erode(img3, kernel, iterations = 2)
        
    return img3


    
def fill_outer_rim(img, rows, cols):
    for i in range(rows):
        j = 0
        while img[i,j] == 0:
            img[i,j] = 1.0
            j = j+1
            if j>=cols:
                break

    for i in range(rows):
        j = -1
        while img[i,j] == 0:
            img[i,j] = 1.0
            j = j-1
            if j < -cols:
                break
            
    for j in range(cols):
        i = 0
        while img[i,j] == 0:
            img[i,j] = 1.0
            i = i + 1
            if i >= rows:
                break

    for j in range(cols):
        i = -1
        while img[i,j] == 0:
            img[i,j] = 1.0
            i = i - 1
            if i < -rows:
                break
    return img
    
def transform(s,g,t,q):
    d = g-s
    theta = np.arctan2(d[1],d[0])
    R = np.array(
        [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]])
    b=t-g
    q=wrap(q-theta)
    return np.append(R.dot(b),q)

def control_law(fwd_err,lat_err,ang_err, t_elapse):
    lin_gain = 0.8
    rot_gain = 2.0 #k
    time_gain = 0.5
    min_ang_vel = - 0.01 * np.sign(ang_err)
    
    eps = 0.05
    max_lin_vel = 0.3
    max_lat_err = 0.05
    min_offset_lin_vel = -0.03 * np.sign(fwd_err)
    lin_vel = -lin_gain*fwd_err+min_offset_lin_vel
    lin_vel = np.clip(lin_vel, -max_lin_vel, min(max_lin_vel,t_elapse*time_gain))

    slide_var = wrap(ang_err+np.arctan(rot_gain*lat_err))
    if np.abs(slide_var) > np.arctan(rot_gain*max_lat_err):
        lin_vel = 0.0

    ang_gain = 0.5 #0.4+np.abs(lin_vel)*0.5 #b1+w_bar
    ang_vel = - rot_gain*lin_vel*np.sin(ang_err)/(1.0+(rot_gain*lat_err)**2) \
              - ang_gain*np.clip(slide_var/eps,-1.0,1.0) \
              + min_ang_vel
    return lin_vel, ang_vel

def define_tf(src, tgt):
    # src = observed pose from source coordinates 
    # tgt = observed pose from target coordinates 
    # if you need robot pose from map coordinates given that from odom coordinates
    # get map_T_robot = map_T_odom * odom_T_robot: you need map_T_odom
    # map_T_odom = define_tf(map, odom)
    src_T_obs = np.zeros((3,3), dtype=np.float32)
    tgt_T_obs = np.zeros((3,3), dtype=np.float32)
    src_T_obs[-1,-1] = 1.0
    tgt_T_obs[-1,-1] = 1.0

    theta = src[2]
    src_T_obs[:2,:2] = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]])
    src_T_obs[:2,2] = np.array([src[0], src[1]],dtype=np.float32).transpose()

    theta = tgt[2]
    tgt_T_obs[:2,:2] = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]])
    tgt_T_obs[:2,2] = np.array([tgt[0], tgt[1]],dtype=np.float32).transpose()

    # print (tgt_T_obs)
    src_T_tgt = np.dot(src_T_obs,  np.linalg.inv(tgt_T_obs))
    #src_T_tgt = np.dot(src_T_obs,  inv_tf(tgt_T_obs))

    return src_T_tgt

def inv_tf(T):
    R = T[:2, :2]
    p = T[:2, 2]
    out = np.zeros_like(T)
    out[:2,:2]=R.transpose()
    out[:2, 2]=-np.dot(R.transpose(), p)
    out[2,2] = 1.0
    
    return out

def tuple_to_hg(p):
    T = np.zeros((3,3), dtype=np.float32)
    T[-1,-1] = 1.0
    theta = p[2]
    T[:2,:2] = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]])
    T[:2,2] = np.array([p[0], p[1]],dtype=np.float32).transpose()
    return T

def hg_to_tuple(T):
    theta = np.arctan2(T[1,0], T[0,0])
    x = T[0,2]
    y = T[1,2]
    return (x,y,theta)
    
if __name__ == "__main__":
    img = (np.random.rand(224,224)>0.75).astype(int)

    # print (img.shape)
    # angle = np.pi*np.random.rand()
    x,y = np.random.randint(0, 224, 2)
    angle = np.arctan2(y-112,x-112)+np.pi/2
    mask = create_circular_mask(224,224, center = (x,y), radius = 20, angle=angle, thick=3)
    img[~mask] = 0

    plt.imshow(img[:,:])
    plt.title('angle=%s'%np.rad2deg(angle))
    plt.plot(x,y,'o')
    plt.show()



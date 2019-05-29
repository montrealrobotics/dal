import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt

def normalize(x):
    if x.min() == x.max():
        return 0.0*x
    x = x-x.min()
    x = x/x.max()
    return x

map_for_LM = np.load('/home/sai/hierar/quiet_room_modified_keehong.npy')
mdt = cv2.resize(map_for_LM,(88,88), interpolation=cv2.INTER_AREA)
mdt = normalize(mdt)
mdt = np.clip(mdt, 0.0, 1.0)
np.save('/home/sai/hierar/real_map_88.npy', mdt )

plt.imshow(map_for_LM, interpolation='nearest')
plt.show()
# map_for_RL = torch.tensor(mdt).float().to(self.device)
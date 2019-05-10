import os
import platform

class Config(object):
    map_width = 384
    map_height = 384
    laser_scan_dim = 360
    num_orientations = 4
    h1 = 32
    h2 = 16
    h3 = 16
    h4 = 32

    ROOTDIR = '~/catkin_ws/src/tb3-anl/tb3_anl/'
    GAMEPATH = os.path.join(ROOTDIR, 'saved_games')
    NETPATH = os.path.join(ROOTDIR, 'saved_nets')

    minibatch_size = 32

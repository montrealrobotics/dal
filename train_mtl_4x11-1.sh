#!/bin/sh
# --collision-radius: robot radius for collision checking
# -n N1 N2 N3: N1 maps N2 episodes N3 steps
# --lidar-noise: select number of random laser rays that are not detected correctly and have inf.
# --map-pixel: size of a pixel in the map (in meters)
# --sigma-xy: uncertainly of transition model. std-dev of gaussian blur. it really affects the convergence of belief.
# --prob-roll-out (-pro): probability of saving the data at each step. set to 1.0 if want to save likelihood, GTL, belief. If not, set to 0.0
# --RL-type: 0 for our original setup.
# --load-map: load a map (nupy matrix) if you have one ('randombox' to use random box map)
# --lidar-sigma: scale of gaussian noise to scan vector
# --flip-map: num of random pixels to flip in the map
# --distort-map: erode/dilate the map.
# -f: show figure
# --no-save: don't save logs and figures.
# if you don't want to schedule LM training gains remove the line with --schedule-pm ......
# -pbs: LM model batch size. collect loss for how many steps. RL model updates after each episode.
# --fov: it defines the heading range that is missing from lidar. set --fov 130 230 for turtlebot3 in montreal
# --process-error (-pe) [xy scale] [theta scale]: motion results in errors that accumulate during an episode. normal distribution. std deviations in meters and radians.
# --init-error( -ie): put robot off the center of the grid. uniform distribution.
# --block-penalty=0.1 : it gives penalty of 0.1 when tried to go fwd against an obstacle, based on scan image.
# you can also use dal_ros_aml.py: does the same as dal.py

SAVE_LOC=$3
UPDATE=''

## this loads random mazes:
MAP="maze"
## this loads random box rooms:
# MAP="randombox"

## this loads custom maps: the file should be saved with np.save() with 0=open space, 1=occupied.
# MAP=maps/mlab-02-map-224x224.npy

## if you are trainging from the scratch:
RLMODEL=none
PMMODEL=none

# if you resume from or test some models 
#RLMODEL='RL/rl.model'
#PMMODEL='LM/densenet121-Pod.mdl'

PMNET='densenet121'
BEL_GRIDS=11
HEADINGS=4
LM_GRIDS=11
EPISODE_LENGTH=15
N_EPISODE=$2
LP=1e-4

if [ $1 = train_lm ]; then
    UPDATE=$UPDATE' --update-pm-by=GTL '
    UPDATE=$UPDATE' --schedule-pm --pm-step-size=1000 --pm-decay=0.5 '
    UPDATE=$UPDATE' -lp='$LP' -pbs=10 --temp=0.1 '
fi

if [ $1 = train_rl ]; then
    UPDATE=$UPDATE' --update-rl -lr=1e-4 --temp=0.1'
    UPDATE=$UPDATE' --schedule-rl --rl-step-size=1000 --rl-decay=0.5 '
fi

if [ $1 = train_both ]; then
    UPDATE=$UPDATE' --update-pm-by=GTL '
    # UPDATE=$UPDATE' --schedule-pm --pm-step-size=200 --pm-decay=0.5 '
    UPDATE=$UPDATE' -lp='$LP' -pbs=10 --temp=0.1 '
    UPDATE=$UPDATE' --update-rl -lr=1e-4  '
    # UPDATE=$UPDATE'--schedule-rl --rl-step-size=100 --rl-decay=0.5 '
fi

if [ $1 = train_both_dry ]; then
    UPDATE=$UPDATE' --update-pm-by=GTL '
    # UPDATE=$UPDATE' --schedule-pm --pm-step-size=200 --pm-decay=0.5 '
    UPDATE=$UPDATE' -lp='$LP' -pbs=10 --temp=0.1 '
    UPDATE=$UPDATE' --update-rl -lr=1e-4  '
    # UPDATE=$UPDATE' --schedule-rl --rl-step-size=100 --rl-decay=0.5 '
    UPDATE=$UPDATE' -f --no-save '
fi

if [ $1 = test ]; then
    UPDATE=' -f --no-save --temp=0.1 '
fi

if [ $1 = aml ]; then
    UPDATE=' --no-save --temp=0.1 --use-aml '
fi

if [ $1 = test-navi ]; then
    UPDATE=' -f --no-save --temp=0.1 --navigate-to 2 7 --init-pose 0 6 3 '
fi

if [ $1 = test_with_gtl ]; then
    UPDATE=' -f --no-save --temp=0.1 -ugl '
fi

if [ $4 = simple ]; then
    MAP_OPTIONS=' -bcm '
fi

if [ $4 = complex ]; then
    MAP_OPTIONS=' --random-thickness --distort-map '
fi

# If you want to train without popping up the figure window but want to save the figures, uncomment:
# MPLBACKEND='AGG' \
python sim/dal.py \
 -v 1 -ug -cr=0.20 --collision-from='scan' \
 -n 1 $N_EPISODE $EPISODE_LENGTH \
 --lidar-noise=10 \
 --lidar-sigma=.03 \
 --load-map=$MAP \
 $MAP_OPTIONS \
 --map-pixel=0.039 \
 --sigma-xy=.5 --trans-belief=stoch-shift \
 --pm-scan-step=3 --gtl-src=hd-cos \
 --n-lm-grids=$LM_GRIDS --n-state-grids=11 --n-state-dirs=4 \
 -fs=1 -rs=1 \
 -nh=$HEADINGS --n-local-grids=$BEL_GRIDS \
 --prob-roll-out=0.00 \
 --rew-bel-gt-nonlog \
 --block-penalty=0.1 --process-error 0.0 0.01 \
 --RL-type=0 \
 --pm-model=$PMMODEL \
 --rl-model=$RLMODEL \
 --pm-net=$PMNET -ch3=ZERO --drop-rate=0.1 \
 --save-loc=$SAVE_LOC \
 --fov 130 230 \
 --init-error=NONE \
 $UPDATE \




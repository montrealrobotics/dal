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


DIM=$1
DIRS=$2
MODE=$3
N_EPISODE=$4
SAVE_LOC=$5
UPDATE=''

## this loads random mazes:
#MAP="maze"

## this loads random box rooms:
# MAP="randombox"

## this loads the SAIT maps:
## the file was saved with np.save() with 0=open space, 1=occupied.
MAP='/home/sai/hierar/qroom_real_map_88.npy'

## to train from the scratch:
RLMODEL=none
PMMODEL0=none
PMMODEL1=none

#RLMODEL=RL/rl.model

if [ $DIM = '11' ]; then
    #PMMODEL=LM/SharonCarter-densenet121-11x11.mdl
    #PMNET='densenet121'
    PMNET0=none
    PMNET1=none
    BEL_GRIDS=11
    LM_GRIDS=11
    GOAL='2 7'    
fi

if [ $DIM = '33' ]; then
    #PMMODEL=LM/densenet201-SpiderMan.mdl
    #PMNET='densenet201'
    PMNET0=none
    PMNET1=none
    BEL_GRIDS=33
    LM_GRIDS=33
    GOAL='6 21'
fi

HEADINGS=$DIRS

LP=1e-3
EPISODE_LENGTH=25
N_MAPS=1

if [ $MODE = train_lm ]; then
    EPISODE_LENGTH=2
    UPDATE=$UPDATE' --update-pm0-by=GTL '
    UPDATE=$UPDATE' --update-pm1-by=GTL '
    UPDATE=$UPDATE' --schedule-pm --pm-step-size=1000 --pm-decay=0.5 '
    UPDATE=$UPDATE' -lp0='$LP' -lp1='$LP' -pbs=30 --temp=0.1 '
fi

if [ $MODE = train_rl ]; then
    EPISODE_LENGTH=25    
    UPDATE=$UPDATE' --update-rl -lr=1e-4 --temp=0.1'
    UPDATE=$UPDATE' --schedule-rl --rl-step-size=1000 --rl-decay=0.5 '
fi

if [ $MODE = train_both ]; then
    EPISODE_LENGTH=10
    UPDATE=$UPDATE' --update-pm-by=GTL '
    UPDATE=$UPDATE' --schedule-pm --pm-step-size=1000 --pm-decay=0.5 '
    UPDATE=$UPDATE' -lp='$LP' -pbs=10 --temp=0.1 '
    UPDATE=$UPDATE' --update-rl -lr=1e-4  '
    UPDATE=$UPDATE'--schedule-rl --rl-step-size=1000 --rl-decay=0.5 '
fi

if [ $MODE = train_both_dry ]; then
    EPISODE_LENGTH=10
    UPDATE=$UPDATE' --update-pm-by=GTL '
    UPDATE=$UPDATE' --schedule-pm --pm-step-size=1000 --pm-decay=0.5 '
    UPDATE=$UPDATE' -lp='$LP' -pbs=10 --temp=0.1 '
    UPDATE=$UPDATE' --update-rl -lr=1e-4  '
    UPDATE=$UPDATE' --schedule-rl --rl-step-size=1000 --rl-decay=0.5 '
    UPDATE=$UPDATE' -f --no-save '
fi

if [ $MODE = test ]; then
    UPDATE=' -f --no-save --temp=0.1 '
fi

if [ $MODE = test-save ]; then
    UPDATE=' -f --temp=0.1 '
fi

if [ $MODE = test-gtl-save ]; then
    UPDATE=' -f --temp=0.1 -ugl '
fi

if [ $MODE = aml ]; then
    UPDATE=' --no-save --temp=0.1 --use-aml '
fi

if [ $MODE = test-navi ]; then
    UPDATE=' -f --no-save --temp=0.1 --navigate-to '$GOAL' '
fi

if [ $MODE = test_with_gtl ]; then
    UPDATE=' -f --no-save --temp=0.1 -ugl '
fi

# If you want to train without popping up the figure window but want to save the figures, uncomment:
MPLBACKEND='AGG' \
python sim/dal_hle2_kee.py \
 -v 1 -ug -cr=0.20 --collision-from='scan' \
 -n $N_MAPS $N_EPISODE $EPISODE_LENGTH \
 --lidar-noise=20 \
 --load-map=$MAP \
 --map-pixel=0.040 \
 --scan-range 0.10 3.5 \
 --sigma-xy=.66 --trans-belief=stoch-shift \
 --pm-scan-step=30 --gtl-src=hd-cos \
 --n-lm-grids=$LM_GRIDS --n-state-grids=11 --n-state-dirs=4 \
 -fs=1 -rs=1 \
 -nh=$HEADINGS --n-local-grids=$BEL_GRIDS \
 --prob-roll-out=0.00 \
 --rew-bel-gt-nonlog \
 --block-penalty=0.1 --process-error 0.0 0.01 \
 --RL-type=0 \
 --pm-model0=$PMMODEL0 \
 --pm-model1=$PMMODEL1 \
 --rl-model=$RLMODEL \
 --pm-net0=$PMNET0 -ch3=ZERO --drop-rate=0.1 \
 --pm-net1=$PMNET1 -ch3=ZERO --drop-rate=0.1 \
 --save-loc=$SAVE_LOC \
 --init-error=NONE \
 $UPDATE \


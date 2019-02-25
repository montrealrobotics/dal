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

UPDATE=''

if [ $1 = train_lm ]; then
    UPDATE='  --update-pm-by=GTL '
    UPDATE=$UPDATE' --schedule-pm --pm-step-size=1000 --pm-decay=0.5 '
    UPDATE=$UPDATE' -lp=1e-4 -pbs=10 '
fi

if [ $1 = train_rl ]; then
    UPDATE=' --update-rl -lr=1e-4 '
    UPDATE=$UPDATE' --schedule-rl --rl-step-size=1000 --rl-decay=0.5 '
fi

if [ $1 = train_both ]; then
    UPDATE='  --update-pm-by=GTL '
    UPDATE=$UPDATE' --schedule-pm --pm-step-size=1000 --pm-decay=0.5 '
    UPDATE=$UPDATE'-lp=1e-4 -pbs=10 '
    UPDATE=$UPDATE'--update-rl -lr=1e-4  '
    UPDATE=$UPDATE'--schedule-rl --rl-step-size=1000 --rl-decay=0.5 '
fi

if [ $1 = test ]; then
    UPDATE=''
fi


# If you want to train without popping up the figure window but want to save the figures, uncomment:
# MPLBACKEND='AGG' \
python src/dal.py \
 -v 1 -ug -cr=0.25 --collision-from='scan' \
 -n 1000 10 30 \
 --lidar-noise=10 \
 --lidar-sigma=.03 \
 --flip-map=100 \
 --load-map='/home/sai/tb3-anl/dal/env_map.npy' \
 --map-pixel=0.04 \
 -temp=0.1 \
 --sigma-xy=0.5 --trans-belief=stoch-shift \
 --pm-scan-step=30 --gtl-src=hd-cos \
 --n-lm-grids=11 --n-state-grids=11 --n-state-dirs=4 \
 -fs=1 -rs=1 \
 -nh=4 --n-local-grids=11 \
 --prob-roll-out=0.00 \
 --rew-bel-gt-nonlog \
 --block-penalty=0.1 \
 --RL-type=0 \
 -lp=1e-4 -pbs=10 \
 --update-rl -lr=1e-4 --rl-model=RL/rl.model \
 --update-pm-by=GTL \
 --schedule-pm --pm-step-size=1000 --pm-decay=0.5 \
 -f \
 --init-error=XY --process-error 0.05 0.01 \
 --fov 130 230 \
 --pm-model=LM/densenet121-Pod.mdl \
 --pm-net=densenet121 -ch3=ZERO --drop-rate=0.1 \
 --rl-model=RL/rl.model \
 $UPDATE

 # -ugl \
 # --markov \

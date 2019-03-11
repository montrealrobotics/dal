# This folder contains a DAL Simulator. 
- `dal.py` runs simulation without ROS.
- `dal_ros_aml.py` can do what dal.py can do for simulation and also run a real robot with ROS.
(Requires ROS and rospy)


# To run DAL simulation (`dal.py`) for training/evaluation on the SAIT map, you can do the following from the parent folder `dal/`:

run `sh run_sait_map.sh [GRIDS] [DIRS] [MODE] [N_EPISODE] [OUTPUT-DIR]`
where
- `GRIDS`: choose 11 or 33, and the script will load the trained models accordingly

- `DIRS`: Number of directions. Recommend trying with one of 4,8,12,16,24,36.

- `MODE`: can be one of the following

: `test`, `train_both`, `train_lm`, `train_rl`, `aml`, `test-navi`, `test_with_gtl`

# To run DAL simulation (dal.py) for training/evaluation on random maze-like maps, you can do the following:

from the parent folder `dal/` run
`sh train_mtl_4x11.sh [MODE] [N_EPISODE] [OUTPUT_DIR] [MAP_OPTION]`
where `[MODE]` is one of the following
- `test`: test trained models. (specify the models in the file `train_mtl_4x11.sh')
- `train_both`: train LM and RL models
- `train_lm`: train LM only
- `train_rl`: train RL only

`[N_EPISODE]` is the number of episodes
`[OUTPUT_DIR]` is your output directory. Output logs and trained models are saved under `./[OUTPUT_DIR]/YYYYMMDD-HHMMSS-NN/`. (Create `[OUTPUT_DIR]` first.)

`[MAP_OPTION]` is either one of these:
- `simple`
- `complex`

You can edit `train_mtl_4x11.sh` to load different maps and trained models or change learning rates.



This folder contains dal_ros_aml.py and dal.py. 

dal.py can run simulation without ROS dependency.

dal_ros_aml.py can do what dal.py can do for simulation and also run a real robot with ROS.

to run dal.py for training/evaluation, you can do the following:
- from the parent folder, run `sh train_mtl_4x11.sh test 10 ./output` for testing trained models
- from the parent folder, run `sh train_mtl_4x11.sh train_both 10 ./output` for training perceptual model and policy
- from the parent folder, run `sh train_mtl_4x11.sh train_lm 10 ./output` for training perceptual model
- from the parent folder, run `sh train_mtl_4x11.sh train_rl 10 ./output` for training policy

10 is the number of episodes, which you can change.

Output logs and trained models are saved under `./output/YYYYMMDD-HHMMSS-NN/`. (Create `./output` first.)

You can edit `train_mtl_4x11.sh` to load maps and trained models or change learning rates.

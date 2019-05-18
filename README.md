# Deep Active Localization (DAL)
Repository for Deep Active Localization research and benchmarks. You can read our full paper on arxiv: https://arxiv.org/abs/1903.01669
- (Minor Note: We are still working on some minor documentations of the code, but it is ready to use)

For training or testing on our custom simulator, see: sim/readme.md

Requirements:
- Python 3.5+
- Pytorch 1.0
- OpenAI Gym
- Numpy
- tensorboardX

Please use this bibtex if you want to cite this repository in your publications:
```
@inproceedings{Krishna2019DeepAL,
  title={Deep Active Localization},
  author={Sai Krishna and Keehong Seo and Dhaivat Bhatt and Vincent Mai and Krishna Murthy and Liam Paull},
  year={2019}
}
```

## Installation

Clone this repository and install the dependencies with `pip3`:

```
git clone https://github.com/montrealrobotics/dal
cd dal
pip3 install -e .
```

## Basic Usage
For running our gym environment (which we call `dal-v0`), you can just do `python main.py` (More instructions will come soon but the code and parameters used are mostly self explanatory. you can also look at a2c_ppo_acktr/arguments.py)

## Acknowledgements
- Gazebo
- pytorch
- openAI gym
- Ikostrikov's baselines repo
- BabyAI

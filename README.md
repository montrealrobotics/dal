# Deep Active Localization (DAL)
Repository for Deep Active Localization research and benchmarks. Accepted to RAL. https://ieeexplore.ieee.org/abstract/document/8784238, https://arxiv.org/abs/1903.01669

Requirements:
- Python 3.5+
- Pytorch 1.0
- OpenAI Gym
- Numpy
- tensorboardX

Please use this bibtex if you want to cite this repository in your publications:
```
@article{gottipati2019deep,
 title={Deep Active Localization},
 author={Gottipati, Sai Krishna and Seo, Keehong and Bhatt, Dhaivat and Mai, Vincent and Murthy, Krishna and Paull, Liam},
 journal={IEEE Robotics and Automation Letters},
 volume={4},
 number={4},
 pages={4394--4401},
 year={2019},
 publisher={IEEE}
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
For training or testing on our custom simulator, see: sim/readme.md

## Acknowledgements
- Gazebo
- pytorch
- openAI gym
- Ikostrikov's baselines repo
- BabyAI

import argparse

import torch


def get_args_iko():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=1,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--env-name', default='dal-v0',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--ir-coef', type=float, default=1.0,
                        help='Coefficient for the intrinsic reward (default: 1.0)')
    parser.add_argument('--singh-coef', type=float, default=0.6,
                        help='Coefficient for the singhs intrinsic reward (default: 0.6)')
    parser.add_argument('--use-singh', action='store_true', default=False,
                        help='Learn singhs intrinsic reward')

     ## GENERAL
    parser.add_argument("-c", "--comment", help="your comment", type=str, default='')
    parser.add_argument("--realbot", action="store_true")
    parser.add_argument("--save-loc", type=str, default="data/anl_logs")
    parser.add_argument("--generate-data", action="store_true")

    
    ## MAPS, EPISODES, MOTIONS
    parser.add_argument("-n", "--num", help = "num envs, episodes, steps", nargs=3, default=[10, 10, 20], type=int)    
    parser.add_argument("-ms", "--map-size", help="low res grid map size, default=11", type=int, default=11)
    parser.add_argument("-sr", "--sub-resolution", type=int, default=1)
    parser.add_argument("--n-headings", "-nh", type=int, default=4)
    parser.add_argument("--rm-cells", help="num of cells to delete from maze", type=int, default=11)
    parser.add_argument("--random-rm-cells", type=int, nargs=2, default=[0,0])
    parser.add_argument("--backward-compatible-maps","-bcm", action="store_true")
    parser.add_argument("--random-thickness", action="store_true")
    parser.add_argument("--thickness", type=float, default=0.0)


    ## Error Sources:
    ## 1. initial pose - uniform pdf
    ## 2. odometry (or control) - gaussian pdf
    ## 3. use scenario: no error or init error + odom error accumulation
    parser.add_argument("-ie", "--init-error", type=str, choices=['NONE','XY','THETA','BOTH'],default='NONE')
    parser.add_argument("-pe", "--process-error", action="store_true")
    parser.add_argument("--fov", help="angles in (fov[0], fov[1]) to be removed", type=float, nargs=2, default=[0, 0])

    ## VISUALIZE INFORMATION
    parser.add_argument("-v", "--verbose", help="increase output verbosity", type=int, default=0, nargs='?', const=1)
    parser.add_argument("-t", "--timer", help="timer period (sec) default 0.1", type=float, default=0.1)
    parser.add_argument("-f", "--figure", help="show figures", action="store_true")
    parser.add_argument("-p", "--print-map", help="print map", action="store_true")


    ## GPU
    parser.add_argument("-ug", "--use-gpu", action="store_false")
    parser.add_argument("-sg", "--set-gpu", help="set cuda visible devices, default none", type=int, default=[],nargs='+')


    ## MOTION(PROCESS) MODEL
    parser.add_argument('--trans-belief', help='select how to fill after transition', choices=['shift','roll','stoch-shift'], default='stoch-shift', type=str)
    parser.add_argument("--fwd-step", "-fs", type=int, default=1)
    parser.add_argument("--rot-step", "-rs", type=int, default=1)

    ## RL-GENERAL
    parser.add_argument('--update-rl', dest='update_rl', action='store_true')
    parser.add_argument('--no-update-rl', dest='update_rl',help="don't update AC model", action="store_false")
    parser.add_argument('--update-ir', dest='update_ir', action='store_true')
    parser.add_argument('--no-update-ir', dest='update_ir',help="don't update IR model", action="store_false")
    parser.set_defaults(update_rl=False, update_ir=False)

    ## RL-ACTION
    parser.add_argument("--manual-control","-mc", action="store_true")
    parser.add_argument('--num-actions', type=int, default=3)
    parser.add_argument('--test-ep', help='number of test episode at the end of each env', type=int, default=0)
    parser.add_argument('-a','--action', help='select action : argmax or multinomial', choices=['argmax','multinomial'], default='multinomial', type=str)

    parser.add_argument('--binary-scan', action='store_true')

    ## RL-PARAMS
    parser.add_argument('-lam', '--lamda', help="weight for intrinsic reward, default=0.7", type=float, default=0.7)
    parser.add_argument('-vlcoeff', '--value_loss_coeff', help="value loss coefficient, default=1.0", type=float, default=1.0)
    parser.add_argument('-cent', '--c-entropy', help="coefficient of entropy in policy loss (0.001)", type=float, default=0.001)


    ## REWARD
    parser.add_argument('--block-penalty', dest='penalty_for_block', help="penalize for blocked fwd", action="store_true")
    parser.add_argument('--rew-explore', help="reward for exploration", action="store_true")
    parser.add_argument('--rew-bel-new', help='reward for new belief pose', action="store_true")
    parser.add_argument('--rew-bel-ent', help="reward for low entropy in belief", action="store_true")
    parser.add_argument('--rew-infogain', help="reward for info gain", action="store_true")
    parser.add_argument('--rew-bel-gt-nonlog', help="reward for correct belief", action="store_true")
    parser.add_argument('--rew-bel-gt', help="reward for correct belief", action="store_true")
    parser.add_argument('--rew-dist', help="reward for distance", action="store_true")
    parser.add_argument('--rew-hit', help="reward for distance being 0", action="store_true")
    parser.add_argument('--rew-inv-dist', help="r=1/(1+d)", action="store_true")
    #parser.set_defaults(reward_for_belief=False, reward_for_dist=False, reward_inv_dist=False, penalty_for_block=False, reward_for_explore=False)

    ## TRUE LIKELIHOOD
    parser.add_argument("--gtl-src", help="source of GTL", choices=['ld', 'hd-cos','hd-corr'], default='hd-cos')
    parser.add_argument("--gtl-output", choices=['softmax','softermax','linear'], default='softmax')


    ## LM-GENERAL
    parser.add_argument("-temp", "--temperature", help="softmax temperature", type=float, default=1.0)
    parser.add_argument('--rot-equiv', action='store_false')
    parser.add_argument('--pm-net', help ="select PM network", choices = ['resnet18', 'resnet50', 'resnet101','conv4','resnet152'], default='resnet152')
    parser.add_argument('--pm-loss', choices=['L1','KL'], default='KL')
    parser.add_argument('--pm-scan-step', type=int, default=5)
    parser.add_argument('--shade', dest="shade", help="shade for scan image", action="store_true")
    parser.add_argument('--no-shade', dest="shade", help="no shade for scan image", action="store_false")
    parser.set_defaults(shade=False)

    parser.add_argument('--pm-batch-size', help='batch size of pm model.', default=1, type=int)

    parser.add_argument("-ugl", "--use-gt-likelihood", help="PM = ground truth likelihood", action="store_true")
    parser.add_argument("--mask", action="store_true", help='mask likelihood with obstacle info')
    parser.add_argument("--n_fc1_out", help="size of FC1 out in PM (32)", type=int, default=32)
    parser.add_argument("--kernel_size", help="size of conv kernel in PM (5)", type=int, default=5)

    ## LM-PARAMS
    parser.add_argument('-upm', '--update-pm-by', help="train PM with GTL,RL,both, none", choices = ['GTL','RL','BOTH','NONE'], default='NONE', type=str)

    ## LOGGING
    parser.add_argument('-ln', "--tflogs-name", help="experiment name to append to the tensor board log files", type=str, default=None)
    parser.add_argument('-tf', '--tflog', dest="tflog",help="tensor board log True/False", action="store_true")
    parser.add_argument('-ntf', '--no-tflog', dest="tflog",help="tensor board log True/False", action="store_false")
    parser.set_defaults(tflog=False)

    parser.add_argument('--save', help="save logs and models", action="store_true", dest='save')
    parser.add_argument('--no-save', help="don't save any logs or models", action="store_false", dest='save')
    parser.set_defaults(save=True)

    parser.add_argument('--prob-roll-out', help="sample probability for roll out (0.01)", type=float, default=0.00)

    ## LOADING MODELS/DATA
    parser.add_argument('--pm-model', help="perceptual model path and file", type=str, default='./gym_dal/resnet152.model')
    parser.add_argument('--rl-model', help="RL model path and file", type=str, default=None)
    parser.add_argument('--ir-model', help="intrinsic reward model path and file", type=str, default=None)
    parser.add_argument('--test-mode', action="store_true")
    parser.add_argument('--test-data-path', type=str, default='')

    parser.add_argument("--n-lm-grids", type=int, default=11)
    parser.add_argument("--n-pre-classes", "-npc", type=int, default=None)
    parser.add_argument("--RL-type", type=int, default=0, choices=[0,1,2]) 
    # 0: original[map+scan+bel], 1: no map[scan+bel], 2:extended[bel+lik+hd-scan+hd-map]
    parser.add_argument("--n-maze-grids", type=int, nargs='+', default=[5,11])
    parser.add_argument("--n-local-grids", type=int, default=11)
    parser.add_argument("--n-state-grids", type=int, default=11)
    parser.add_argument("--n-state-dirs", type=int, default=4)
    parser.add_argument("--lidar-noise", help="number of random noisy rays in a scan", type=int, default=0)
    parser.add_argument("--lidar-sigma", help="sigma for lidar (1d) range", type=float, default=0)
    parser.add_argument("--scan-range", help="[min, max] scan range (m)", type=float, nargs=2, default=[0.10, 3.5])
        ## COLLISION
    parser.add_argument("--collision-radius", "-cr", type=float, default=0.25)
    parser.add_argument("--collision-from", type=str, choices=['none','map','scan'], default='map')

    
    args_iko = parser.parse_args()

    args_iko.cuda = not args_iko.no_cuda and torch.cuda.is_available()

    return args_iko

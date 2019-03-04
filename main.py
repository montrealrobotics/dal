import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args_iko
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
# from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot
from gym_dal.envs import dal_env

from networks import RewardModel
from a2c_ppo_acktr import arguments

args_iko = arguments.get_args_iko()

assert args_iko.algo in ['a2c', 'ppo', 'acktr']
if args_iko.recurrent_policy:
    assert args_iko.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args_iko.num_env_steps) // args_iko.num_steps // args_iko.num_processes

torch.manual_seed(args_iko.seed)
torch.cuda.manual_seed_all(args_iko.seed)

if args_iko.cuda and torch.cuda.is_available() and args_iko.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

try:
    os.makedirs(args_iko.log_dir)
except OSError:
    files = glob.glob(os.path.join(args_iko.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args_iko.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

# print("ppo epoch = ",args_iko.ppo_epoch)
# print("block penalty = ",args_iko.penalty_for_block)

from datetime import datetime
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
writer = SummaryWriter(log_dir='runs/' 
            # + 'ugl' + str(args_iko.use_gt_likelihood)
            # + 'block-pen-' + str(args_iko.penalty_for_block) + '_'
            # + 'explore-' + str(args_iko.rew_explore) + '_'
            # + 'bel-new-' + str(args_iko.rew_bel_new) + '_'
            # + 'bel-ent-' + str(args_iko.rew_bel_ent) + '_'
            # + 'infogain-' + str(args_iko.rew_infogain) + '_'
            # + 'bel-gt-nolog-' + str(args_iko.rew_bel_gt_nonlog) + '_'
            + 'bel-gt-' + str(args_iko.rew_bel_gt) + '_'
            # + 'dist-' + str(args_iko.rew_dist) + '_'
            # + 'hit-' + str(args_iko.rew_hit) + '_'
            # + 'inv-dist-' + str(args_iko.rew_inv_dist) + '_'
            + 'algo_' + str(args_iko.algo) + '_'
            # + 'lr-rl' + str(args_iko.lr) + '_'
            # + 'eps' + str(args_iko.eps) + '_'
            # + 'alpha' + str(args_iko.alpha) + '_'
            # + 'tau' + str(args_iko.tau) + '_'
            # + 'ent-c' + str(args_iko.entropy_coef) + '_'
            + 'vl-c' + str(args_iko.value_loss_coef) + '_'
            + 'bs' + str(args_iko.num_mini_batch) + '_' 
            + 'num-steps' + str(args_iko.num_steps) + '_'
            + 'lr' + str(args_iko.lr)
            # + 'singh-c' + str(args_iko.singh_coef) + '_'
            # + 'use-singh' + str(args_iko.use_singh) + '_'
            + str(current_time))

def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args_iko.cuda else "cpu")

    if args_iko.vis:
        from visdom import Visdom
        viz = Visdom(port=args_iko.port)
        win = None

    envs = make_vec_envs(args_iko.env_name, args_iko.seed, args_iko.num_processes,
                        args_iko.gamma, args_iko.log_dir, args_iko.add_timestep, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args_iko.recurrent_policy})
    actor_critic.to(device)

    action_shape = 3
    reward_model = RewardModel(11 * 11 * 6, 1, 64, 64)
    reward_model.to(device)

    if args_iko.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args_iko.value_loss_coef,
                               args_iko.entropy_coef, lr=args_iko.lr,
                               eps=args_iko.eps, alpha=args_iko.alpha,
                               max_grad_norm=args_iko.max_grad_norm)
    elif args_iko.algo == 'ppo':
        agent = algo.PPO(actor_critic, args_iko.clip_param, args_iko.ppo_epoch, args_iko.num_mini_batch,
                         args_iko.value_loss_coef, args_iko.entropy_coef, args_iko.use_singh, reward_model, lr=args_iko.lr,
                               eps=args_iko.eps,
                               max_grad_norm=args_iko.max_grad_norm)
    elif args_iko.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args_iko.value_loss_coef,
                               args_iko.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args_iko.num_steps, args_iko.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    for j in range(num_updates):

        if args_iko.use_linear_lr_decay:
            # decrease learning rate linearly
            if args_iko.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args_iko.lr)

        if args_iko.algo == 'ppo' and args_iko.use_linear_clip_decay:
            agent.clip_param = args_iko.clip_param  * (1 - j / float(num_updates))

        reward_train = []
        reward_block_penalty = []
        reward_bel_gt = []
        reward_bel_gt_nonlog = []
        reward_infogain = []
        reward_bel_ent = []
        reward_hit = []
        reward_dist = []
        reward_inv_dist = []

        for step in range(args_iko.num_steps):
            # Sample actions
            # print(step, args_iko.num_steps)
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            reward_train.append(reward)
            # print("infos is ", infos)
            # reward_b.append(infos[0]['auxillary_reward'])
            # print("infos is ",infos[0]['auxillary_reward'])
            reward_block_penalty.append(infos[0]['reward_block_penalty'])
            reward_bel_gt.append(infos[0]['reward_bel_gt'])
            reward_bel_gt_nonlog.append(infos[0]['reward_bel_gt_nonlog'])
            reward_infogain.append(infos[0]['reward_infogain'])
            reward_bel_ent.append(infos[0]['reward_bel_ent'])
            reward_hit.append(infos[0]['reward_hit'])
            reward_dist.append(infos[0]['reward_dist'])
            reward_inv_dist.append(infos[0]['reward_inv_dist'])
            # print(reward)

            reward.to(device)
            reward_model.to(device)
            if args_iko.use_singh:
                # print("using learning IR")
                my_reward = reward_model(obs.clone().to(device), action.clone().float()).detach()
                my_reward.to(device)
                reward = reward + args_iko.singh_coef * my_reward.type(torch.FloatTensor)

            # for info in infos:
            #     if 'episode' in info.keys():
            #         episode_rewards.append(info['episode']['r'])
            #         print("infos is ",infos[0]['auxillary_reward'])
            #         print("info is",info['episode']['r'] )

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)


        # print("mean reward_a", np.mean(reward_train))
        # print("mean reward_block_penalty", np.mean(reward_block_penalty))
        # print("mean reward_bel_gt", np.mean(reward_bel_gt))
        # print("mean reward_bel_gt_nonlog", np.mean(reward_bel_gt_nonlog))
        # print("mean reward_infogain", np.mean(reward_infogain))
        # print("mean reward_bel_ent", np.mean(reward_bel_ent))
        # print("mean reward_hit", np.mean(reward_hit))
        # print("mean reward_dist", np.mean(reward_dist))
        # print("mean reward_inv_dist", np.mean(reward_inv_dist))

        total_num_steps = (j + 1) * args_iko.num_processes * args_iko.num_steps
        writer.add_scalar('mean_reward_train', np.mean(reward_train), total_num_steps)
        writer.add_scalar('mean_reward_block_penalty', np.mean(reward_block_penalty), total_num_steps)
        writer.add_scalar('mean_reward_bel_gt', np.mean(reward_bel_gt), total_num_steps)
        writer.add_scalar('mean_reward_bel_gt_nonlog', np.mean(reward_bel_gt_nonlog), total_num_steps)
        writer.add_scalar('mean_reward_infogain', np.mean(reward_infogain), total_num_steps)
        writer.add_scalar('mean_reward_bel_ent', np.mean(reward_bel_ent), total_num_steps)
        writer.add_scalar('mean_reward_hit', np.mean(reward_hit), total_num_steps)
        writer.add_scalar('mean_reward_dist', np.mean(reward_dist), total_num_steps)
        writer.add_scalar('mean_reward_inv_dist', np.mean(reward_inv_dist), total_num_steps)
        
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args_iko.use_gae, args_iko.gamma, args_iko.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args_iko.save_interval == 0 or j == num_updates - 1) and args_iko.save_dir != "":
            save_path = os.path.join(args_iko.save_dir, args_iko.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args_iko.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, 'ugl' + str(args_iko.use_gt_likelihood)
            + 'block-pen-' + str(args_iko.penalty_for_block) + '_'
            + 'explore-' + str(args_iko.rew_explore) + '_'
            + 'bel-new-' + str(args_iko.rew_bel_new) + '_'
            + 'bel-ent-' + str(args_iko.rew_bel_ent) + '_'
            + 'infogain-' + str(args_iko.rew_infogain) + '_'
            + 'bel-gt-nolog-' + str(args_iko.rew_bel_gt_nonlog) + '_'
            + 'bel-gt-' + str(args_iko.rew_bel_gt) + '_'
            + 'dist-' + str(args_iko.rew_dist) + '_'
            + 'hit-' + str(args_iko.rew_hit) + '_'
            + 'inv-dist-' + str(args_iko.rew_inv_dist) + args_iko.algo + ".pt"))

        total_num_steps = (j + 1) * args_iko.num_processes * args_iko.num_steps

        if j % args_iko.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("mean reward_a", np.mean(reward_a))
            print("mean_reward_b", np.mean(reward_b))
            # print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
            #     format(j, total_num_steps,
            #            int(total_num_steps / (end - start)),
            #            len(episode_rewards),
            #            np.mean(episode_rewards),
            #            np.median(episode_rewards),
            #            np.min(episode_rewards),
            #            np.max(episode_rewards), dist_entropy,
            #            value_loss, action_loss))
            # writer.add_scalar('mean_reward', np.mean(episode_rewards), total_num_steps)
            # writer.add_scalar('min_reward', np.min(episode_rewards), total_num_steps)
            # writer.add_scalar('max_reward', np.max(episode_rewards), total_num_steps)
            # writer.add_scalar('success_rate', np.mean(episode_successes), total_num_steps)

        if (args_iko.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args_iko.eval_interval == 0):
            eval_envs = make_vec_envs(
                args_iko.env_name, args_iko.seed + args_iko.num_processes, args_iko.num_processes,
                args_iko.gamma, eval_log_dir, args_iko.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args_iko.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args_iko.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))

        if args_iko.vis and j % args_iko.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args_iko.log_dir, args_iko.env_name,
                                  args_iko.algo, args_iko.num_env_steps)
            except IOError:
                pass
    writer.close()


if __name__ == "__main__":
    main()

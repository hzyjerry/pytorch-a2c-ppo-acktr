import os, sys, time, copy, glob
from collections import deque

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo.a2c_ppo_acktr import algo
from ppo.a2c_ppo_acktr.arguments import get_args
from ppo.a2c_ppo_acktr.envs import make_vec_envs
from ppo.a2c_ppo_acktr.model import Policy
from ppo.a2c_ppo_acktr.storage import RolloutStorage
from ppo.a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from ppo.a2c_ppo_acktr.visualize import visdom_plot


args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

if args.num_rollouts == 0:
    # Find a number of rollouts such that num_rollouts % num_processes == 0 and num_rollouts >= 30
    while args.num_rollouts < 30:
        args.num_rollouts += args.num_processes
if args.num_rollouts > 0:
    assert args.num_rollouts % args.num_processes == 0, 'num_rollouts must be divisable by num_processes'

num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    try:
        for f in files:
            os.remove(f)
    except PermissionError as e:
        pass

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    try:
        for f in files:
            os.remove(f)
    except PermissionError as e:
        pass

def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = make_vec_envs(args.env_name, args.seed, 1,
                        args.gamma, args.log_dir, args.add_timestep, device, False)

    # Determine if this is a dual robot (multi agent) environment.
    obs = envs.reset()
    action = torch.tensor([envs.action_space.sample()])
    _, _, _, info = envs.step(action)
    dual_robots = 'dual_robots' in info[0]
    if dual_robots:
        obs_robot_len = info[0]['obs_robot_len'] // 2
        action_robot_len = info[0]['action_robot_len'] // 2
        obs_robot1 = obs[:, :obs_robot_len]
        obs_robot2 = obs[:, obs_robot_len:]
        if len(obs_robot1[0]) != obs_robot_len or len(obs_robot2[0]) != obs_robot_len:
            print('robot 1 obs shape:', len(obs_robot1[0]), 'obs space robot shape:', (obs_robot_len,))
            print('robot 2 obs shape:', len(obs_robot2[0]), 'obs space robot shape:', (obs_robot_len,))
            exit()

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)

    if dual_robots:
        # Reset environment
        obs = envs.reset()
        obs_robot1 = obs[:, :obs_robot_len]
        obs_robot2 = obs[:, obs_robot_len:]
        action_space_robot1 = spaces.Box(low=np.array([-1.0]*action_robot_len), high=np.array([1.0]*action_robot_len), dtype=np.float32)
        action_space_robot2 = spaces.Box(low=np.array([-1.0]*action_robot_len), high=np.array([1.0]*action_robot_len), dtype=np.float32)

    if args.load_policy is not None:
        if dual_robots:
            actor_critic_robot1, actor_critic_robot2, ob_rms = torch.load(args.load_policy)
        else:
            actor_critic, ob_rms = torch.load(args.load_policy)
        vec_norm = get_vec_normalize(envs)
        if vec_norm is not None:
            vec_norm.ob_rms = ob_rms
    else:
        if dual_robots:
            actor_critic_robot1 = Policy([obs_robot_len], action_space_robot1,
                base_kwargs={'recurrent': args.recurrent_policy})
            actor_critic_robot2 = Policy([obs_robot_len], action_space_robot2,
                base_kwargs={'recurrent': args.recurrent_policy})
        else:
            actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy})
    if dual_robots:
        actor_critic_robot1.to(device)
        actor_critic_robot2.to(device)
    else:
        actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        if dual_robots:
            agent_robot1 = algo.PPO(actor_critic_robot1, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                             args.value_loss_coef, args.entropy_coef, lr=args.lr,
                                   eps=args.eps,
                                   max_grad_norm=args.max_grad_norm)
            agent_robot2 = algo.PPO(actor_critic_robot2, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                             args.value_loss_coef, args.entropy_coef, lr=args.lr,
                                   eps=args.eps,
                                   max_grad_norm=args.max_grad_norm)
        else:
            agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                             args.value_loss_coef, args.entropy_coef, lr=args.lr,
                                   eps=args.eps,
                                   max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    if dual_robots:
        rollouts_robot1 = RolloutStorage(args.num_steps, args.num_rollouts if args.num_rollouts > 0 else args.num_processes,
                            [obs_robot_len], action_space_robot1,
                            actor_critic_robot1.recurrent_hidden_state_size)
        rollouts_robot2 = RolloutStorage(args.num_steps, args.num_rollouts if args.num_rollouts > 0 else args.num_processes,
                            [obs_robot_len], action_space_robot2,
                            actor_critic_robot2.recurrent_hidden_state_size)
        if args.num_rollouts > 0:
            rollouts_robot1.obs[0].copy_(torch.cat([obs_robot1 for _ in range(args.num_rollouts // args.num_processes)] + [obs_robot1[:(args.num_rollouts % args.num_processes)]], dim=0))
            rollouts_robot2.obs[0].copy_(torch.cat([obs_robot2 for _ in range(args.num_rollouts // args.num_processes)] + [obs_robot2[:(args.num_rollouts % args.num_processes)]], dim=0))
        else:
            rollouts_robot1.obs[0].copy_(obs_robot1)
            rollouts_robot2.obs[0].copy_(obs_robot2)
        rollouts_robot1.to(device)
        rollouts_robot2.to(device)
    else:
        rollouts = RolloutStorage(args.num_steps, args.num_rollouts if args.num_rollouts > 0 else args.num_processes,
                            envs.observation_space.shape, envs.action_space,
                            actor_critic.recurrent_hidden_state_size)
        obs = envs.reset()
        if args.num_rollouts > 0:
            rollouts.obs[0].copy_(torch.cat([obs for _ in range(args.num_rollouts // args.num_processes)] + [obs[:(args.num_rollouts % args.num_processes)]], dim=0))
        else:
            rollouts.obs[0].copy_(obs)
        rollouts.to(device)

    deque_len = args.num_rollouts if args.num_rollouts > 0 else (args.num_processes if args.num_processes > 10 else 10)
    if dual_robots:
        episode_rewards_robot1 = deque(maxlen=deque_len)
        episode_rewards_robot2 = deque(maxlen=deque_len)
    else:
        episode_rewards = deque(maxlen=deque_len)

    start = time.time()
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                if dual_robots:
                    update_linear_schedule(agent_robot1.optimizer, j, num_updates, args.lr)
                    update_linear_schedule(agent_robot2.optimizer, j, num_updates, args.lr)
                else:
                    update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            if dual_robots:
                agent_robot1.clip_param = args.clip_param  * (1 - j / float(num_updates))
                agent_robot2.clip_param = args.clip_param  * (1 - j / float(num_updates))
            else:
                agent.clip_param = args.clip_param  * (1 - j / float(num_updates))

        reward_list_robot1 = [[] for _ in range(args.num_processes)]
        reward_list_robot2 = [[] for _ in range(args.num_processes)]
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                if dual_robots:
                    value_robot1, action_robot1, action_log_prob_robot1, recurrent_hidden_states_robot1 = actor_critic_robot1.act(
                            rollouts_robot1.obs[step, :args.num_processes],
                            rollouts_robot1.recurrent_hidden_states[step, :args.num_processes],
                            rollouts_robot1.masks[step, :args.num_processes])
                    value_robot2, action_robot2, action_log_prob_robot2, recurrent_hidden_states_robot2 = actor_critic_robot2.act(
                            rollouts_robot2.obs[step, :args.num_processes],
                            rollouts_robot2.recurrent_hidden_states[step, :args.num_processes],
                            rollouts_robot2.masks[step, :args.num_processes])
                else:
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts.obs[step, :args.num_processes],
                            rollouts.recurrent_hidden_states[step, :args.num_processes],
                            rollouts.masks[step, :args.num_processes])

            # Obser reward and next obs
            if dual_robots:
                action = torch.cat((action_robot1, action_robot2), dim=-1)
                obs, reward, done, infos = envs.step(action)
                obs_robot1 = obs[:, :obs_robot_len]
                obs_robot2 = obs[:, obs_robot_len:]
                for i, info in enumerate(infos):
                    reward_list_robot1[i].append(info['reward_robot1'])
                    reward_list_robot2[i].append(info['reward_robot2'])
                reward_robot1 = torch.tensor([[info['reward_robot1']] for info in infos])
                reward_robot2 = torch.tensor([[info['reward_robot2']] for info in infos])
            else:
                obs, reward, done, infos = envs.step(action)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    if dual_robots:
                        episode_rewards_robot1.append(np.sum(reward_list_robot1[i]))
                        episode_rewards_robot2.append(np.sum(reward_list_robot2[i]))
                    else:
                        episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            if dual_robots:
                rollouts_robot1.insert(obs_robot1, recurrent_hidden_states_robot1, action_robot1, action_log_prob_robot1, value_robot1, reward_robot1, masks)
                rollouts_robot2.insert(obs_robot2, recurrent_hidden_states_robot2, action_robot2, action_log_prob_robot2, value_robot2, reward_robot2, masks)
            else:
                rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        if args.num_rollouts > 0 and (j % (args.num_rollouts // args.num_processes) != 0):
            # Only update the policies when we have performed num_rollouts simulations
            continue

        with torch.no_grad():
            if dual_robots:
                next_value_robot1 = actor_critic_robot1.get_value(rollouts_robot1.obs[-1],
                                                    rollouts_robot1.recurrent_hidden_states[-1],
                                                    rollouts_robot1.masks[-1]).detach()
                next_value_robot2 = actor_critic_robot2.get_value(rollouts_robot2.obs[-1],
                                                    rollouts_robot2.recurrent_hidden_states[-1],
                                                    rollouts_robot2.masks[-1]).detach()
            else:
                next_value = actor_critic.get_value(rollouts.obs[-1],
                                                    rollouts.recurrent_hidden_states[-1],
                                                    rollouts.masks[-1]).detach()

        if dual_robots:
            rollouts_robot1.compute_returns(next_value_robot1, args.use_gae, args.gamma, args.tau)
            rollouts_robot2.compute_returns(next_value_robot2, args.use_gae, args.gamma, args.tau)
            value_loss_robot1, action_loss_robot1, dist_entropy_robot1 = agent_robot1.update(rollouts_robot1)
            value_loss_robot2, action_loss_robot2, dist_entropy_robot2 = agent_robot2.update(rollouts_robot2)
            rollouts_robot1.after_update()
            rollouts_robot2.after_update()
        else:
            rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
            value_loss, action_loss, dist_entropy = agent.update(rollouts)
            rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            if dual_robots:
                save_model_robot1 = actor_critic_robot1
                save_model_robot2 = actor_critic_robot2
                if args.cuda:
                    save_model_robot1 = copy.deepcopy(actor_critic_robot1).cpu()
                    save_model_robot2 = copy.deepcopy(actor_critic_robot2).cpu()
                save_model = [save_model_robot1, save_model_robot2,
                              getattr(get_vec_normalize(envs), 'ob_rms', None)]
            else:
                save_model = actor_critic
                if args.cuda:
                    save_model = copy.deepcopy(actor_critic).cpu()
                save_model = [save_model,
                              getattr(get_vec_normalize(envs), 'ob_rms', None)]
            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and (len(episode_rewards_robot1) > 1 if dual_robots else len(episode_rewards) > 1):
            end = time.time()
            if dual_robots:
                print("Robot1 updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".
                    format(j, total_num_steps,
                           int(total_num_steps / (end - start)),
                           len(episode_rewards_robot1),
                           np.mean(episode_rewards_robot1),
                           np.median(episode_rewards_robot1),
                           np.min(episode_rewards_robot1),
                           np.max(episode_rewards_robot1), dist_entropy_robot1,
                           value_loss_robot1, action_loss_robot1))
                print("Robot2 updates {}, Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                    format(j, len(episode_rewards_robot2),
                           np.mean(episode_rewards_robot2),
                           np.median(episode_rewards_robot2),
                           np.min(episode_rewards_robot2),
                           np.max(episode_rewards_robot2), dist_entropy_robot2,
                           value_loss_robot2, action_loss_robot2))
            else:
                print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                    format(j, total_num_steps,
                           int(total_num_steps / (end - start)),
                           len(episode_rewards),
                           np.mean(episode_rewards),
                           np.median(episode_rewards),
                           np.min(episode_rewards),
                           np.max(episode_rewards), dist_entropy,
                           value_loss, action_loss))
            sys.stdout.flush()

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, args.num_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            if dual_robots:
                eval_episode_rewards_robot1 = []
                eval_episode_rewards_robot2 = []
            else:
                eval_episode_rewards = []

            obs = eval_envs.reset()
            if dual_robots:
                obs_robot1 = obs[:, :obs_robot_len]
                obs_robot2 = obs[:, obs_robot_len:]
                eval_recurrent_hidden_states_robot1 = torch.zeros(args.num_processes,
                                actor_critic_robot1.recurrent_hidden_state_size, device=device)
                eval_recurrent_hidden_states_robot2 = torch.zeros(args.num_processes,
                                actor_critic_robot2.recurrent_hidden_state_size, device=device)
            else:
                eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            eval_reward_list_robot1 = [[] for _ in range(args.num_processes)]
            eval_reward_list_robot2 = [[] for _ in range(args.num_processes)]
            while (len(eval_episode_rewards_robot1) < 10 if dual_robots else len(eval_episode_rewards) < 10):
                with torch.no_grad():
                    if dual_robots:
                        _, action_robot1, _, eval_recurrent_hidden_states_robot1 = actor_critic_robot1.act(
                            obs_robot1, eval_recurrent_hidden_states_robot1, eval_masks, deterministic=True)
                        _, action_robot2, _, eval_recurrent_hidden_states_robot2 = actor_critic_robot2.act(
                            obs_robot2, eval_recurrent_hidden_states_robot2, eval_masks, deterministic=True)
                    else:
                        _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                            obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                if dual_robots:
                    action = torch.cat((action_robot1, action_robot2), dim=-1)
                    obs, reward, done, infos = eval_envs.step(action)
                    obs_robot1 = obs[:, :obs_robot_len]
                    obs_robot2 = obs[:, obs_robot_len:]
                    for i, info in enumerate(infos):
                        eval_reward_list_robot1[i].append(info['reward_robot1'])
                        eval_reward_list_robot2[i].append(info['reward_robot2'])
                else:
                    obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                reset_rewards = False
                for info in infos:
                    if 'episode' in info.keys():
                        if dual_robots:
                            reset_rewards = True
                            eval_episode_rewards_robot1.append(np.sum(eval_reward_list_robot1[i]))
                            eval_episode_rewards_robot2.append(np.sum(eval_reward_list_robot2[i]))
                        else:
                            eval_episode_rewards.append(info['episode']['r'])
                if reset_rewards:
                    eval_reward_list_robot1 = [[] for _ in range(args.num_processes)]
                    eval_reward_list_robot2 = [[] for _ in range(args.num_processes)]

            eval_envs.close()

            if dual_robots:
                print(" Evaluation using {} episodes: robot1 mean reward {:.5f}, robot2 mean reward {:.5f}\n".
                     format(len(eval_episode_rewards_robot1),
                            np.mean(eval_episode_rewards_robot1), np.mean(eval_episode_rewards_robot2)))
            else:
                print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                    format(len(eval_episode_rewards),
                           np.mean(eval_episode_rewards)))
            sys.stdout.flush()

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_env_steps)
            except IOError:
                pass


if __name__ == "__main__":
    import argparse
    main()

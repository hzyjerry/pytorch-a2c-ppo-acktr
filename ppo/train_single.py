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
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)


    if args.load_policy is not None:
        actor_critic, ob_rms = torch.load(args.load_policy)
        vec_norm = get_vec_normalize(envs)
        if vec_norm is not None:
            vec_norm.ob_rms = ob_rms
    else:
        actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                             args.value_loss_coef, args.entropy_coef, lr=args.lr,
                                   eps=args.eps,
                                   max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)


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
    episode_rewards = deque(maxlen=deque_len)

    start = time.time()
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(num_updates))

        reward_list_robot1 = [[] for _ in range(args.num_processes)]
        reward_list_robot2 = [[] for _ in range(args.num_processes)]
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step, :args.num_processes],
                        rollouts.recurrent_hidden_states[step, :args.num_processes],
                        rollouts.masks[step, :args.num_processes])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        if args.num_rollouts > 0 and (j % (args.num_rollouts // args.num_processes) != 0):
            # Only update the policies when we have performed num_rollouts simulations
            continue

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()

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
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()
            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]
            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and (len(episode_rewards) > 1):
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(j, total_num_steps,
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

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            eval_reward_list_robot1 = [[] for _ in range(args.num_processes)]
            eval_reward_list_robot2 = [[] for _ in range(args.num_processes)]
            while (len(eval_episode_rewards) < 10):
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                reset_rewards = False
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])
                if reset_rewards:
                    eval_reward_list_robot1 = [[] for _ in range(args.num_processes)]
                    eval_reward_list_robot2 = [[] for _ in range(args.num_processes)]

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(len(eval_episode_rewards),
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

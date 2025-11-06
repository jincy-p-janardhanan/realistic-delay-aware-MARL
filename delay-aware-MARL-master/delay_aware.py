import argparse
import torch
import time
import os
import numpy as np
from gymnasium.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.noise import DelayOUNoise
from algorithms.maddpg import MADDPG
from tqdm import tqdm 

# FIXED: Proper CUDA detection and initialization
def setup_cuda():
    """Setup CUDA with proper device selection for Colab"""
    if torch.cuda.is_available():
        # In Colab, GPU 0 is the default
        device_id = 0
        torch.cuda.set_device(device_id)
        print(f"CUDA available: {torch.cuda.get_device_name(device_id)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        return True
    else:
        print("✗ CUDA not available, using CPU")
        return False

USE_CUDA = setup_cuda()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            np.random.seed(seed + rank * 1000)
            env.reset(seed=seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def compute_virtual_action(action_buffer, delay_float):
    """
    Compute virtual effective action for non-integral delays.
    
    Args:
        action_buffer: List of past actions [newest, ..., oldest]
        delay_float: Non-integral delay (e.g., 2.3)
    
    Returns:
        Virtual effective action: (1-f)*a[I+1] + f*a[I]
    """
    if delay_float == 0:
        return action_buffer[0]
        
    I = int(np.floor(delay_float))  # Integer part
    f = delay_float - I  # Fractional part
    
    if I >= len(action_buffer):
        # If delay exceeds buffer, use oldest action
        return action_buffer[-1]
    
    if f == 0:
        # Pure integral delay
        return action_buffer[I]
    else:
        # Non-integral delay: interpolate between two actions
        if I + 1 >= len(action_buffer):
            # If we can't interpolate, use the oldest action
            return action_buffer[-1]
        return (1 - f) * action_buffer[I] + f * action_buffer[I + 1]


def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Environment: {config.env_id}")
    print(f"Device: {'CUDA (GPU)' if USE_CUDA else 'CPU'}")
    print(f"Min delay: {config.min_delay}")
    print(f"Max delay: {config.max_delay}")
    print(f"OU theta: {config.ou_theta}")
    print(f"OU sigma: {config.ou_sigma}")
    print(f"Episodes: {config.n_episodes}")
    print(f"Episode length: {config.episode_length}")
    print(f"Exploration episodes: {config.n_exploration_eps}")
    print(f"Learning rate: {config.lr}")
    print(f"Gamma: {config.gamma}")
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Batch size: {config.batch_size}")
    print(f"Rollout threads: {config.n_rollout_threads}")
    print("="*60 + "\n")
    
    # Calculate buffer size needed based on max_delay
    delay_buffer_size = int(np.ceil(config.max_delay)) + 1
    print(f"[INFO] Delay buffer size: {delay_buffer_size}")
    
    maddpg = MADDPG.init_from_env_with_delay(
        env, 
        agent_alg=config.agent_alg,
        adversary_alg=config.adversary_alg,
        tau=config.tau,
        lr=config.lr,
        hidden_dim=config.hidden_dim,
        gamma=config.gamma,
        min_delay=config.min_delay,
        max_delay=config.max_delay,
        use_sigmoid=True
    )
    
    # Initialize OU noise processes for each environment and agent
    # Mean is set to midpoint of delay range
    delay_mean = (config.min_delay + config.max_delay) / 2.0
    ou_processes = []
    for env_idx in range(config.n_rollout_threads):
        env_ou_list = []
        for agent_idx in range(maddpg.nagents):
            ou = OUNoise(
                mu=delay_mean,
                theta=config.ou_theta,
                sigma=config.ou_sigma,
                x0=delay_mean
            )
            env_ou_list.append(ou)
        ou_processes.append(env_ou_list)
    
    for i, agent in enumerate(maddpg.agents):
        print(f"[INFO] Agent {i} policy input dim: {agent.policy.fc1.in_features}")
    
    # Calculate observation dimension for replay buffer
    replay_buffer = ReplayBuffer(
        config.buffer_length, 
        maddpg.nagents,
        [env.observation_space[i].shape[0] + env.action_space[i].shape[0] * delay_buffer_size 
         for i in range(maddpg.nagents)],
        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
         for acsp in env.action_space]
    )
    
    t = 0
    pbar = tqdm(range(0, config.n_episodes, config.n_rollout_threads),
                desc="Training progress", dynamic_ncols=True)
    
    # Storage for delay statistics
    all_delays = []
    
    for ep_i in pbar:
        obs = env.reset()
        
        # FIXED: Proper device switching
        if USE_CUDA:
            maddpg.prep_rollouts(device='gpu')
        else:
            maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        # Initialize action buffers for each environment
        last_agent_actions = []
        for env_idx in range(config.n_rollout_threads):
            env_agent_buffers = []
            for agent_idx in range(maddpg.nagents):
                zero_actions = [np.zeros(env.action_space[agent_idx].shape[0]) 
                               for _ in range(delay_buffer_size)]
                env_agent_buffers.append(zero_actions)
            last_agent_actions.append(env_agent_buffers)
        
        # Reset OU processes at episode start
        for env_ou_list in ou_processes:
            for ou in env_ou_list:
                ou.reset()
        
        # Append action history to observations for ALL environments
        for env_idx in range(config.n_rollout_threads):
            for a_i in range(len(obs[env_idx])):
                agent_obs = obs[env_idx][a_i]
                for action_idx in range(delay_buffer_size):
                    agent_obs = np.append(agent_obs, last_agent_actions[env_idx][a_i][action_idx])
                obs[env_idx, a_i] = agent_obs
        
        episode_delays = []  # Track delays for this episode
        
        for et_i in range(config.episode_length):
            # Get observations for all agents
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            
            # FIXED: Move observations to GPU if using CUDA
            if USE_CUDA:
                torch_obs = [obs.cuda() for obs in torch_obs]
            
            # Get actions from policies
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            agent_actions = [ac.data.cpu().numpy().astype(np.float32) for ac in torch_agent_actions]

            # Clip actions
            for a_i in range(len(agent_actions)):
                low = env.action_space[a_i].low
                high = env.action_space[a_i].high
                agent_actions[a_i] = np.clip(agent_actions[a_i], low, high).astype(np.float32)
            
            actions = []
            step_delays = []
            
            for env_idx in range(config.n_rollout_threads):
                # Get current actions for this environment
                current_actions = [agent_actions[a_i][env_idx] for a_i in range(maddpg.nagents)]
                
                # Compute virtual effective actions using OU-generated delays
                env_actions = []
                env_step_delays = []
                for agent_idx in range(maddpg.nagents):
                    # Sample delay from OU process
                    raw_delay = ou_processes[env_idx][agent_idx].sample()
                    # Clip to [min_delay, max_delay]
                    current_delay = np.clip(raw_delay, config.min_delay, config.max_delay)
                    env_step_delays.append(current_delay)
                    
                    virtual_action = compute_virtual_action(
                        last_agent_actions[env_idx][agent_idx], 
                        current_delay
                    )
                    # Clip to valid action range
                    low, high = env.action_space[agent_idx].low, env.action_space[agent_idx].high
                    virtual_action = np.clip(virtual_action, low, high).astype(np.float32)
                    env_actions.append(virtual_action)
                
                step_delays.append(env_step_delays)
                
                # Update action buffers: shift and add new action
                for agent_idx in range(maddpg.nagents):
                    last_agent_actions[env_idx][agent_idx].pop()
                    last_agent_actions[env_idx][agent_idx].insert(0, current_actions[agent_idx])
                
                actions.append(env_actions)
            
            episode_delays.append(step_delays)
            
            # Step environment with virtual effective actions
            next_obs, rewards, dones, infos = env.step(actions)
            
            # Append action history to next observations
            for env_idx in range(config.n_rollout_threads):
                for a_i in range(len(next_obs[env_idx])):
                    agent_obs = next_obs[env_idx][a_i]
                    for action_idx in range(delay_buffer_size):
                        agent_obs = np.append(agent_obs, last_agent_actions[env_idx][a_i][action_idx])
                    next_obs[env_idx, a_i] = agent_obs
            
            # FIXED: Collect virtual actions and push ONCE (removed duplicate)
            virtual_actions_per_agent = []
            for agent_idx in range(maddpg.nagents):
                # Collect virtual actions across all parallel environments
                agent_virtual_actions = np.array([actions[env_idx][agent_idx] 
                                                for env_idx in range(config.n_rollout_threads)])
                virtual_actions_per_agent.append(agent_virtual_actions)

            # Store virtual actions (what actually executed) - ONLY ONCE!
            replay_buffer.push(obs, virtual_actions_per_agent, rewards, next_obs, dones)
            
            obs = next_obs
            t += config.n_rollout_threads
            
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                
                # Sample once, use for all agents
                sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA)
                
                # Update all agents (excluding runner if nagents > 3)
                for a_i in range(maddpg.nagents):
                    maddpg.update(sample, a_i, logger=logger)
                
                # Update all target networks
                maddpg.update_all_targets()
                
                if USE_CUDA:
                    maddpg.prep_rollouts(device='gpu')
                else:
                    maddpg.prep_rollouts(device='cpu')
        
        # Compute delay statistics for this episode
        episode_delays = np.array(episode_delays)  # shape: (episode_length, n_envs, n_agents)
        mean_delay = np.mean(episode_delays)
        std_delay = np.std(episode_delays)
        all_delays.extend(episode_delays.flatten())
        
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        
        # Log & update progress bar
        reward_str = " | ".join([f"A{i}: {a_ep_rew:.2f}" for i, a_ep_rew in enumerate(ep_rews)])
        delay_str = f"Delay: {mean_delay:.2f}±{std_delay:.2f}"
        pbar.set_description(f"Ep {ep_i + 1}/{config.n_episodes} [{reward_str}] {delay_str}")
        
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalars('agent%i/mean_episode_rewards' % a_i, {'reward': a_ep_rew}, ep_i)
        
        # Log delay statistics
        logger.add_scalar('delays/mean', mean_delay, ep_i)
        logger.add_scalar('delays/std', std_delay, ep_i)
        logger.add_scalar('delays/min', np.min(episode_delays), ep_i)
        logger.add_scalar('delays/max', np.max(episode_delays), ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    
    # Print final delay statistics
    print("\n" + "="*60)
    print("DELAY STATISTICS")
    print("="*60)
    print(f"Mean delay: {np.mean(all_delays):.3f}")
    print(f"Std delay: {np.std(all_delays):.3f}")
    print(f"Min delay: {np.min(all_delays):.3f}")
    print(f"Max delay: {np.max(all_delays):.3f}")
    print("="*60 + "\n")
    
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=40000, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=12000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')
    parser.add_argument("--min_delay", 
                        default=2.0, type=float,
                        help="Minimum delay in time steps")
    parser.add_argument("--max_delay", 
                        default=4.0, type=float,
                        help="Maximum delay in time steps")
    parser.add_argument("--ou_theta",
                        default=0.15, type=float,
                        help="OU process mean reversion rate (higher = faster return to mean)")
    parser.add_argument("--ou_sigma",
                        default=0.3, type=float,
                        help="OU process volatility/noise scale")

    config = parser.parse_args()

    run(config)
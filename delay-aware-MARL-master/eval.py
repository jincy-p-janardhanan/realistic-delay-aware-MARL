import os
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path

from algorithms.maddpg import MADDPG
from utils.make_env import make_env
from utils.noise import DelayOUNoise

def check_task_completed(env_adapter, threshold=0.15):
    """Check if all agents have reached their target landmarks."""
    try:
        # Try different paths to find the unwrapped environment
        if hasattr(env_adapter, 'env'):
            if hasattr(env_adapter.env, 'unwrapped'):
                raw_env = env_adapter.env.unwrapped
            elif hasattr(env_adapter.env, 'env'):
                raw_env = env_adapter.env.env
            else:
                raw_env = env_adapter.env
        else:
            raw_env = env_adapter
        
        if not hasattr(raw_env, 'world'):
            print(f"[DEBUG] No 'world' attribute. Type: {type(raw_env)}")
            return False
            
        agents = raw_env.world.agents
        landmarks = raw_env.world.landmarks
        
        if len(agents) != len(landmarks):
            print(f"[DEBUG] Agent/landmark mismatch: {len(agents)} vs {len(landmarks)}")
            return False
        
        distances = []
        for agent, landmark in zip(agents, landmarks):
            distance = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
            distances.append(distance)
        
        print(f"[DEBUG] Distances: {distances}")
        
        if all(d <= threshold for d in distances):
            return True
        return False
        
    except Exception as e:
        print(f"[DEBUG] Error in check_task_completed: {e}")
        import traceback
        traceback.print_exc()
        return False


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


def run_evaluation(args):
    """Main evaluation function"""
    
    # ==================== ENVIRONMENT SETUP ====================
    print(f"\n[INFO] Loading environment: {args.env_name}")
    env = make_env(args.env_name, render_mode='rgb_array', max_cycles=args.max_cycles)
    obs = env.reset(seed=args.seed)

    # Initialize rendering and get viewer
    _ = env.render()

    # Navigate PettingZoo structure to find viewer and fix bounds
    mpe_env = env.env.unwrapped
    inner_env = getattr(mpe_env, "env", None)

    if inner_env and hasattr(inner_env, "viewer") and inner_env.viewer:
        import types
        viewer = inner_env.viewer
        viewer.set_bounds = types.MethodType(
            lambda self, *_, **__: setattr(self, 'left', -1.5) or 
                                   setattr(self, 'right', 1.5) or 
                                   setattr(self, 'bottom', -1.5) or 
                                   setattr(self, 'top', 1.5), viewer)
        viewer.set_bounds(-1.5, 1.5, -1.5, 1.5)
        print("[INFO] Camera bounds fixed to [-1.5, 1.5]")

    # ==================== LOAD MODEL ====================
    print(f"[INFO] Loading trained MADDPG model from: {args.model_path}")
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    maddpg = MADDPG.init_from_save(args.model_path)
    maddpg.prep_rollouts(device=args.device)

    n_agents = len(env.agents)
    print(f"[INFO] Number of agents: {n_agents}")

    expected_input_dims = [agent.policy.fc1.in_features for agent in maddpg.agents]
    for i, dim in enumerate(expected_input_dims):
        print(f"  Agent {i}: input dim {dim}")

    # ==================== DELAY SETUP ====================
    use_delays = args.min_delay > 0 or args.max_delay > 0
    
    if use_delays:
        print(f"\n[INFO] Evaluation with time-varying delays:")
        print(f"  Min delay: {args.min_delay}")
        print(f"  Max delay: {args.max_delay}")
        print(f"  OU theta: {args.ou_theta}")
        print(f"  OU sigma: {args.ou_sigma}")
        
        # Calculate buffer size based on max_delay
        delay_buffer_size = int(np.ceil(args.max_delay)) + 1
        
        # Initialize OU processes for each agent
        ou_processes = []
        for agent_idx in range(n_agents):
            ou = DelayOUNoise(
                min_delay=args.min_delay,
                max_delay=args.max_delay,
                theta=args.ou_theta,
                sigma=args.ou_sigma,
                dt=1.0
            )
            ou_processes.append(ou)
        
        # Initialize action buffers for each agent
        action_buffers = []
        for agent_idx in range(n_agents):
            # Get action dimension from environment
            action_dim = env.action_space[agent_idx].shape[0]
            zero_actions = [np.zeros(action_dim, dtype=np.float32) 
                           for _ in range(delay_buffer_size)]
            action_buffers.append(zero_actions)
        
        print(f"  Action buffer size: {delay_buffer_size}")
        
        # Augment initial observations with action history
        obs_list = [obs[a] for a in env.agents] if isinstance(obs, dict) else obs
        for a_i in range(n_agents):
            agent_obs = obs_list[a_i]
            for action_idx in range(delay_buffer_size):
                agent_obs = np.append(agent_obs, action_buffers[a_i][action_idx])
            obs_list[a_i] = agent_obs
        
        delay_history = []
    else:
        print("\n[INFO] Evaluation WITHOUT delays")
        obs_list = [obs[a] for a in env.agents] if isinstance(obs, dict) else obs
        delay_history = None

    # ==================== ROLLOUT ====================
    print(f"\n[INFO] Starting evaluation rollout (max {args.max_steps} steps)...")

    frames = []
    step_i = 0
    total_rewards = {agent: 0.0 for agent in env.agents}

    while step_i < args.max_steps:
        # Debug: print positions periodically
        if args.verbose and step_i % 50 == 0:
            try:
                print(f"Step {step_i} | Landmark pos: {mpe_env.world.landmarks[0].state.p_pos}")
            except:
                pass

        # Prepare observations as torch tensors
        torch_obs = []
        for i, o in enumerate(obs_list):
            o_flat = np.asarray(o, dtype=np.float32).ravel()
            expected_dim = expected_input_dims[i]
            
            # Pad or truncate to expected input dimension
            if o_flat.size < expected_dim:
                o_padded = np.concatenate([o_flat, np.zeros(expected_dim - o_flat.size, dtype=np.float32)])
            else:
                o_padded = o_flat[:expected_dim]
            
            torch_obs.append(torch.tensor(np.expand_dims(o_padded, 0), dtype=torch.float32))

        # Get agent actions from policy
        with torch.no_grad():
            torch_actions = maddpg.step(torch_obs, explore=False)
        policy_actions = [ta.cpu().numpy().squeeze().astype(np.float32) for ta in torch_actions]

        # Apply delays if enabled
        if use_delays:
            env_actions = []
            step_delays = []
            
            for agent_idx in range(n_agents):
                # Sample delay from OU process
                current_delay = ou_processes[agent_idx].sample_delay()
                step_delays.append(current_delay)
                
                # Compute virtual action with delay
                virtual_action = compute_virtual_action(
                    action_buffers[agent_idx],
                    current_delay
                )
                
                # Clip to valid action range
                low = env.action_space[agent_idx].low
                high = env.action_space[agent_idx].high
                virtual_action = np.clip(virtual_action, low, high).astype(np.float32)
                env_actions.append(virtual_action)
                
                # Update action buffer: remove oldest, add newest
                action_buffers[agent_idx].pop()
                action_buffers[agent_idx].insert(0, policy_actions[agent_idx])
            
            delay_history.append(step_delays)
            actions = env_actions
        else:
            actions = policy_actions

        # Step environment
        next_obs, rewards, dones, infos = env.step(actions)
        
        # Accumulate rewards
        if isinstance(rewards, dict):
            for agent, reward in rewards.items():
                total_rewards[agent] += reward
        else:
            for i, reward in enumerate(rewards):
                total_rewards[env.agents[i]] += reward

        # Check termination
        done_flags = list(dones.values()) if isinstance(dones, dict) else dones
        task_completed = check_task_completed(env, threshold=args.completion_threshold)
        
        if all(done_flags) or task_completed:
            print(f"[INFO] Episode finished at step {step_i} (task_completed={task_completed})")
            break

        # Capture frame
        try:
            frame = env.render()
            if frame is not None:
                frames.append(np.array(frame))
        except Exception as e:
            if step_i == 0:
                print(f"[WARN] Render failed: {e}")

        # Update observations
        next_obs_list = [next_obs[a] for a in env.agents] if isinstance(next_obs, dict) else next_obs
        
        # Augment observations with action history if using delays
        if use_delays:
            for a_i in range(n_agents):
                agent_obs = next_obs_list[a_i]
                for action_idx in range(delay_buffer_size):
                    agent_obs = np.append(agent_obs, action_buffers[a_i][action_idx])
                next_obs_list[a_i] = agent_obs
        
        obs_list = next_obs_list
        step_i += 1

    print(f"[INFO] Rollout complete — captured {len(frames)} frames")
    
    # Print reward summary
    print("\n[INFO] Episode Rewards:")
    for agent, reward in total_rewards.items():
        print(f"  {agent}: {reward:.2f}")
    print(f"  Total: {sum(total_rewards.values()):.2f}")
    
    # Print delay statistics if applicable
    if use_delays and delay_history:
        delay_array = np.array(delay_history)
        print("\n[INFO] Delay Statistics:")
        print(f"  Mean: {np.mean(delay_array):.3f}")
        print(f"  Std:  {np.std(delay_array):.3f}")
        print(f"  Min:  {np.min(delay_array):.3f}")
        print(f"  Max:  {np.max(delay_array):.3f}")

    # ==================== VIDEO GENERATION ====================
    if frames:
        # Determine output format
        output_path = Path(args.output)
        output_ext = output_path.suffix.lower()
        
        if output_ext == '.gif':
            # Save as GIF using imageio or PIL
            try:
                import imageio
                imageio.mimsave(str(output_path), frames, fps=args.fps)
                print(f"\n[SUCCESS] Saved GIF -> {output_path} ({len(frames)} frames @ {args.fps} fps)")
            except ImportError:
                print("[ERROR] imageio not installed. Install with: pip install imageio")
                print("[INFO] Falling back to MP4 format...")
                output_path = output_path.with_suffix('.mp4')
                save_video_mp4(frames, str(output_path), args.fps)
        else:
            # Save as MP4 (default)
            if output_ext != '.mp4':
                output_path = output_path.with_suffix('.mp4')
            save_video_mp4(frames, str(output_path), args.fps)
        
        # Display in Jupyter/Colab if available
        try:
            from IPython.display import Video, display
            if output_path.suffix == '.mp4':
                display(Video(filename=str(output_path), embed=True))
        except ImportError:
            pass
    else:
        print("\n[ERROR] No frames captured — check render configuration.")

    # env.close()
    print("\n[INFO] Evaluation complete!")


def save_video_mp4(frames, output_path, fps):
    """Save frames as MP4 video"""
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        video_writer.write(frame)
    
    video_writer.release()
    print(f"\n[SUCCESS] Saved video -> {output_path} ({len(frames)} frames @ {fps} fps)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained MADDPG model")
    
    # Required arguments
    parser.add_argument("model_path", type=str,
                        help="Path to trained model file (.pt)")
    parser.add_argument("output", type=str,
                        help="Output file path (.mp4 or .gif)")
    
    # Environment arguments
    parser.add_argument("--env_name", type=str, default="simple_spread",
                        help="Environment name (default: simple_spread)")
    parser.add_argument("--max_cycles", type=int, default=150,
                        help="Max cycles per episode (default: 50)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    
    # Evaluation arguments
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode (default: 500)")
    parser.add_argument("--completion_threshold", type=float, default=0.15,
                        help="Distance threshold for task completion (default: 0.15)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for inference (default: cpu)")
    
    # Delay arguments
    parser.add_argument("--min_delay", type=float, default=0.0,
                        help="Minimum delay in timesteps (default: 0.0, no delay)")
    parser.add_argument("--max_delay", type=float, default=0.0,
                        help="Maximum delay in timesteps (default: 0.0, no delay)")
    parser.add_argument("--ou_theta", type=float, default=0.15,
                        help="OU process mean reversion rate (default: 0.15)")
    parser.add_argument("--ou_sigma", type=float, default=0.3,
                        help="OU process volatility (default: 0.3)")
    
    # Video arguments
    parser.add_argument("--fps", type=int, default=20,
                        help="Frames per second for output video (default: 20)")
    
    # Display arguments
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed debug information")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.min_delay < 0:
        parser.error("min_delay must be non-negative")
    if args.max_delay < args.min_delay:
        parser.error("max_delay must be >= min_delay")
    
    run_evaluation(args)
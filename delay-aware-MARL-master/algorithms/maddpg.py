import numpy as np
from pettingzoo.mpe.simple_speaker_listener_v4 import parallel_env as ssl_env
from pettingzoo.mpe.simple_spread_v3 import parallel_env as ss_env
from pettingzoo.mpe.simple_reference_v3 import parallel_env as sr_env
from pettingzoo.mpe.simple_tag_v3 import parallel_env as st_env
from pettingzoo.mpe.simple_push_v3 import parallel_env as sp_env

class MultiAgentEnvAdapter:
    """
    Adapter to convert PettingZoo parallel env (dict obs/action) to old MPE style list interface
    """
    def __init__(self, pettingzoo_env):
        self.env = pettingzoo_env
        obs_dict, _ = self.env.reset()
        self.agents = list(obs_dict.keys())
        self.n = len(self.agents)
        # Get agent types for MADDPG initialization
        self.agent_types = ['agent'] * len(self.agents)  # All cooperative in simple_spread
        
    def reset(self, **kwargs):
        obs_dict, _ = self.env.reset(**kwargs)
        obs_n = [obs_dict[a] for a in self.agents]
        return obs_n

    def step(self, action_n):
        """
        action_n: list of actions for each agent in self.agents order
        Returns: obs_n, reward_n, done_n, info_n
        """
        # Convert list to dict for PettingZoo env
        actions = {a: act for a, act in zip(self.agents, action_n)}
        obs_dict, rewards_dict, terminations, truncations, infos_dict = self.env.step(actions)

        obs_n = [obs_dict[a] for a in self.agents]
        reward_n = [rewards_dict[a] for a in self.agents]
        done_n = [terminations[a] or truncations[a] for a in self.agents]
        info_n = {'n': [infos_dict[a] for a in self.agents]}

        return obs_n, reward_n, done_n, info_n

    def render(self):
        """Render the environment with fixed camera bounds"""
        frame = self.env.render()
        if hasattr(self.env.unwrapped, "viewer") and self.env.unwrapped.viewer is not None:
            viewer = self.env.unwrapped.viewer
            if not hasattr(viewer, "_camera_locked"):
                viewer.set_bounds(-1.5, 1.5, -1.5, 1.5)
                import types
                def fixed_render(self, *args, **kwargs):
                    from pettingzoo.utils import rendering
                    self._build_geometry()
                    self._draw_world()
                    rendering.render_geometry(self.geoms, self.width, self.height)
                    return self.get_array() if self.render_mode == "rgb_array" else None
                viewer.render = types.MethodType(fixed_render, viewer)
                viewer._camera_locked = True
        return frame

    @property
    def action_space(self):
        return [self.env.action_space(a) for a in self.agents]
    
    @property
    def observation_space(self):
        return [self.env.observation_space(a) for a in self.agents]


def make_env(scenario_name, discrete_action=False, render_mode=None, max_cycles=100, **env_kwargs):
    """
    Create a multi-agent environment from PettingZoo MPE.
    
    Args:
        scenario_name: Name of the scenario (e.g., 'simple_spread')
        discrete_action: Whether to use discrete actions (default: False for continuous)
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        max_cycles: Maximum number of cycles per episode (default: 100)
        **env_kwargs: Additional environment parameters
    
    Returns:
        MultiAgentEnvAdapter wrapping the PettingZoo environment
    """
    scenario_dict = {
        'simple_speaker_listener': ssl_env,
        'simple_spread': ss_env,
        'simple_reference': sr_env,
        'simple_tag': st_env,
        'simple_push': sp_env,
    }
    
    if scenario_name not in scenario_dict:
        raise ValueError(f"Scenario {scenario_name} not found. Available: {list(scenario_dict.keys())}")
    
    # Build environment parameters
    env_params = {
        'max_cycles': max_cycles,  # CRITICAL: Must match episode_length in training
        'continuous_actions': not discrete_action,
        'render_mode': render_mode
    }
    env_params.update(env_kwargs)
    
    print(f"\n{'='*60}")
    print(f"CREATING ENVIRONMENT: {scenario_name}")
    print(f"{'='*60}")
    print(f"  Continuous actions: {env_params['continuous_actions']}")
    print(f"  Max cycles: {env_params['max_cycles']}")
    if env_kwargs:
        print(f"  Additional params: {env_kwargs}")
    
    # Create PettingZoo environment
    env = scenario_dict[scenario_name](**env_params)
    
    # Inspect the raw environment
    obs_dict, _ = env.reset()
    print(f"\n  Number of agents: {len(obs_dict)}")
    print(f"  Agent names: {list(obs_dict.keys())}")
    
    for agent_name in obs_dict.keys():
        obs_shape = obs_dict[agent_name].shape
        action_space = env.action_space(agent_name)
        obs_space = env.observation_space(agent_name)
        
        print(f"\n  {agent_name}:")
        print(f"    Obs space: {obs_space} (shape: {obs_shape})")
        print(f"    Act space: {action_space}", end="")
        if hasattr(action_space, 'shape'):
            print(f" (shape: {action_space.shape})")
        elif hasattr(action_space, 'n'):
            print(f" (n: {action_space.n})")
        else:
            print()
    
    print(f"{'='*60}\n")
    
    # Wrap in adapter
    env = MultiAgentEnvAdapter(env)
    return env
# multiagent_env_adapter.py
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
        # print(f"[DEBUG] Agents: {self.agents}, Total: {self.n}")
    def reset(self, **kwargs):
        obs_dict, _ = self.env.reset(**kwargs)
        obs_n = [obs_dict[a] for a in self.agents]
        
        # ADD THESE DEBUG PRINTS HERE:
        # print(f"[DEBUG] obs_n types: {[type(o) for o in obs_n]}")
        # print(f"[DEBUG] obs_n shapes: {[o.shape if hasattr(o, 'shape') else 'no shape' for o in obs_n]}")
        # print(f"[DEBUG] obs_n dtypes: {[o.dtype if hasattr(o, 'dtype') else 'no dtype' for o in obs_n]}")
        
        return obs_n

    def step(self, action_n):
        """
        action_n: list of actions for each agent in self.agents order
        Returns: obs_n, reward_n, done_n, info_n
        """
        # map list -> dict for PettingZoo env
        for i, act in enumerate(action_n):
            # print(f"[DEBUG] Agent {i} action type: {type(act)}, dtype: {act.dtype if hasattr(act, 'dtype') else 'N/A'}")
            # print(f"[DEBUG] Agent {i} action: {act}, min: {act.min()}, max: {act.max()}")
            
            # Verify bounds
            if hasattr(act, 'min') and hasattr(act, 'max'):
                if act.min() < 0 or act.max() > 1:
                    print(f"[ERROR] Agent {i} action OUT OF BOUNDS!")
        actions = {a: act for a, act in zip(self.agents, action_n)}
        obs_dict, rewards_dict, terminations, truncations, infos_dict = self.env.step(actions)

        obs_n = [obs_dict[a] for a in self.agents]
        reward_n = [rewards_dict[a] for a in self.agents]
        done_n = [terminations[a] or truncations[a] for a in self.agents]
        info_n = {'n': [infos_dict[a] for a in self.agents]}

        return obs_n, reward_n, done_n, info_n

    def render(self):
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
        spaces = [self.env.action_space(a) for a in self.agents]
        for i, sp in enumerate(spaces):
            # print(f"[DEBUG] Agent {self.agents[i]} Action space: {sp}, Type: {type(sp)}")
            pass
        return spaces
    @property
    def observation_space(self):
        spaces = [self.env.observation_space(a) for a in self.agents]
        for i, sp in enumerate(spaces):
            # print(f"[DEBUG] Agent {self.agents[i]} Observation space: {sp}, Type: {type(sp)}")
            pass
        return spaces


def make_env(scenario_name, discrete_action=False, render_mode=None, **env_kwargs):
    scenario_dict = {
        'simple_speaker_listener': ssl_env,
        'simple_spread': ss_env,
        'simple_reference': sr_env,
        'simple_tag': st_env,
        'simple_push': sp_env,
    }
    if scenario_name not in scenario_dict:
        raise ValueError(f"Scenario {scenario_name} not found in MPE2 environments")
    
    env_params = {
        'max_cycles': env_kwargs.pop('max_cycles', 25),
        'continuous_actions': not discrete_action,
        'render_mode': render_mode
    }
    env_params.update(env_kwargs)
    
    print(f"\n[MAKE_ENV DEBUG] Creating {scenario_name} with params:")
    print(f"  continuous_actions: {env_params['continuous_actions']}")
    print(f"  max_cycles: {env_params['max_cycles']}")
    print(f"  extra_params: {env_kwargs}")
    
    env = scenario_dict[scenario_name](**env_params)
    
    # Check the raw environment before adapter
    print(f"\n[MAKE_ENV DEBUG] Raw PettingZoo environment created")
    obs_dict, _ = env.reset()
    print(f"  Agents: {list(obs_dict.keys())}")
    for agent_name in obs_dict.keys():
        obs_shape = obs_dict[agent_name].shape
        action_space = env.action_space(agent_name)
        obs_space = env.observation_space(agent_name)
        print(f"  {agent_name}:")
        print(f"    Observation shape: {obs_shape}")
        print(f"    Observation space: {obs_space}")
        print(f"    Action space: {action_space}")
        if hasattr(action_space, 'shape'):
            print(f"    Action shape: {action_space.shape}")
        elif hasattr(action_space, 'n'):
            print(f"    Action n (discrete): {action_space.n}")
    
    env = MultiAgentEnvAdapter(env)
    return env

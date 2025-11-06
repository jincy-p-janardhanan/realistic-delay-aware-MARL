import torch
import torch.nn.functional as F
from gymnasium.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent
import numpy as np

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False, use_sigmoid=False, min_delay=2.0, max_delay=4.0):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
            use_sigmoid (bool): Use sigmoid activation for continuous actions ([0,1] range)
            min_delay (float): Minimum delay in time steps
            max_delay (float): Maximum delay in time steps
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim, use_sigmoid=use_sigmoid, 
                                 min_delay=min_delay, max_delay=max_delay,
                                 **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'gpu'
        self.critic_dev = 'gpu'
        self.trgt_pol_dev = 'gpu'
        self.trgt_critic_dev = 'gpu'
        self.niter = 0
        self.use_sigmoid = use_sigmoid
        self.min_delay = min_delay
        self.max_delay = max_delay

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]

    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        if self.alg_types[agent_i] == 'MADDPG':
            if self.discrete_action:
                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                zip(self.target_policies, next_obs)]
            else:
                all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        else:  # DDPG
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(next_obs[agent_i])),
                                       dim=1)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        if self.alg_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((*obs, *acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())

        if vf_loss < 0:
            print(f"[ERROR] Negative vf_loss: {vf_loss.item()}")
            print(f"  actual_value range: [{actual_value.min()}, {actual_value.max()}]")
            print(f"  target_value range: [{target_value.min()}, {target_value.max()}]")
    
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_action:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    all_pol_acs.append(onehot_from_logits(pi(ob)))
                else:
                    all_pol_acs.append(pi(ob))
            vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                              dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def update_adversaries(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents[:2]:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1
        
    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        for a in self.agents:
            a.policy = fn(a.policy)
        self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += oacsp.shape[0] if isinstance(oacsp, Box) else oacsp.n
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {
            'gamma': gamma, 'tau': tau, 'lr': lr,
            'hidden_dim': hidden_dim,
            'alg_types': alg_types,
            'agent_init_params': agent_init_params,
            'discrete_action': discrete_action
        }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance
    
    @classmethod    
    def init_from_env_with_delay(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, 
                      min_delay=2.0, max_delay=4.0, use_sigmoid=True):
        """
        Instantiate instance of this class from multi-agent environment with support for
        time-varying delays specified by min_delay and max_delay.
        
        Args:
            min_delay (float): Minimum delay in time steps
            max_delay (float): Maximum delay in time steps (used for buffer sizing)
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        
        # Calculate buffer size needed: ceiling of max_delay + 1
        delay_buffer_size = int(np.ceil(max_delay)) + 1
        
        print(f"[MADDPG DEBUG] Min delay: {min_delay}")
        print(f"[MADDPG DEBUG] Max delay: {max_delay}")
        print(f"[MADDPG DEBUG] Delay buffer size: {delay_buffer_size}")
        
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            
            num_out_pol = get_shape(acsp)
            
            # Policy input: observation + action_history
            # Buffer size is based on max_delay to handle worst case
            num_in_pol = obsp.shape[0] + delay_buffer_size * num_out_pol
            
            print(f"[MADDPG DEBUG] Agent: obs_dim={obsp.shape[0]}, action_dim={num_out_pol}")
            print(f"[MADDPG DEBUG]   Policy input dim: {num_in_pol} = {obsp.shape[0]} + {delay_buffer_size}*{num_out_pol}")
            
            # In init_from_env_with_delay:
            if algtype == "MADDPG":
                num_in_critic = 0
                # Add all observations (which already include action history)
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                    # Add action history size
                    action_dim = get_shape(env.action_space[0]) 
                    num_in_critic += delay_buffer_size * action_dim
                
                # Add current actions only (not action history again!)
                for oacsp in env.action_space:
                    action_dim = get_shape(oacsp)
                    num_in_critic += action_dim  # ONLY current action
                
                print(f"[MADDPG DEBUG]   Critic input dim (MADDPG): {num_in_critic}")
            else:  # DDPG
                num_in_critic = obsp.shape[0] + get_shape(acsp) * (1 + delay_buffer_size)
                print(f"[MADDPG DEBUG]   Critic input dim (DDPG): {num_in_critic}")
            
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        
        init_dict = {
            'gamma': gamma, 'tau': tau, 'lr': lr,
            'hidden_dim': hidden_dim,
            'alg_types': alg_types,
            'agent_init_params': agent_init_params,
            'discrete_action': discrete_action,
            'use_sigmoid': use_sigmoid,
            'min_delay': min_delay,
            'max_delay': max_delay
        }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance
    
    @classmethod    
    def init_from_env_with_runner(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, 
                      min_delay=2.0, max_delay=4.0, file_name=''):
        """
        Instantiate with a pre-trained runner agent (agent 2) for time-varying delays.
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        
        delay_buffer_size = int(np.ceil(max_delay)) + 1
        
        for acsp, obsp, algtype, atype in zip(env.action_space, env.observation_space,
                                       alg_types, env.agent_types):
            if atype == 'adversary':
                num_in_pol = obsp.shape[0] + delay_buffer_size * acsp.shape[0]
            else:
                num_in_pol = obsp.shape[0]
            
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:
                discrete_action = True
                get_shape = lambda x: x.n
            
            num_out_pol = get_shape(acsp)
            
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
                    num_in_critic += delay_buffer_size * get_shape(oacsp)
                
                # Adjust for runner not having delay awareness
                num_in_critic -= 2 * delay_buffer_size
            else:
                if atype == 'adversary':
                    num_in_critic = obsp.shape[0] + get_shape(acsp) * (1 + delay_buffer_size)
                else:
                    num_in_critic = obsp.shape[0] + get_shape(acsp)
            
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        
        init_dict = {
            'gamma': gamma, 'tau': tau, 'lr': lr,
            'hidden_dim': hidden_dim,
            'alg_types': alg_types,
            'agent_init_params': agent_init_params,
            'discrete_action': discrete_action,
            'use_sigmoid': True,
            'min_delay': min_delay,
            'max_delay': max_delay
        }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        save_dict = torch.load(file_name)
        instance.agents[2].load_policy_params(save_dict['agent_params'][2])
        return instance  
    
    @classmethod    
    def init_from_env_with_runner_delay_unaware(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, 
                      min_delay=0.0, max_delay=0.0, file_name=''):
        """
        Instantiate with delay-unaware runner (no action history in observations).
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype, atype in zip(env.action_space, env.observation_space,
                                       alg_types, env.agent_types):
            num_in_pol = obsp.shape[0]
            
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                    
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {
            'gamma': gamma, 'tau': tau, 'lr': lr,
            'hidden_dim': hidden_dim,
            'alg_types': alg_types,
            'agent_init_params': agent_init_params,
            'discrete_action': discrete_action,
            'use_sigmoid': True,
            'min_delay': min_delay,
            'max_delay': max_delay
        }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        save_dict = torch.load(file_name)
        instance.agents[2].load_policy_params(save_dict['agent_params'][2])
        return instance
    
    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance
    
    @classmethod
    def init_runner_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        instance.agents[2].load_params(save_dict['agent_params'][2])
        return instance
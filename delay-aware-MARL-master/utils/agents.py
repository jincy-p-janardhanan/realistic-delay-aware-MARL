import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import numpy as np

device = 'cuda'

class DDPGAgent(object):
    """
    Delay-aware wrapper for DDPG with action history observation augmentation.
    Supports time-varying delays between min_delay and max_delay.
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True, delay_step=None, 
                 min_delay=None, max_delay=None, use_sigmoid=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input (obs + action_history)
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
            hidden_dim (int): number of hidden dimensions
            lr (float): learning rate
            discrete_action (bool): whether action space is discrete
            delay_step (float): DEPRECATED - use min_delay/max_delay instead
            min_delay (float): minimum delay in timesteps (can be fractional)
            max_delay (float): maximum delay in timesteps (can be fractional)
            use_sigmoid (bool): use sigmoid for continuous actions (outputs [0,1])
        """
        self.discrete_action = discrete_action
        self.use_sigmoid = use_sigmoid
        
        # Handle backward compatibility: if delay_step is provided, use it for both min/max
        if delay_step is not None:
            self.min_delay = delay_step
            self.max_delay = delay_step
            print(f"[DDPGAgent] Using legacy delay_step={delay_step} (both min and max)")
        elif min_delay is not None and max_delay is not None:
            self.min_delay = min_delay
            self.max_delay = max_delay
        else:
            # Default values if nothing is specified
            self.min_delay = 0.0
            self.max_delay = 0.0
            print("[DDPGAgent] WARNING: No delay parameters specified, defaulting to 0")
        
        print(f"[DDPGAgent] Initializing agent:")
        print(f"  Policy input dim: {num_in_pol}")
        print(f"  Policy output dim: {num_out_pol}")
        print(f"  Critic input dim: {num_in_critic}")
        print(f"  Min delay: {self.min_delay}")
        print(f"  Max delay: {self.max_delay}")
        print(f"  Use sigmoid: {use_sigmoid}")
        
        # Policy network - uses sigmoid for continuous actions
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action,
                                 use_sigmoid=use_sigmoid)
        
        # Critic network
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        
        # Target networks
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action,
                                        use_sigmoid=use_sigmoid)
        
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        if next(self.policy.parameters()).is_cuda:
            device = torch.device("cuda")
            obs = obs.to(device)
        else:
            device = torch.device("cpu")
            obs = obs.to(device)
        action = self.policy(obs)
        
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                noise = Variable(Tensor(self.exploration.noise()),
                                requires_grad=False).to(device)
                action = action + noise
            
            # Clamp to [0, 1] if using sigmoid, or [-1, 1] if using tanh
            if self.use_sigmoid:
                # Use epsilon = 1e-6 to stay strictly inside (0, 1)
                epsilon = 1e-6
                action = action.clamp(epsilon, 1.0 - epsilon)
            else:
                # For tanh: clamp to (-1 + eps, 1 - eps)
                epsilon = 1e-6
                action = action.clamp(-1.0 + epsilon, 1.0 - epsilon)
        
        # EXTRA SAFETY: Ensure strictly within bounds and correct dtype
        # (This block appears redundant with the above, but kept for compatibility)
        if self.use_sigmoid:
            epsilon = 1e-6
            action = action.clamp(epsilon, 1.0 - epsilon)
        else:
            epsilon = 1e-6
            action = action.clamp(-1.0 + epsilon, 1.0 - epsilon)
        
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
    
    def load_policy_params(self, params):
        """Load only policy parameters (useful for transfer learning)"""
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
    
    def get_delay_info(self):
        """Return delay configuration for this agent"""
        return {
            'min_delay': self.min_delay,
            'max_delay': self.max_delay,
            'delay_range': self.max_delay - self.min_delay,
            'mean_delay': (self.min_delay + self.max_delay) / 2.0
        }
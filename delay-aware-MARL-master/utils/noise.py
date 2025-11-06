import numpy as np


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    """
    Ornstein-Uhlenbeck noise process.
    
    Can be used for both action exploration and delay generation.
    For action exploration: action_dimension = action space dimension
    For delay generation: action_dimension = 1 (scalar delay)
    """
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2, dt=1.0):
        """
        Args:
            action_dimension: Dimension of the noise vector
            scale: Scaling factor applied to the final noise
            mu: Mean/equilibrium value the process reverts to
            theta: Mean reversion rate (higher = faster return to mean)
            sigma: Volatility/noise magnitude
            dt: Time step size (for numerical integration)
        """
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        """Reset the process to its initial state (mean value)"""
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        """
        Generate next noise sample using Ornstein-Uhlenbeck process.
        
        Returns:
            Noise vector scaled by self.scale
        """
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
    
    def sample(self):
        """
        Generate a single scalar sample (useful for delay generation).
        
        Returns:
            Scalar value without scaling
        """
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(len(x))
        self.state = x + dx
        return self.state[0] if self.action_dimension == 1 else self.state
    
    def set_params(self, mu=None, theta=None, sigma=None, scale=None):
        """
        Update OU process parameters dynamically.
        
        Args:
            mu: New mean value (optional)
            theta: New mean reversion rate (optional)
            sigma: New volatility (optional)
            scale: New scaling factor (optional)
        """
        if mu is not None:
            self.mu = mu
        if theta is not None:
            self.theta = theta
        if sigma is not None:
            self.sigma = sigma
        if scale is not None:
            self.scale = scale


class DelayOUNoise(OUNoise):
    """
    Specialized OU noise for generating time-varying delays.
    Automatically clips values to [min_delay, max_delay] range.
    """
    def __init__(self, min_delay, max_delay, theta=0.15, sigma=0.2, dt=1.0):
        """
        Args:
            min_delay: Minimum delay value
            max_delay: Maximum delay value
            theta: Mean reversion rate (higher = faster return to mean)
            sigma: Volatility/noise magnitude
            dt: Time step size
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        
        # Set mean to midpoint of delay range
        mu = (min_delay + max_delay) / 2.0
        
        # Initialize with scale=1.0 since we handle bounds separately
        super().__init__(action_dimension=1, scale=1.0, mu=mu, theta=theta, sigma=sigma, dt=dt)
    
    def sample_delay(self):
        """
        Generate a delay value clipped to [min_delay, max_delay].
        
        Returns:
            Scalar delay value
        """
        # Get raw OU sample
        raw_value = super().sample()
        
        # Clip to valid delay range
        delay = np.clip(raw_value, self.min_delay, self.max_delay)
        
        return delay
    
    def get_info(self):
        """Return information about the delay generator"""
        return {
            'min_delay': self.min_delay,
            'max_delay': self.max_delay,
            'current_state': self.state[0],
            'mean': self.mu,
            'theta': self.theta,
            'sigma': self.sigma
        }
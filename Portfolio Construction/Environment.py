import gym
from gym import spaces
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis as kurt
import matplotlib.pyplot as plt

class TradingEnvironment(gym.Env):
    """
    A simple trading environment of N assets whose returns are modeled by geometric brownian motion.

    Parameters
    ----------
    S0 : float
        Initial capital of the agent

    N : int
        Number of assets in the universe

    min_drift : float
        Lower bound of the drift of the assets

    max_drift : float
        Upper bound of the drift of the assets

    min_volatility : float
        Lower bound of the volatility of the assets

    max_volatility : float
        Upper bound of the volatility of the assets

    T : int
        Number of time steps per episode

    H : int
        Number of previous time steps to be accounted for in observations

    """
    def __init__(
            self, S0=100_000, N=10, min_drift=-0.05, max_drift=0.05, min_volatility=0.05, 
            max_volatility=0.25, T=252, H=20
            ):
        super(TradingEnvironment, self).__init__()
        
        self.N = N                              # number of assets in the universe

        self.min_drift = min_drift              # lower bound of the drift of the assets
        self.max_drift = max_drift              # upper bound of the drift of the assets
        self.min_volatility = min_volatility    # lower bound of the volatility of the assets
        self.max_volatility = max_volatility    # upper bound of the volatility of the assets

        self.T = T                              # maximal time
        self.t = H                              # current time
        self.H = H                              # how far back in time to look

        self.S0 = S0                            # initial capital
        self.St = S0                            # capital at time t
        self.Wt = np.zeros(self.N)              # weights of the assets in the portfolio at time t (initially 0)

        self.portfolio_returns = [0]            # list of portfolio returns (%) at each time step
        self.portfolio_values = [S0]            # list of portfolio values ($) at each time step

        # Initializing Action and Observation Spaces
        self.set_action_space()
        self.set_observation_space()

        # Initializing the state
        self.reset()

    def reset(self):
        """
        Resets the environment
        
        Parameters
        ----------
        None
            
        Returns
        -------
        tuple
            (obs, info)

        """
        
        self.t = self.H
        self.St = self.S0
        self.Wt = np.zeros(self.N)
        self.portfolio_returns = [0]
        self.portfolio_values = [self.S0]
        self.simulate_market()
        
        return (self.state(), {})

    def set_action_space(self):
        """
        Sets the action space of the environment

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        self.action_space = spaces.Box(low=0, high=1, shape=(self.N,), dtype=np.float32)

    def set_observation_space(self):
        """
        Sets the observation space of the environment as an 5 x N array of floats 
        representing statistical measures of the last H returns of each asset:
            - the first row contains the mean 
            - the second row contains the standard deviations 
            - the third row contains the variance 
            - the fourth row contains the skewness
            - the fifth row contains the kurtosis

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        drifts = spaces.Box(low=-np.inf, high=np.inf, shape=(self.N,), dtype=np.float32)
        volatilities = spaces.Box(low=0, high=np.inf, shape=(self.N,), dtype=np.float32)
        variance = spaces.Box(low=0, high=np.inf, shape=(self.N,), dtype=np.float32)
        skewness = spaces.Box(low=-np.inf, high=np.inf, shape=(self.N,), dtype=np.float32)
        kurtosis = spaces.Box(low=-np.inf, high=np.inf, shape=(self.N,), dtype=np.float32)
        
        self.observation_space = spaces.Tuple((drifts, volatilities, variance, skewness, kurtosis))

    def simulate_market(self):
        """
        Simulates the market by generating random returns for each asset based on geometric Brownian motion

        Parameters
        ----------
        None

        Returns
        -------
        None

        """


        # Generating random drifts and volatilities for each asset
        self.drifts = np.random.uniform(self.min_drift, self.max_drift, self.N)
        self.volatilities = np.random.uniform(self.min_volatility, self.max_volatility, self.N)

        # Generating random returns for each asset based on geometric Brownian motion
        asset_returns = np.zeros((self.T, self.N))
        for i in range(self.N):
            asset_returns[:,i] = 1 - np.exp((self.drifts[i] - 0.5 * self.volatilities[i]**2) + self.volatilities[i] * np.random.normal(0, 1, self.T))

        # Saving the returns as an attribute
        self.asset_returns = asset_returns
        

    def state(self):
        """
        Returns the observation space

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            (drifts, volatilities, variance, skewness, kurtosis)
        
        """

        returns = self.asset_returns[self.t-self.H:self.t]
        
        drifts = returns.mean(axis=0).astype(np.float32)
        volatilities = returns.std(axis=0).astype(np.float32)
        variance = returns.var(axis=0).astype(np.float32)
        skewness = skew(returns, axis=0).astype(np.float32)
        kurtosis = kurt(returns, axis=0).astype(np.float32)

        return (drifts, volatilities, variance, skewness, kurtosis)
    
    def step(self, action):
        """
        Takes a step in the environment based on the agent taking an action
        
        Parameters
        ----------
        action : np array
            The weights of the assets in the new portfolio
            
        Returns
        -------
        obs : tuple
            the observation for time step t

        reward : float
            the reward for time step t

        done : bool
            whether the episode is over

        info : dict
            additional information

        """

        self.t += 1
        
        # calculate the returns at time t
        returns = self.asset_returns[self.t]
        
        net_return = np.dot(action, returns)

        # update the portfolio values
        self.St = self.St * (1 + net_return)
        self.Wt = action
        self.portfolio_values.append(self.St)
        self.portfolio_returns.append(net_return)
        
        if self.t + 1 == self.T:
            done = True
        else:
            done = False
        
        return (self.state(), self.calculate_reward(), done, {})

    def calculate_reward(self):
        """
        Returns the reward
        
        Parameters
        ----------
        None
        
        Returns
        -------
        reward : float
            the reward
        
        """
        Returns, Risk, Sharpe, Sortino, MaxDD, VaR = self.calculte_portfolio_metrics()

        reward = Returns + Sharpe + Sortino - Risk - MaxDD - VaR

        return reward
        

    def calculte_portfolio_metrics(self):
        """
        Calculates the portfolio performance by the following metrics:
            - Return
            - Risk
            - Sharpe Ratio
            - Sortino Ratio
            - Maximum Drawdown
            - Calmar Ratio
            - Value at Risk
        
        Parameters
        ----------
        None
        
        Returns
        -------
        metrics : tuple
            (return, risk, sharpe_ratio, sortino_ratio, maximum_drawdown, value_at_risk)
        
        """
        net_return = (self.St - self.S0) / self.S0 * 100
        risk = np.std(self.portfolio_returns)
        sharpe_ratio = np.mean(self.portfolio_returns) / risk
        sortino_ratio = np.mean(self.portfolio_returns) / np.std([r for r in self.portfolio_returns if r < 0])
        maximum_drawdown = np.max(np.maximum.accumulate(self.portfolio_values) - self.portfolio_values) / np.max(self.portfolio_values)
        value_at_risk = np.percentile(self.portfolio_returns, 5) 

        return (net_return, risk, sharpe_ratio, sortino_ratio, maximum_drawdown, value_at_risk)

    def render(self):
        """
        Renders the environment
        
        Parameters
        ----------
        mode : str
            The mode in which the environment is rendered
            
        Returns
        -------
        None
        """
        pass


env = TradingEnvironment(C=0)
while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.render()
        break
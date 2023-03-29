import gym
from gym import spaces
import numpy as np


class TradingEnvironment(gym.Env):
    """
    A simple trading environment of N assets whose returns are modeled by geometric brownian motion.

    """
    def __init__(
            self, 
            initial_capital: float = 100_000, 
            num_assets: int = 10,
            maximal_time: int = 252,
            hind_sight: int = 30,
            investment_horizon: int = 30,       # number of days to hold the portfolio
            transaction_cost: float = 0.001,    # transaction cost as a percentage of the transaction value
            mu_mean: float = 0,                 # average drift of all the assets
            mu_deviation: float = 0.2,          # deviation in the drift of the assets
            sigma_mean: float = 0,              # average volatility of all the assets
            sigma_deviation: float = 1,         # deviation in the volatility of the assets
            ):
        super(TradingEnvironment, self).__init__()
        np.set_printoptions(suppress=True)

        self.num_assets = num_assets
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_cost = transaction_cost

        self.maximal_time = maximal_time
        self.hind_sight = hind_sight
        self.current_time = self.hind_sight
        self.investment_horizon = investment_horizon

        self.current_weights = np.zeros(self.num_assets)
        self.portfolio_returns = [0]
        self.portfolio_values = [self.initial_capital]

        self.mu_mean = mu_mean
        self.mu_deviation = mu_deviation
        self.sigma_mean = sigma_mean
        self.sigma_deviation = sigma_deviation
        
        self._set_action_space()
        self._set_observation_space()

        self.reset()

    def _set_action_space(self):
        """
        Sets the action space of the environment

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)

    def _set_observation_space(self):
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
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5, self.num_assets), dtype=np.float32)

    def _simulate_market(self) -> None:
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
        self.drifts = np.random.normal(self.mu_mean, self.mu_deviation, self.num_assets)
        self.volatilities = np.random.normal(self.sigma_mean, self.sigma_deviation, self.num_assets)

        # Generating random returns for each asset based on geometric Brownian motion
        normal_returns = np.random.normal(size=(self.maximal_time, self.num_assets))
        asset_returns = np.exp(
            (self.drifts - 0.5 * self.volatilities**2)[:, np.newaxis] * np.ones((self.maximal_time, self.num_assets)) + 
            self.volatilities[:, np.newaxis] * normal_returns.T) - 1

        # Saving the returns as an attribute
        self.asset_returns = asset_returns
    
    def _calculate_state_value(self) -> float:
        """
        Calculates the value of the state
        
        Parameters
        ----------
        None
        
        Returns
        -------
        state_value : float
            the value of the state
        
        """
        state_value = 0
        
        return state_value

    def _calculate_reward(self):
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
        Returns, Risk, Sharpe, Sortino, MaxDD, VaR = self._calculte_portfolio_metrics()

        reward = Returns + Sharpe + Sortino - Risk - MaxDD - VaR

        return reward
        

    def _calculte_portfolio_metrics(self):
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

    def state(self) -> tuple:
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
        # Get the last H returns for each asset
        returns = self.asset_returns[-self.hind_sight:, :]
        
        # Calculate the statistical measures
        mean = np.mean(returns, axis=0)
        std = np.std(returns, axis=0)
        variance = np.var(returns, axis=0)
        skewness = np.mean(((returns - mean) / std) ** 3, axis=0)
        kurtosis = np.mean(((returns - mean) / std) ** 4, axis=0) - 3

        return np.stack((mean, std, variance, skewness, kurtosis), axis=0)

    def reset(self) -> tuple:
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
        
        self.current_time = self.hind_sight
        self.current_capital = self.initial_capital
        self.current_weights = np.zeros(self.num_assets)
        self.portfolio_returns = [0]
        self.portfolio_values = [self.initial_capital]
        self._simulate_market()
        
        return (self.state(), {})
    
    def step(self, action: np.array) -> tuple:
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

        self.current_time += 1
        
        # calculate the returns at time t
        returns = self.asset_returns[self.current_time]
        
        net_return = np.dot(action, returns)

        # update the portfolio values
        self.current_capital = self.current_capital * (1 + net_return)
        self.current_weights = action
        self.portfolio_values.append(self.current_capital)
        self.portfolio_returns.append(net_return)
        
        resultant_state = self.state()

        done = (self.current_time == self.maximal_time - 1)
        
        reward = self._calculate_reward()
        
        info = {}
        
        return (resultant_state, reward, done, info)
    
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


env = TradingEnvironment()
while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.render()
        break
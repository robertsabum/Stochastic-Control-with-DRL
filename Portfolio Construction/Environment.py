import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class TradingEnvironment(gym.Env):
    """
    A simple trading environment of N assets whose returns are modeled by geometric brownian motion.

    """
    def __init__(
            self, 
            initial_capital: float = 100_000, 
            num_assets: int = 5,
            year_length: int = 252,             # number of days in the simulation (20 years)
            num_years: int = 2,                 # number of years in the simulation
            transaction_cost: float = 0.0001,   # transaction cost as a percentage of the transaction value
            min_drift: float = -0.2,            # minimum drift of the assets
            max_drift: float = 0.2,             # maximum drift of the assets
            min_volatility: float = 0.10,       # minimum volatility of the assets
            max_volatility: float = 0.25,       # maximum volatility of the assets
            mean_starting_price: float = 100,   # mean starting price of the assets
            std_starting_price: float = 10      # standard deviation of the starting price of the assets
            ):
        super(TradingEnvironment, self).__init__()
        np.set_printoptions(suppress=True)

        self.num_assets = num_assets
        self.initial_capital = initial_capital
        self.current_portfolio_value = initial_capital
        self.transaction_cost = transaction_cost

        self.period_length = year_length
        self.maximal_time = year_length * num_years
        self.current_time = year_length

        self.current_weights = np.zeros(self.num_assets)
        self.portfolio_returns = [0]
        self.portfolio_values = [self.initial_capital]

        self.mean_asset_returns = np.random.uniform(min_drift, max_drift, num_assets)
        self.asset_volatilities = np.random.uniform(min_volatility, max_volatility, num_assets)
        self.initial_asset_prices = np.random.normal(mean_starting_price, std_starting_price, num_assets)

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

        self.action_space = spaces.Box(low=0, high=np.inf, shape=(self.num_assets,), dtype=np.float32)

    def _set_observation_space(self):
        """
        Sets the observation space of the environment as tuple containing:
            - the current set of weights
            - the last investment horizon mean returns of the assets
            - the last investment horizon covariance matrix of the assets

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        
        current_weights = spaces.Box(low=0, high=np.inf, shape=(self.num_assets,), dtype=np.float32)
        mean_returns = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets,), dtype=np.float32)
        covariance_matrix = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets, self.num_assets), dtype=np.float32)

        self.observation_space = spaces.Tuple((current_weights, mean_returns, covariance_matrix))

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

        self.asset_returns = np.exp(
            (self.mean_asset_returns - 0.5 * self.asset_volatilities**2) * 
            (1/self.period_length) + 
            self.asset_volatilities * np.sqrt(1/self.period_length) * 
            np.random.normal(size=(self.maximal_time, self.num_assets))) - 1
        
        self.asset_prices = np.cumprod(self.asset_returns + 1, axis=0) * self.initial_asset_prices
    
    def _asset_statistical_measures(self, time: int) -> tuple:
        """
        Calculates the statistical measures of the assets
        
        Parameters
        ----------
        time : int
            the time at which the measures are calculated
            
        Returns
        -------
        dict
            {mean_returns, volatilities, covariance_matrix, skewness, kurtosis}
        
        """
        returns = self.asset_returns[time - self.period_length:time]
        
        mean = np.mean(returns, axis=0)
        standard_deviation = np.std(returns, axis=0)
        skewness = (1 / (standard_deviation**3)) * np.sum((returns - mean)**3, axis=0)
        kurtosis = (1 / (standard_deviation**4)) * np.sum((returns - mean)**4, axis=0)
        covariance_matrix = np.cov(returns, rowvar=False)

        return {
            'mean_returns': mean,
            'volatilities': standard_deviation,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'covariance_matrix': covariance_matrix
        }
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        Calculates the reward of the state-action pair
        
        Parameters
        ----------
        action : np.ndarray
            the action taken by the agent
            
        Returns
        -------
        reward : float
            the reward of the action pair
        
        """
        stats = self._asset_statistical_measures(self.current_time)
        reward = self.portfolio_returns[-1] + np.dot(stats['mean_returns'], action) - (1/2 * np.dot(action, np.dot(stats['covariance_matrix'], action))) 
        
        return reward

    def _calculte_portfolio_metrics(self):
        """
        Calculates the portfolio performance by the following metrics:
            - Return
            - Risk
            - Sharpe Ratio
            - Maximum Drawdown
            - Value at Risk
        
        Parameters
        ----------
        None
        
        Returns
        -------
        metrics : tuple
            (return, risk, sharpe_ratio, maximum_drawdown, value_at_risk)
        
        """
        portfolio_return = (self.current_portfolio_value / self.initial_capital) - 1 
        risk = np.std(self.portfolio_returns)
        sharpe_ratio = np.mean(self.portfolio_returns) / risk
        maximum_drawdown = np.min((self.portfolio_values - np.maximum.accumulate(self.portfolio_values)) / np.maximum.accumulate(self.portfolio_values))
        value_at_risk = np.percentile(self.portfolio_returns, 5)

        return (portfolio_return, risk, sharpe_ratio, maximum_drawdown, value_at_risk)

    def state(self) -> tuple:
        """
        Returns the observation space

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            (current_weights, mean_returns, covariance_matrix)
        
        """        

        # Calculate the statistical measures
        stats = self._asset_statistical_measures(self.current_time)

        return (self.current_weights, stats['mean_returns'], stats['covariance_matrix'])

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
        
        self.current_time = self.period_length
        self.current_portfolio_value = self.initial_capital
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
        action /= np.sum(action)
        
        # calculate the returns at time t
        days_returns = self.asset_returns[self.current_time]
        transaction_cost = np.sum(np.abs(self.current_weights - action)) * self.transaction_cost
        weighted_return = np.dot(action, days_returns)

        net_return = weighted_return - transaction_cost


        # adjust weights to account for the changes in individual asset prices
        self.current_weights = action * (days_returns + 1) / (weighted_return + 1)

        # update the portfolio values
        self.current_portfolio_value *= (1 + net_return)
        self.portfolio_values.append(self.current_portfolio_value)
        self.portfolio_returns.append(net_return)
        
        next_state = self.state()

        done = (self.current_time == self.maximal_time - 1)
        
        reward = self._calculate_reward(action)

        info = {"return": net_return, "transaction_cost": transaction_cost * self.portfolio_values[-2], "reward": reward, "done": done}
        
        return (next_state, reward, done, info)
    
    def _render_terminal(self):
        """
        Renders the environment in the terminal
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        portfolio_return, risk, sharpe_ratio, maximum_drawdown, value_at_risk = self._calculte_portfolio_metrics()
        
        print("=" * 50)
        print("Current time: {}".format(self.current_time - 251))
        print("Current portfolio value: {}".format(self.current_portfolio_value))
        print("Current weights: {}".format(self.current_weights))
        print("Return\t: {}".format(portfolio_return))
        print("Risk\t: {}".format(risk))
        print("Sharpe\t: {}".format(sharpe_ratio))
        print("Max DD\t: {}".format(maximum_drawdown))
        print("VaR\t: {}".format(value_at_risk))
        print("")

    def _render_plot(self):
        """
        Renders the environment in a graph
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.plot(self.portfolio_values)
        plt.title("Portfolio Value")
        plt.xlabel("Time (days)")
        plt.ylabel("Value ($)")
        plt.show()
        
    def render(self, mode='terminal'):
        """
        Renders the environment
        
        Parameters
        ----------
        mode : str
            the mode to render the environment in
            
        Returns
        -------
        None
        """
        if mode == 'terminal':
            self._render_terminal()

        elif mode == 'plot':
            self._render_plot()


if __name__ == "__main__":

    env = TradingEnvironment()
    state = env.state()
    while True:
        next_state, reward, done, info = env.step(np.ones(5) / 5)
        print("Info: {}".format(info))

        if done:
            break

    env.render(mode='terminal')
    env.render(mode='plot')
import pickle
import sys
import gym
from gym import spaces
import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

class TradingEnvironment(gym.Env):
    """
    A simple trading environment of N assets based on historical data of the SP500

    """
    def __init__(
            self, 
            initial_capital: float = 100_000,   # initial capital of the portfolio
            num_assets: int = 5,                # number of assets in the universe each episode
            transaction_cost: float = 0.0001,   # transaction cost as a percentage of the transaction value
            hind_sight: int = 252,              # number of previous time steps to include in the observation
            maximal_time: int = 2520,           # maximum number of time steps in the environment (10 years of trading days)
            data_file: str = 'data.csv'         # file containing the prices of the assets
            ):
        super(TradingEnvironment, self).__init__()
        np.set_printoptions(suppress=True)

        self.__num_assets = num_assets
        self.__load_data(data_file)

        self.__initial_capital = initial_capital
        self.__current_portfolio_value = initial_capital
        self.__transaction_cost = transaction_cost

        self.__maximal_time = maximal_time
        self.__current_time = hind_sight
        self.__hind_sight = hind_sight

        self.__current_weights = np.zeros(num_assets)
        self.__portfolio_returns = [0]
        self.__portfolio_values = [initial_capital]

        self.__rng = Generator(PCG64())

        self.__universe_history = set()

        self.__set_action_space()
        self.__set_observation_space()

        self.reset()

    def __set_action_space(self):
        """
        Sets the action space of the environment

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.__num_assets,), dtype=np.float32)

    def __set_observation_space(self):
        """
        Sets the observation space of the environment as a vector containing:
            - the current set of weights
            - mean returns of the assets
            - volatilities of the assets
            - skewness of the assets
            - kurtosis of the assets

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.__num_assets * 5,), dtype=np.float32)

    def __load_data(self, file: str) -> None:
        """
        Loads the historical data of the assets

        Parameters
        ----------
        file : str
            the file containing the historical data of the assets

        Returns
        -------
        None

        """

        self.__asset_prices = pd.read_csv(file, index_col=0)
        self.__asset_returns = np.log(self.__asset_prices / self.__asset_prices.shift(1))
        self.__asset_returns.iloc[0] = 0
        
    def __sample_assets(self) -> None:
        """
        Chooses random assets to form the universe of assets for the current episode

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        universe = np.sort(self.__rng.choice(self.__asset_prices.columns, self.__num_assets, replace=False))
        while str(universe) in self.__universe_history:
            universe = self.__rng.choice(self.__asset_prices.columns, self.__num_assets, replace=False)

        self.__universe_history.add(str(universe))
        self.__current_universe = universe

    def __asset_statistics(self) -> tuple:
        """
        Calculates the statistical measures of the assets
        
        Parameters
        ----------
        time : int
            the time at which the measures are calculated
            
        Returns
        -------
        dict
            {mean_returns, volatilities, skewness, kurtosis}
        
        """
        returns = self.__asset_returns[self.__current_universe].iloc[self.__current_time - self.__hind_sight:self.__current_time].values
        
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

    def __calculte_portfolio_metrics(self):
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
        portfolio_return = (self.__current_portfolio_value / self.__initial_capital) - 1
        risk = np.std(self.__portfolio_returns) * np.sqrt(self.__current_time)
        sharpe_ratio = np.mean(self.__portfolio_returns) / risk * self.__current_time
        maximum_drawdown = np.min((self.__portfolio_values - np.maximum.accumulate(self.__portfolio_values)) / np.maximum.accumulate(self.__portfolio_values))
        value_at_risk = np.percentile(self.__portfolio_returns, 5)

        return (portfolio_return, risk, sharpe_ratio, maximum_drawdown, value_at_risk)
    
    def __render_terminal(self):
        """
        Renders the environment in the terminal
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        portfolio_return, risk, sharpe_ratio, maximum_drawdown, value_at_risk = self.__calculte_portfolio_metrics()
        
        print("=" * 50)
        print("Current time: {}".format(self.__current_time))
        print("Assets in the universe: {}".format(self.__current_universe))
        print("Current portfolio value: {}".format(self.__current_portfolio_value))
        print("Current weights: {}".format(self.__current_weights))
        print("Return\t: {}".format(portfolio_return))
        print("Risk\t: {}".format(risk))
        print("Sharpe\t: {}".format(sharpe_ratio))
        print("Max DD\t: {}".format(maximum_drawdown))
        print("VaR\t: {}".format(value_at_risk))
        print("")

    def __render_plot(self):
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
        plt.plot(self.__portfolio_values)
        plt.title("Portfolio Value")
        plt.xlabel("Time (days)")
        plt.ylabel("Value ($)")
        plt.show()

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
        stats = self.__asset_statistics()
        state = np.concatenate((self.__current_weights, stats['mean_returns'], stats['volatilities'], stats['skewness'], stats['kurtosis']))
        return state
    
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

        self.__current_time += 1
        action /= np.sum(action)
        
        # calculate the returns at time t
        days_returns = self.__asset_returns[self.__current_universe].iloc[self.__current_time].values
        transaction_cost = np.sum(np.abs(self.__current_weights - action)) * self.__transaction_cost
        weighted_return = np.dot(action, days_returns)

        net_return = weighted_return - transaction_cost


        # adjust weights to account for the changes in individual asset prices
        self.__current_weights = action * (days_returns + 1) / (weighted_return + 1)

        # update the portfolio values
        self.__current_portfolio_value *= (1 + net_return)
        self.__portfolio_values.append(self.__current_portfolio_value)
        self.__portfolio_returns.append(net_return)
        
        next_state = self.state()

        done = (self.__current_time == self.__maximal_time - 1)
        
        statistics = self.__asset_statistics()
        reward = net_return - 0.5 * np.dot(np.dot(action, statistics["covariance_matrix"]), action)

        info = {
            "return": round(net_return, 4),
            "transaction_cost": round(transaction_cost * self.__portfolio_values[-2], 4),
            "reward": round(reward, 4),
            "done": done
            }
        
        return (next_state, reward, done, info)

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
        
        self.__current_time = self.__hind_sight
        self.__current_portfolio_value = self.__initial_capital
        self.__current_weights = np.zeros(self.__num_assets)
        self.__portfolio_returns = [0]
        self.__portfolio_values = [self.__initial_capital]
        self.__sample_assets()
        
        return self.state()
        
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
            self.__render_terminal()

        elif mode == 'plot':
            self.__render_plot()

    def close(self):
        """
        Closes the environment

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        sys.exit(1)

    def save(self, path: str):
        """
        Saves the environment

        Parameters
        ----------
        path : str
            The path to which the environment should be saved

        Returns
        -------
        None

        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str):
        """
        Loads the environment

        Parameters
        ----------
        path : str
            The path from which the environment should be loaded

        Returns
        -------
        None

        """
        with open(path, 'rb') as f:
            return pickle.load(f)
        

# env = TradingEnvironment()
# while True:
#     action = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
#     next_state, reward, done, info = env.step(action)
#     print(info)
#     if done:
#         break

# env.render(mode='terminal')
# env.render(mode='plot')

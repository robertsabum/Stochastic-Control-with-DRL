import gym
from gym import spaces
import numpy as np

class CityEnv(gym.Env):
    """
    A simple environment of a city with N locations and a single bus that aims to serve as many passengers as 
    possible minimizingtotal waiting time from bus stop arrival to destination arrival as well as operating cost.

    TODO: 
        - Add an appropriate reward function

    Parameters
    ----------
    N : int
        Number of locations in the city

    P : int
        City population (Maximum number of passengers in the system)

    C : float
        Cost of running the bus per unit time

    T : int
        Maximum number of time steps in an episode

    bus_capacity : int
        Maximum number of passengers the bus can carry

    lambda_mean : float
        Average values for lambda (lambda being the average number of travelers between a pair of locations)

    lambda_deviation : float
        Standard deviation of lambda values

    delta_mean : float
        Average values for delta (delta being the average travel time between a pair of locations)

    delta_deviation : float
        Standard deviation of delta values

    beta_mean : float
        Average values for beta (beta being the volitility in traffic conditions)

    beta_deviation : float
        Standard deviation of beta values

    """

    def __init__(self, N=20, P=100, C=1, T=1000, bus_capacity=50, 
                 lambda_mean=0.5, lambda_deviation=0.1,
                 delta_mean=10, delta_deviation=1,
                 beta_mean=0.5, beta_deviation=0.1):
        super(CityEnv, self).__init__()

        self.N = N
        self.P = P
        self.C = C

        self.generate_demands(lambda_mean, lambda_deviation)
        self.generate_travel_times(delta_mean, delta_deviation, beta_mean, beta_deviation)

        self.bus_stops = [[] for _ in range(N)]
        self.bus_location = np.random.randint(0, N-1)
        self.bus_capacity = bus_capacity
        self.bus_passengers = []

        self.T = T
        self.t = 0
        self.total_waiting_time = 0
        
        self.served_passengers = 0

        self.generate_action_space()
        self.generate_observation_space()

    def generate_action_space(self):
        """
        Generates an action space for the environment where the agent can choose to drive
        the bus to any location

        Returns
        -------
        None

        """
        self.action_space = spaces.Discrete(self.N)

    def generate_observation_space(self):
        """
        Generates an observation space for the environment where the agent can observe
        the number of passengers waiting at each bus stop

        Returns
        -------
        None

        """
        self.observation_space = spaces.Box(low=0, high=self.P, shape=(self.N,))

    def generate_demands(self, mean_lambda, lambda_deviation):
        """
        Generates a matrix of demand between each pair of locations
        where each element (i, j) is a random number sampled from a normal distribution
        and represents the average number of passengers wishing to travel between location i and j

        Parameters
        ----------
        mean_lambda : int
            Average values for lambda (How busy is the city on average between each pair of locations?)

        lambda_deviation : int
            Standard deviation of lambda values (How irregular is the city's demand across all pairs of locations?)

        Returns
        -------
        None
        
        """

        self.demand = np.random.normal(mean_lambda, lambda_deviation, size=(self.N, self.N))

        np.fill_diagonal(self.demand, 0)

        self.demand = np.clip(self.demand, 0, None)

    def generate_travel_times(self, delta_mean, delta_deviation, beta_mean, beta_deviation):
        """
        Generates an N x N matrix element (i, j) is tuple of the form (mean, standard deviation)
        which are used to calculate the travel time between location i and j

        Parameters
        ----------
        delta_mean : int
            Average travel time between each pair of locations

        delta_deviation : int
            Standard deviation of travel times between each pair of locations

        beta_mean : int
            Average volatility in travel times between each pair of locations

        beta_deviation : int
            Standard deviation of volatility in travel times between each pair of locations

        Returns
        -------
        None

        """

        self.average_travel_times = np.random.normal(delta_mean, delta_deviation, size=(self.N, self.N))
        self.average_travel_times = np.clip(self.average_travel_times, 0, None)
        np.fill_diagonal(self.average_travel_times, 1) # Used to punish the agent for not moving

        self.travel_time_volatility = np.random.normal(beta_mean, beta_deviation, size=(self.N, self.N))
        self.travel_time_volatility = np.clip(self.travel_time_volatility, 0, None)
        np.fill_diagonal(self.travel_time_volatility, 0)


    def new_passenger(self, origin, destination):
        return {
            'origin': origin,
            'destination': destination,
            'origin_time': self.t,
        }

    def simulate_passenger_arrivals(self, duration):
        """
        Simulates the arrival of passengers at each stop

        Parameters
        ----------
        duration : int
            How long to simulate the arrival of passengers

        Returns
        -------
        None

        """
        arrivals = np.random.poisson(self.demand * duration)    # sample random number of passengers wishing to travel between each pair of locations
        for origin in range(self.N):    
            queue = [
                self.new_passenger(origin, destination)         # create a new passenger with origin and destination
                for destination in range(self.N)                # for each destination
                for _ in range(arrivals[origin, destination])   # for each passenger arriving at the origin
                ]

            np.random.shuffle(queue)                            # shuffle the queue
            self.bus_stops[origin].extend(queue)                # add the queue to the bus stop

    def simulate_travel_times(self):
        
        return np.random.normal(self.average_travel_times, self.travel_time_volatility)

    def pick_up_passengers(self):
        """
        Picks up passengers at the current bus stop

        Returns
        -------
        int
            Number of passengers picked up

        """
        picked_up = self.bus_stops[self.bus_location][:self.bus_capacity - len(self.bus_passengers)]
        self.bus_passengers.extend(picked_up)
        self.bus_stops[self.bus_location] = self.bus_stops[self.bus_location][self.bus_capacity - len(self.bus_passengers):]

        return len(picked_up)

    def drop_off_passengers(self):
        """
        Drops off passengers who planned to get off at the current bus stop

        Returns
        -------
        None

        """
        dropped_off = [passenger for passenger in self.bus_passengers if passenger['destination'] == self.bus_location]
        self.total_waiting_time += sum(self.t - passenger['origin_time'] for passenger in dropped_off)
        self.served_passengers += len(dropped_off)

        self.bus_passengers = [passenger for passenger in self.bus_passengers if passenger['destination'] != self.bus_location]

        return len(dropped_off)

    def drive(self, destination):
        """
        Drives the bus to the destination

        Parameters
        ----------
        destination : int
            The bus stop to drive to

        Returns
        -------
        None

        """
        
        travel_time = self.simulate_travel_times()[self.bus_location, destination]
        self.t += travel_time
        self.bus_location = destination

        return travel_time
    
    def state(self):
        """
        Returns the state of the environment

        Returns
        -------
        np.ndarray
            The number of passengers waiting at each bus stop

        """
        return np.array([len(self.bus_stops[i]) for i in range(self.N)])

    def reset(self):
        """
        Resets the environment

        Returns
        -------
        np.ndarray
            The number of passengers waiting at each bus stop

        """
        self.bus_location = np.random.randint(self.N-1)
        self.bus_passengers = []
        self.bus_stops = [[] for _ in range(self.N)]
        self.t = 0
        self.total_waiting_time = 0
        self.served_passengers = 0

        return self.state()

    def step(self, action):
        """
        Performs one step of the environment

        Parameters
        ----------
        action : int
            The bus stop to drive to

        Returns
        -------
        tuple
            (observation, reward, done, info)

        """
        travel_time = self.drive(action)
        drop_offs = self.drop_off_passengers()
        pick_ups = self.pick_up_passengers()

        done = self.t >= self.duration

        reward = drop_offs + pick_ups - travel_time

        return self.state(), reward, done, {}

    def render_bus_stops(self):
        for i in range(self.N):
            print(f'Bus stop {i}: {len(self.bus_stops[i])} passengers')

        print(f"total passengers: {sum(len(self.bus_stops[i]) for i in range(self.N))}")

    def render_bus(self):
        print(f'Bus location: {self.bus_location}')
        print(f'Bus passengers: {len(self.bus_passengers)}')

        
 
        
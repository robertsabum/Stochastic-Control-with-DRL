import sys
import gym
from gym import spaces
import numpy as np
from numpy.random import Generator, PCG64


class CityEnv(gym.Env):
    """
    A simple environment of a city with N locations and a single bus that aims to serve as many passengers as 
    possible minimizing total waiting time from bus stop arrival to destination arrival as well as operating cost.
    """

    def __init__(
        self,
        num_locations: int = 10,
        population: int = 100,
        bus_cost: float = 1,
        max_time_steps: int = 1000,
        bus_capacity: int = 50,
        lambda_mean: float = 0.25,      # Lambda being the average number of passengers wishing to travel between a pair of locations per time step
        lambda_deviation: float = 0.2,
        delta_mean: float = 10,         # Delta being the average travel time between a pair of locations
        delta_deviation: float = 5,
        beta_mean: float = 0.5,         # Beta being the volatility in travel time between a pair of locations
        beta_deviation: float = 0.1
    ):
        super(CityEnv, self).__init__()

        np.set_printoptions(suppress=True)

        self.num_locations = num_locations
        self.population = population
        self.passengers_in_transit = 0
        self.bus_cost = bus_cost

        self._generator = Generator(PCG64())

        self._generate_demands(lambda_mean, lambda_deviation)
        self._generate_travel_times(
            delta_mean, delta_deviation, 
            beta_mean, beta_deviation
        )

        self.bus_stops = [[] for _ in range(num_locations)]
        self.bus_location = self._generator.integers(num_locations - 1)
        self.bus_capacity = bus_capacity
        self.bus_passengers = []

        self.max_time_steps = max_time_steps
        self.current_time_step = 0
        self.current_episode = 0
        self.total_waiting_time = 0

        self.served_passengers = 0

        self._generate_action_space()
        self._generate_observation_space()

        self.reset()

    def _generate_action_space(self) -> None:
        """
        Generates an action space for the environment where the agent can choose to drive
        the bus to any location
        """
        self.action_space = spaces.Discrete(self.num_locations)

    def _generate_observation_space(self) -> None:
        """
        Generates an observation space for the environment where the agent can observe
        the number of passengers waiting at each bus stop
        """
        self.observation_space = spaces.Box(low=0, high=self.population, shape=(self.num_locations,))

    def _generate_demands(self, lambda_mean: float, lambda_deviation: float) -> None:
        """
        Generates a matrix of demand between each pair of locations
        where each element (i, j) is a random number sampled from a normal distribution
        and represents the average number of passengers wishing to travel between location i and j
        per time step
        """
        self.demand = self._generator.normal(lambda_mean, lambda_deviation, size=(self.num_locations, self.num_locations))
        np.fill_diagonal(self.demand, 0)
        self.demand = np.clip(self.demand, 0.01, self.population)

    def _generate_travel_times(self, delta_mean: float, delta_deviation: float, 
                               beta_mean: float, beta_deviation: float) -> None:
        """
        Generates a matrix of travel times between each pair of locations
        where each element (i, j) is a random number sampled from a normal distribution
        and represents the average travel time between location i and j
        """
        self.average_travel_times = self._generator.normal(delta_mean, delta_deviation, size=(self.num_locations, self.num_locations))
        self.average_travel_times = np.fill_diagonal(self.average_travel_times, 0)
        self.average_travel_times = np.clip(self.average_travel_times, 1, np.inf)

        self.traffic_volatility = self._generator.normal(beta_mean, beta_deviation, size=(self.num_locations, self.num_locations))
        self.traffic_volatility = np.fill_diagonal(self.traffic_volatility, 0)
        self.traffic_volatility = np.clip(self.traffic_volatility, 0, np.inf)

    def _new_passenger(self, origin: int, destination: int) -> dict:
        """
        Creates a new passenger with an origin and destination
        """
        return {
            'origin': origin,
            'destination': destination,
            'origin_time': self.current_time_step,
        }

    def _simulate_passenger_arrivals(self, duration: int) -> None:
        """
        Simulates passenger arrivals at each bus stop
        """

        arrivals = self._generator.poisson(self.demand * duration)

        for origin in range(self.num_locations):    
            queue = [
                self._new_passenger(origin, destination)     
                for destination in range(self.num_locations) 
                for _ in range(arrivals[origin, destination])
                ]

            np.random.shuffle(queue)                         
            self.bus_stops[origin].extend(queue)             

    def _simulate_travel_times(self) -> np.ndarray:

        travel_times = self._generator.normal(self.average_travel_times, self.traffic_volatility)
        np.clip(travel_times, 1, self.max_time_steps)

        return travel_times
    
    def _pick_up_passengers(self) -> int:
        """
        Picks up passengers waiting at the bus stop and adds them to the bus
        """
        picked_up = self.bus_stops[self.bus_location][:self.bus_capacity - len(self.bus_passengers)]
        self.bus_passengers.extend(picked_up)
        self.bus_stops[self.bus_location] = self.bus_stops[self.bus_location][self.bus_capacity - len(self.bus_passengers):]

        return len(picked_up)

    def _drop_off_passengers(self) -> int:
        """
        Drops off passengers at their destination and removes them from the bus
        """
        dropped_off = [passenger for passenger in self.bus_passengers if passenger['destination'] == self.bus_location]
        self.total_waiting_time += sum(self.t - passenger['origin_time'] for passenger in dropped_off)
        self.served_passengers += len(dropped_off)
        
        self.bus_passengers = [passenger for passenger in self.bus_passengers if passenger['destination'] != self.bus_location]

        return len(dropped_off)

    def _drive_bus_to(self, destination: int) -> int:
        """
        Drives the bus to the destination
        """
        travel_time = self._simulate_travel_times()[self.bus_location, destination]
        self.bus_location = destination

        return travel_time

    def state(self) -> np.ndarray:
        """
        Returns the state of the environment
        """
        return np.array([len(bus_stop) for bus_stop in self.bus_stops])
    
    def _calculate_state_value(self) -> float:
        """
        Calculates the value of the state looking at every passenger on the 
        bus and every passenger waiting at each bus stop and calculating on average
        how long has everybody waited
        """
        total_waiting_time = 0
        total_passengers = 0

        for bus_stop in self.bus_stops:
            total_waiting_time += sum(self.t - passenger['origin_time'] for passenger in bus_stop)
            total_passengers += len(bus_stop)

        for passenger in self.bus_passengers:
            total_waiting_time += self.t - passenger['origin_time']
            total_passengers += 1

        return -1 * (total_waiting_time / total_passengers)

    def simulate_passing_time(self, duration: int = 1) -> None:
        """
        Simulates the passage of time in the environment
        """
        self._simulate_passenger_arrivals(duration)
        self.t += duration
    
    def step(self, action):
        """
        The environment takes a step using the action taken by the agent. The action represents the location of the bus
        that the agent wishes to drive to. The environment then simulates the passage of time and returns the next
        
        Parameters
        ----------
        action : int
            The action taken by the agent, represents the location of the bus

        Returns
        -------
        observation : numpy.ndarray
            Observation of the environment after the action has been taken by the agent

        reward : float
            Reward obtained by the agent after taking the action

        done : bool
            Whether the episode has ended

        info : dict
            Information useful for debugging/evaluation purposes

        """

        # Drive the bus to the action location
        travel_time = self._drive_bus_to(action)
        dropped_off = self._drop_off_passengers()
        picked_up = self._pick_up_passengers()
        self._simulate_passenger_arrivals(travel_time)
        self.current_time_step += travel_time
        self.served_passengers += dropped_off

        # Check if the episode has ended
        done = self.current_time_step >= self.max_time_steps

        # Generate observation
        observation = self.state()

        # Generate reward
        reward = self._calculate_state_value()

        # Generate info dictionary
        info = {
            'stopped at': action,
            'passengers_dropped_off': self.served_passengers,
            'passengers_picked_up': picked_up,
            'total_passengers_serviced': self.served_passengers,
            'average_waiting_time': self.total_waiting_time / self.served_passengers
            }

        return observation, reward, done, info

    def reset(self):
        """
        Resets the environment to its initial state

        Returns
        -------
        observation : numpy.ndarray
            The initial observation of the environment
        """

        self.bus_stops = [[] for _ in range(self.num_locations)]
        self.bus_location = np.random.randint(0, self.num_locations-1)
        self.bus_passengers = []
        self.total_waiting_time = 0
        self.t = 0
        self.served_passengers = 0

        return self.state()

    def render(self):
        """
        Renders the environment

        Parameters
        ----------
        mode : str
            The mode in which to render the environment
        """
        
        print(f'current episode: {self.current_episode}')
        print(f'Current time step: {self.current_time_step}')
        print(f'Bus location: {self.bus_location}')
        print(f'Bus passengers: {len(self.bus_passengers)}')
        print("passengers waiting at each bus stop:")
        for i, bus_stop in enumerate(self.bus_stops):
            print(f'Bus stop {i}: {len(bus_stop)}')

    def close(self):
        """
        Closes the environment
        """
        sys.exit(1)

env = CityEnv()
env.simulate_passing_time(60)
env.render()

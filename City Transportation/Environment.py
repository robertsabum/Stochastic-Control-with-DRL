import pickle
import random
import sys
import gym
from gym import spaces
import numpy as np
from numpy.random import Generator, PCG64


class CityEnv(gym.Env):
    """
    A simple environment of a city with N locations and a single bus that aims to serve as many passengers as 
    possible minimizing average waiting time from bus stop arrival to destination arrival.

    """

    def __init__(
        self,
        num_locations: int = 10,
        population: int = 1000,
        maximal_time: int = 1440,
        bus_capacity: int = 50,
        lambda_mean: float = 0.00005,       # Lambda -> Average number of passengers wishing to travel between a pair of locations (per time step per person)
        lambda_deviation: float = 0.0002,
        delta_mean: float = 10,             # Delta  -> Average travel time between a pair of locations
        delta_deviation: float = 5,
        beta_mean: float = 0.5,             # Beta   -> Volatility in travel time between a pair of locations
        beta_deviation: float = 0.1
    ):
        super(CityEnv, self).__init__()

        np.set_printoptions(suppress=True)

        self.__num_locations = num_locations
        self.__population = population
        self.__passengers_in_transit = 0

        self.__rng = Generator(PCG64())

        self.__generate_demand_parameters(lambda_mean, lambda_deviation)
        self.__generate_travel_time_parameters(delta_mean, delta_deviation, beta_mean, beta_deviation)

        self.__bus_stops = [[] for _ in range(num_locations)]
        self.__bus_location = self.__rng.integers(num_locations - 1)
        self.__bus_capacity = bus_capacity
        self.__bus_passengers = []
        self.__current_demand = np.zeros((num_locations, num_locations))

        self.__maximal_time = maximal_time
        self.__current_time = 0
        self.__total_waiting_time = 0

        self.__served_passengers = 0

        self.__generate_action_space()
        self.__generate_observation_space()

        self.reset()

    def __generate_action_space(self) -> None:
        """
        Generates an action space for the environment where the agent can choose to drive
        the bus to any location

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.action_space = spaces.Discrete(self.__num_locations)

    def __generate_observation_space(self) -> None:
        """
        Generates an observation space for the environment where the agent can observe
        the number of passengers waiting at each bus stop

        Parameters
        ----------
        None

        Returns
        -------
        None
        
        """
        self.observation_space = spaces.Box(low=0, high=self.__population, shape=(self.__num_locations**2,))

    def __generate_demand_parameters(self, lambda_mean: float, lambda_deviation: float) -> None:
        """
        Generates a matrix of demand between each pair of locations
        where each element (i, j) is a random number sampled from a normal distribution
        and represents the average number of passengers wishing to travel between location i and j
        per time step

        Parameters
        ----------
        lambda_mean : float
            The average lambda value (lambda -> average number of passengers)

        lambda_deviation : float
            The standard deviation of the lambda values

        Returns
        -------
        None
        
        """
        self.demand_matrix = self.__rng.normal(lambda_mean, lambda_deviation, size=(self.__num_locations, self.__num_locations)) * self.__population
        np.fill_diagonal(self.demand_matrix, 0)
        self.demand_matrix = np.clip(self.demand_matrix, 0.01, self.__population) 

    def __generate_travel_time_parameters(self, delta_mean: float, delta_deviation: float, 
                               beta_mean: float, beta_deviation: float) -> None:
        """
        Generates two N x N matrices of average travel times and travel time deviations
        respectively between each pair of locations where each element (i, j) is a random 
        number sampled from a normal distribution.

        Parameters
        ----------
        delta_mean : float
            The average delta value (delta -> average travel time)

        delta_deviation : float
            The standard deviation of the delta values

        beta_mean : float
            The average beta value (beta -> travel time volatility)

        beta_deviation : float
            The standard deviation of the beta values

        Returns
        -------
        None
        
        """
        self.__average_travel_times = self.__rng.normal(delta_mean, delta_deviation, size=(self.__num_locations, self.__num_locations))
        self.__average_travel_times = np.clip(self.__average_travel_times, 1, None)
        np.fill_diagonal(self.__average_travel_times, 0)

        self.__traffic_volatility = self.__rng.normal(beta_mean, beta_deviation, size=(self.__num_locations, self.__num_locations))
        self.__traffic_volatility = np.clip(self.__traffic_volatility, 0.01, None)
        np.fill_diagonal(self.__traffic_volatility, 0)

    def __new_passenger(self, origin: int, destination: int) -> dict:
        """
        Creates a new passenger with an origin and destination

        Parameters
        ----------
        origin : int
            The station from which the passenger starts

        destination : int
            The station to which the passenger is going

        Returns
        -------
        dict
            A dictionary containing the passenger's origin, destination and origin time
        
        """
        return {
            'origin': origin,
            'destination': destination,
            'origin_time': self.__current_time,
        }

    def __simulate_passenger_arrivals(self, duration: float) -> None:
        """
        Simulates passenger arrivals at each bus stop

        Parameters
        ----------
        duration : float
            For how long the simulation should run

        Returns
        -------
        None
        
        """

        scale_factor = max(0, (1 - (self.__passengers_in_transit / self.__population)))
        arrivals = self.__rng.poisson(self.demand_matrix * duration * scale_factor)

        for origin in range(self.__num_locations):    
            queue = [
                self.__new_passenger(origin, destination)
                for destination in range(self.__num_locations)
                for _ in range(arrivals[origin, destination])
                ]

            np.random.shuffle(queue)
            self.__bus_stops[origin].extend(queue)
        
        self.__current_demand += arrivals
        self.__passengers_in_transit += sum(sum(arrivals))

    def __simulate_travel_time(self, origin: int, destination: int) -> float:
        """
        Simulates travel times between each pair of locations

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            A matrix of travel times between each pair of locations

        """

        travel_time = self.__rng.normal(self.__average_travel_times[origin, destination], self.__traffic_volatility[origin, destination])
        travel_time = max(1, travel_time)

        return travel_time
    
    def __simulate_passing_time(self, duration: float = 1) -> None:
        """
        Simulates the passage of time in the environment

        Parameters
        ----------
        duration : float
            For how long the simulation should run

        Returns
        -------
        None
        
        """
        self.__simulate_passenger_arrivals(duration)
        self.__current_time += duration
    
    def __pick_up_passengers(self) -> int:
        """
        Picks up passengers waiting at the bus stop and adds them to the bus

        Parameters
        ----------
        None

        Returns
        -------
        int
            The number of passengers picked up
        
        """
        num_picked_up = 0
        while len(self.__bus_passengers) < self.__bus_capacity:
            try:
                self.__bus_passengers.append(self.__bus_stops[self.__bus_location].pop(0))
                num_picked_up += 1

            except IndexError:
                break
    
        return num_picked_up

    def __drop_off_passengers(self) -> int:
        """
        Drops off passengers at their destination and removes them from the bus

        Parameters
        ----------
        None

        Returns
        -------
        int
            The number of passengers dropped off
        
        """
        num_dropped_off = 0
        for passenger in self.__bus_passengers:
            if passenger['destination'] == self.__bus_location:
                self.__total_waiting_time += self.__current_time - passenger['origin_time']
                self.__bus_passengers.remove(passenger)
                num_dropped_off += 1
                self.__served_passengers += 1
                self.__current_demand[passenger['origin'], passenger['destination']] -= 1

        return num_dropped_off

    def __drive_bus_to(self, destination: int) -> float:
        """
        Drives the bus to the destination

        Parameters
        ----------
        destination : int
            The station to which the bus should drive

        Returns
        -------
        float
            The time it took to drive to the destination
        
        """
        travel_time = self.__simulate_travel_time(self.__bus_location, destination)
        self.__simulate_passing_time(travel_time)
        self.__bus_location = destination

        return travel_time
    
    def _calculate_state_value(self) -> float:
        """
        Calculates the value of the state looking at every passenger on the 
        bus and every passenger waiting at each bus stop and calculating on average
        how long has everybody waited

        Parameters
        ----------
        None

        Returns
        -------
        float
            The value of the state (average waiting time of every passenger both waiting and on the bus)
        
        """
        total_waiting_time = 0

        for bus_stop in self.__bus_stops:
            total_waiting_time += sum(self.__current_time - passenger['origin_time'] for passenger in bus_stop)

        for passenger in self.__bus_passengers:
            total_waiting_time += self.__current_time - passenger['origin_time']

        return -1 * (total_waiting_time / self.__passengers_in_transit)

    def state(self) -> np.ndarray:
        """
        Returns the state of the environment

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            The state of the environment (number of passengers waiting at each bus stop, 
            number of passengers on the bus, approximate travel times between each pair of locations)
        
        """
        

        passengers_waiting = self.__current_demand.flatten()
        passengers_on_bus = np.zeros(self.__num_locations)
        for passenger in self.__bus_passengers:
            passengers_on_bus[passenger['destination']] += 1
        approximate_travel_times = self.__rng.normal(self.__average_travel_times[self.__bus_location], self.__traffic_volatility[self.__bus_location])

        state = np.concatenate((passengers_waiting, passengers_on_bus, approximate_travel_times))
        
        return state
    
    def step(self, action):
        """
        The environment takes a step using the action taken by the agent. The action represents the location of the bus
        that the agent wishes to drive to. The environment then simulates the passage of time and returns the next
        
        Parameters
        ----------
        action : int
            The action taken by the agent, represents the location to which the agent wishes to drive

        Returns
        -------
        observation : numpy.ndarray
            Observation of the environment after the action has been taken by the agent

        reward : float
            The value of the resulting state

        done : bool
            Whether the episode has ended

        info : dict
            Information useful for debugging/evaluation purposes

        """

        assert self.action_space.contains(action), f'{action} ({type(action)}) invalid'

        # Drive the bus to the desired location
        self.__drive_bus_to(action)
        num_dropped_off = self.__drop_off_passengers()
        num_picked_up = self.__pick_up_passengers()

        # Check if the episode has ended
        done = self.__current_time >= self.__maximal_time

        # Generate observation
        observation = self.state()

        # Generate reward
        reward = num_dropped_off

        # Generate info dictionary
        info = {
            'current_time': self.__current_time,
            'stopped at': action,
            'passengers_dropped_off': num_dropped_off,
            'passengers_picked_up': num_picked_up,
            'total_passengers_serviced': self.__served_passengers
            }

        return observation, reward, done, info

    def reset(self):
        """
        Resets the environment to its initial state

        Parameters
        ----------
        None

        Returns
        -------
        observation : numpy.ndarray
            The initial observation of the environment
        """

        self.__bus_stops = [[] for _ in range(self.__num_locations)]
        self.__bus_location = np.random.randint(0, self.__num_locations-1)
        self.__bus_passengers = []
        self.__total_waiting_time = 0
        self.__current_time = 0
        self.__served_passengers = 0
        self.__passengers_in_transit = 0
        
        return self.state()

    def render(self):
        """
        Renders the environment

        Parameters
        ----------
        mode : str
            The mode in which to render the environment

        Returns
        -------
        None

        """
        
        print(f'Current time: {self.__current_time}')
        print(f'Bus location: {self.__bus_location}')
        print(f'Bus passengers: {len(self.__bus_passengers)}')
        print("passengers waiting at each bus stop:")
        for i, bus_stop in enumerate(self.__bus_stops):
            print(f'Bus stop {i}: {len(bus_stop)}')

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

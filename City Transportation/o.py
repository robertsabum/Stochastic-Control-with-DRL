
import gym
from gym import spaces
import numpy as np

class TravelTimeNetwork():
    """
    A graph representing the travel time between N locations. 
    Mean and standard deviation of the travel time are randomly 
    generated uniformly for each pair of locations.

    Parameters
    ----------
    N : int
        Number of locations in the city

    min_average : int
        Lower bound of the average travel time between two locations (Time between two closest locations)

    max_average : int
        Upper bound of the average travel time between two locations (Time between two furthest locations)

    min_std : int
        Lower bound of the standard deviation of the travel time between two locations (Most regular travel time)

    max_std : int
        Upper bound of the standard deviation of the travel time between two locations (Most irregular travel time)

    """
    def __init__(self, N, min_average=5, max_average=30, min_std=1, max_std=5):
        self.N = N

        self.travel_time_mean = np.random.randint(min_average, max_average, size=(N, N))
        self.travel_time_std = np.random.randint(min_std, max_std, size=(N, N))

    def get_travel_time_matrix(self):
        """
        Returns a matrix of travel times between all pairs of locations

        Parameters
        ----------
        None

        Returns
        -------
        travel_time_matrix : np.array
            A matrix of travel times between all pairs of locations

        """

        travel_time_matrix = np.random.normal(self.travel_time_mean, self.travel_time_std)
        travel_time_matrix = travel_time_matrix.astype(int)
        return travel_time_matrix
    
class DemandNetwork():
    """
    A graph representing the number of passengers waiting for the bus
    at each stop. The average number of passengers arriving at a
    stop (lambda) is randomly generated from the normal distribution
    for each location.
    
    Parameters
    ----------
    N : int
        Number of locations in the city

    mean : float
        Average values for lambda (How generaly busy is the city across all locations?)

    standard_deviation : float
        Standard deviation of lambda values (How irregular is the city's demand across different locations?)
    
    """
    def __init__(self, N, mean=5, standard_deviation=2):

        self.N = N
        self.average_passenger_arrivals = np.random.normal(mean, standard_deviation, size=(N, N)).astype(int)
        np.fill_diagonal(self.average_passenger_arrivals, 0)
        self.waiting_passengers = np.zeros((N, N))

    def simulate_passenger_arrivals(self, time):
        """
        Simulates the arrival of passengers at each stop

        Parameters
        ----------
        time : int
            How long to simulate the arrival of passengers

        Returns
        -------
        None

        """

        self.waiting_passengers += np.random.poisson(self.average_passenger_arrivals * time)

    def serve_waiting_passengers(self, stop, destination, count):
        """
        Serves the waiting passengers at a stop

        Parameters
        ----------
        stop : int
            Stop being served

        destination : int
            Destination of the passengers being served

        count : int
            Number of passengers being served

        Returns
        -------
        None

        """
            
        self.waiting_passengers[stop][destination] -= count
        

    def get_waiting_passenger_matrix(self):
        """
        Returns the an N x N matrix where element (i, j) is the number 
        of passengers waiting to get from location i to location j

        Parameters
        ----------
        None

        Returns
        -------
        waiting_passengers : np.array
            N x N matrix of waiting passengers

        """

        return self.waiting_passengers
    
    def reset_waiting_passengers(self):
        """
        Resets the number of waiting passengers

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        self.waiting_passengers = np.zeros((self.N, self.N))

class Passenger():
    """
    A passenger riding the bus

    Parameters
    ----------
    start_time : int
        Time the passenger gets to the bus stop

    boarding_stop : int
        Stop the passenger boards the bus

    destination_stop : int
        Stop the passenger gets off the bus

    """
    def __init__(self, start_time, boarding_stop, destination_stop):
        self.start_time = start_time
        self.boarding_stop = boarding_stop
        self.destination_stop = destination_stop

    def calculate_service_time(self, arrival_time):
        return arrival_time - self.start_time

    def get_destination(self):
        return self.destination_stop
    
class Bus():
    """
    A bus in the city

    Parameters
    ----------
    start : int
        Location of the bus

    capacity : int
        Number of passengers the bus can carry

    """
    def __init__(self, starting_location=0, capacity=50):
        self.location = starting_location
        self.capacity = capacity
        self.passengers = []

    def add_passenger(self, passenger):
        self.passengers.append(passenger)

    def remove_passenger(self, passenger):
        self.passengers.remove(passenger)

    def drop_off_passengers(self, destination):
        for passenger in self.passengers:
            if passenger.destination == destination:
                self.remove_passenger(passenger)

    def pick_up_passengers(self, passengers):
        for passenger in passengers:
            if len(self.passengers) < self.capacity:
                self.add_passenger(passenger)

    def move(self, destination):
        self.location = destination

    def reset(self):
        self.location = 0
        self.passengers = []
        

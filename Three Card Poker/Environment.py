import sys
import gym
from gym import spaces
import numpy as np
from numpy.random import Generator, PCG64

class ThreeCardPokerEnv(gym.Env):
    """Simple No limit Three Card Poker environment"""

    def __init__(
            self,
            bet_increment: int = 20,
            initial_stack: int = 1000,
            max_rounds: int = 100,
            ):
        
        self.bet_increment = bet_increment
        self.initial_stack = initial_stack
        self.max_rounds = max_rounds
        
        self.rng = Generator(PCG64())
        self._set_action_space()
        self._set_observation_space()

    
    def _set_action_space(self):
        """
        Set a discrete action space for the environment
        where:
            - 0 = fold
            - 1 = call
            - 2 = raise
        
        """
        self.action_space = spaces.Discrete(3)

    def _set_observation_space(self):
        """
        A state vector containing the following:
            - player_one stack
            - player_two stack
            - player_one bet
            - player_two bet
            - pot
            - player_one hand
            - player_two hand
            - Community cards

        """
        max_chips = self.initial_stack * 2
        player_one_stack = spaces.Discrete(max_chips + 1)
        player_two_stack = spaces.Discrete(max_chips + 1)
        player_one_bet = spaces.Discrete(max_chips + 1)
        player_two_bet = spaces.Discrete(max_chips + 1)
        pot = spaces.Discrete(max_chips + 1)
        player_one_hand = spaces.Discrete(53)
        player_two_hand = spaces.Discrete(53)
        community_card_1 = spaces.Discrete(53)
        community_card_2 = spaces.Discrete(53)
        community_card_3 = spaces.Discrete(53)

        self.observation_space = spaces.Tuple((
            player_one_stack,
            player_two_stack,
            player_one_bet,
            player_two_bet,
            pot,
            player_one_hand,
            player_two_hand,
            community_card_1,
            community_card_2,
            community_card_3,
        ))
        
    def reset(self):
        """
        Reset the environment to the initial state
        """
        
        self.pot = 0
        self.player_one = {'stack': self.initial_stack, 'hand': [0, 0], 'bet': 0}
        self.player_two = {'stack': self.initial_stack, 'hand': [0, 0], 'bet': 0}
        self.community_cards = [0, 0, 0]
        self.round = 0
        self.turn = 0
from Card import Card
import random

class Deck:
    def __init__(self):
        self.cards = []
        for suit in ["Clubs", "Diamonds", "Hearts", "Spades"]:
            for rank in range(1, 14):
                self.cards.append(Card(rank, suit))
        self.__shuffle()

    def __len__(self):
        return len(self.cards)
    
    def __str__(self):
        s = ""
        for i in range(len(self.cards)):
            s += str(self.cards[i]) + "\n"
        return s
    
    def __shuffle(self):
        random.shuffle(self.cards)
    
    def deal(self, n=1):
        return self.cards.pop(n)

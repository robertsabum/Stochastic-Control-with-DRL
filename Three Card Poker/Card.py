class Card:
    def __init__(self, rank, suit):
        self.__rank = rank
        self.__suit = suit
        self.__id = self.__rank + {"Clubs": 0, "Diamonds": 2, "Hearts": 1, "Spades": 3}[suit] * 13

    def __str__(self):
        if self.__rank == 11:
            rank = "Jack"
        elif self.__rank == 12:
            rank = "Queen"
        elif self.__rank == 13:
            rank = "King"
        elif self.__rank == 1:
            rank = "Ace"
        else:
            rank = str(self.__rank)
        return rank + " of " + self.__suit

    def __eq__(self, other):
        return self.__rank == other.rank

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.__rank < other.rank

    def __le__(self, other):

        if self.__rank == 1:
            return False
        elif other.rank == 1:
            return True
        else:
            return self.__rank <= other.rank

    def __gt__(self, other):
        return not self.__le__(other) and self.__ne__(other)

    def __ge__(self, other):
        return self.__rank >= other.rank

    @property
    def rank(self):
        return self.__rank

    @property
    def suit(self):
        return self.__suit
    
    @property
    def id(self):
        return self.__id
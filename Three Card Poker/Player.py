class Player:
    def __init__(self, name: str, chips: int, id: str):
        self.__name = name
        self.__chips = chips
        self.__hand = []
        self.__bet = 0
        self.__folded = False
        self.__all_in = False

    def __str__(self):
        info = self.__name + " has " + str(self.__chips) + " chips and " + str(self.__bet) + " in the pot"
        if self.__folded:
            info += " and has folded"
        if self.__all_in:
            info += " and is all in"
        return info
    
    @property
    def hand(self):
        return self.__hand

    @property
    def bet(self):
        return self.__bet

    @property
    def chips(self):
        return self.__chips

    @property
    def name(self):
        return self.__name

    @property
    def folded(self):
        return self.__folded

    @property
    def allIn(self):
        return self.__all_in

    def Fold(self):
        print(self.__name, "folds")
        self.__folded = True

    def Check(self):
        print(self.__name, "checks")

    def Call(self, pot):
        print(self.__name, "calls", amount)
        amount = pot - self.__bet
        self.__bet += amount
        self.__chips -= amount
        return amount

    def Bet(self, amount):
        print(self.__name, "bets", amount)
        self.__bet += amount
        self.__chips -= amount
        return amount

    def Raise(self, raise_amount, last_bet):
        print(self.__name, "raises by", raise_amount)
        amount = last_bet - self.__bet + raise_amount
        self.__bet += amount
        self.__chips -= amount
        return amount

    def GoAllIn(self):
        print(self.__name, "goes all in")
        self.__all_in = True
        return self.__bet(self.__chips)

    def reset(self):
        self.__hand = []
        self.__bet = 0
        self.__folded = False
        self.__all_in = False

    def addChips(self, amount):
        self.__chips += amount

    def addCard(self, card):
        self.__hand.append(card)
    
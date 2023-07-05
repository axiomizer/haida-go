class HyperParameters:
    def __init__(self, hardware):
        if hardware == 0:  # lappy
            self.batch_size = 20
            self.episodes = 250
            self.games_saved = self.episodes*20
            self.simulations = 16
            self.steps_per_checkpoint = 1000
        elif hardware == 1:  # original
            self.batch_size = 2048
            self.episodes = 25000
            self.games_saved = self.episodes*20
            self.simulations = 1600
            self.steps_per_checkpoint = 1000
        else:
            raise ValueError("Invalid hardware type")


hp = HyperParameters(0)

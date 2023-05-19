
class Battery():

    """
        Implementation of the energy storage.
    """

    def __init__(self, initial_energy):
        self.current_energy = initial_energy

    def remove_energy(self, energy):
        """ Removes energy from battery """
        self.current_energy = self.current_energy - energy

        if self.current_energy < 0:
            self.current_energy = 0

    def reset_battery(self, initial_energy): #Joule
        """ Resets current_energy to intial value """
        self.current_energy = initial_energy
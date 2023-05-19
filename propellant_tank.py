
class Propellant_Tank():

    """
        Implementation of the water storage tank.
    """

    def __init__(self, initial_mass):
        self.current_mass = initial_mass

    def remove_mass(self, mass):
        """ Removes mass from water tank """
        self.current_mass = self.current_mass - mass

        if self.current_mass < 0:
            self.current_mass = 0

    def reset_propellant_tank(self, initial_mass = 0.2):
        """ Resets current_mass to intial value """
        self.current_mass = initial_mass
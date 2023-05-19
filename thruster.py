
import numpy as np

class Thruster():


    def __init__(self):
        pass


    def get_thrust(self, propellant_used, t_burn, g0 = 9.81):

        """
        Computes Thrust as a function of burn duration.

        :param mass_flow: Mass flow in kg/s of thruster.
        :param t_burn: Burn duration in seconds [s], i.e. How long the thruster is ON.
        :param g0: gravity acceleration in m/s^2

        """

        Isp = 350 #self.get_isp(t_burn)
        if t_burn == 0:
            mass_flow = 0
        else:
            mass_flow = propellant_used / t_burn

        return Isp * g0 * mass_flow


    def get_delta_v(self, T, dt, mass_spacecraft):
        dv = T / self.mass_flow * np.log(mass_spacecraft / (mass_spacecraft - self.mass_flow * dt))
        return dv

    

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    th = Thruster()

    t = np.linspace(1, 10, 50)

    Isp = th.get_isp(t)
    thrust = th.get_thrust(5e-4, t)

    plt.plot(t, Isp)
    plt.show()

    plt.plot(t, thrust)
    plt.show()


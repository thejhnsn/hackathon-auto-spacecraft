import numpy as np
import matplotlib.pyplot as plt

# General things
from astropy import units as u

from poliastro.bodies import Earth, Mars, Sun, Neptune
from poliastro.twobody import Orbit

# For propagation
from poliastro.core.propagation import func_twobody
from poliastro.twobody.propagation import CowellPropagator

from astropy.coordinates import (
    CartesianRepresentation,
    get_body_barycentric_posvel,
)

class Orbit_Propagator():

    RTOL = 1e-5

    def __init__(self, com_orb, obs_orb):
        self.acceleration_com = None
        self.positions_com    = None
        self.orb_com          = None

        self.acceleration_obs = None
        self.positions_obs    = None
        self.orb_obs          = None

        self.init_orb_com = Orbit.from_classical(Neptune, com_orb['a'] << u.km, com_orb['e'] << u.one, com_orb['i'] << u.deg, 
                                            com_orb['raan'] << u.deg , com_orb['argp'] << u.deg, com_orb['nu'] * np.pi / 180 << u.rad)

        self.init_orb_obs = Orbit.from_classical(Neptune, obs_orb['a'] << u.km, obs_orb['e'] << u.one, obs_orb['i'] << u.deg, 
                                            obs_orb['raan'] << u.deg , obs_orb['argp'] << u.deg, obs_orb['nu'] << u.deg)

    def reset_orbits(self):
        """ 
        Resets current orbit to initial orbit 
            Generally run at the beginning of a new episode
        """
        self.orb_com = self.init_orb_com
        self.positions_com = np.array([self._get_xy_coordinates_com()])

        self.orb_obs = self.init_orb_obs
        self.positions_obs = np.array([self._get_xy_coordinates_obs()])


    def propagate_orbits(self, dt, t_burn):

        """ Propagates the orbit over time taking into account if enough mass is currently available

        :param dt: timestep size
        :param t_burn: Thruster burn duration
        """

        assert self.orb_obs is not None, "Call reset orbit before propagating"
        
        if t_burn == 0: # Thruster OFF
            self.acceleration_com = 0
            self.orb_com = self.orb_com.propagate(dt << u.s, method = CowellPropagator(f = self.f_com, rtol = 1e-8))
            self._append_new_position_com()
            # print('No thrust')
            self.acceleration_obs = 0
            self.orb_obs = self.orb_obs.propagate(dt << u.s, method = CowellPropagator(f = self.f_obs, rtol = 1e-8))
            self._append_new_position_obs()
            return
        
        if t_burn < dt: # Not enough mass to turn thruster for full dt seconds
            self.orb_com = self.orb_com.propagate(t_burn << u.s, method = CowellPropagator(f = self.f_com, rtol = self.RTOL))
            self.acceleration_com = 0
            self.orb_com = self.orb_com.propagate(dt - t_burn << u.s, method = CowellPropagator(f = self.f_com, rtol = self.RTOL))
            self._append_new_position_com()

            self.acceleration_obs = 0
            self.orb_obs = self.orb_obs.propagate(dt << u.s, method = CowellPropagator(f = self.f_obs, rtol = 1e-8))
            self._append_new_position_obs()
            return
        
        '''
        Here should never happen?
        self.orb_com = self.orb_com.propagate(t_burn << u.s, method = CowellPropagator(f = self.f, rtol = 1e-8))
        self._append_new_position_com()
        '''
        return


    def get_orbit_points(self, orbit):
        """ 
        Returns a numpy array with 100 points of the orbit. Useful for rendering initial and final orbit.
        """
        ephem = orbit.to_ephem()
        X = ephem.sample(ephem.epochs[:100]).x.value
        Y = ephem.sample(ephem.epochs[:100]).y.value
        XY = np.column_stack((X,Y))
        return XY    


    def _get_xy_coordinates_obs(self):
        """
        Returns the X and Y coordinate of the current position of the spacecraft. Useful to render the orbits
        """
        return self.orb_obs.r[0:2]

    def _append_new_position_obs(self):
        """ 
        Stores current position (X,Y) of the spacecraft. Useful to render the orbits
        """
        self.positions_obs = np.append(self.positions_obs, [self._get_xy_coordinates_obs()], axis = 0)
        return
    
    def _get_xy_coordinates_com(self):
        """
        Returns the X and Y coordinate of the current position of the spacecraft. Useful to render the orbits
        """
        return self.orb_com.r[0:2]

    def _append_new_position_com(self):
        """ 
        Stores current position (X,Y) of the spacecraft. Useful to render the orbits
        """
        self.positions_com = np.append(self.positions_com, [self._get_xy_coordinates_com()], axis = 0)
        return

    def compute_distance_between_spacecraft(self):
        return np.linalg.norm(self.orb_obs.r.value - self.orb_com.r.value)

    def compute_distance_to_final_orbit(self):

        """ Used to compute Rewards """
        # nu = self.orb.nu.value

        # if nu < 0:
        #     nu = np.pi + nu

        orb_final = Orbit.from_classical(Neptune, self.final_orb.a, self.final_orb.ecc, self.final_orb.inc, 
                                            self.final_orb.raan , self.final_orb.argp, self.orb.argp + self.orb.nu)

        r_current = self.orb.r.value
        # print(r_current)
        # print(self.orb.nu, nu)
        # print(self.orb.argp)
        # print(self.orb.argp + self.orb.nu)
        r_final   = orb_final.r.value
        # print(r_final)
        # print(r_current)
        diff = r_current - r_final
        # print(diff)
        return np.sqrt(diff[0]**2 + diff[1]**2)

    def orbit_compare(self):
        orb_final = self.init_orb_com
        '''Orbit.from_classical(Neptune, self.com_orb.a, self.final_orb.ecc, self.final_orb.inc, 
                                            self.final_orb.raan , self.final_orb.argp, self.orb.argp + self.orb.nu)
        '''
        r = orb_final.r.value[0:2]

        return r

        
    # ----- Functions for Orbit Propagation ------   
    def f_com(self, t0, u_, k):

        du_kep = func_twobody(t0, u_, k)

        ax, ay, az = self.accel_com(t0, u_, k)

        du_ad = np.array([0, 0, 0, ax, ay, az])

        return du_kep + du_ad

    def accel_com(self, t0, state, k):

        """Constant acceleration aligned with the velocity. """

        v_vec = state[3:]

        norm_v = (v_vec[0]**2 + v_vec[1]**2 + v_vec[2]**2)** 0.5

        return -self.acceleration_com * (1e-3) * v_vec / norm_v

    # --------------------------------------------
    # ----- Functions for Orbit Propagation ------   
    def f_obs(self, t0, u_, k):

        du_kep = func_twobody(t0, u_, k)

        ax, ay, az = self.accel_obs(t0, u_, k)

        du_ad = np.array([0, 0, 0, ax, ay, az])

        return du_kep + du_ad

    def accel_obs(self, t0, state, k):

        """Constant acceleration aligned with the velocity. """

        v_vec = state[3:]

        norm_v = (v_vec[0]**2 + v_vec[1]**2 + v_vec[2]**2)** 0.5

        return -self.acceleration_obs * (1e-3) * v_vec / norm_v

    # --------------------------------------------



if __name__ == "__main__":

    import matplotlib.pyplot as plt
        # -------------- SEMIMAJOR AXIS         ECCENTRICITY        INCLINATION (º)          RAAN (º)             ARGP (º)           TRUE ANOMALY (º) -------------
    INITIAL_ORBIT = {'a': (6378.0 + 200.0),  'e' : 0.0,           'i': 0.0,            'raan': 0.0,         'argp': 0.0,            'nu': 0.0}
    FINAL_ORBIT   = {'a': (6378.0 + 500.0),  'e' : 0.0,           'i': 0.0,            'raan': 0.0,         'argp': 0.0,            'nu': 0.0}

    orb_prop = Orbit_Propagator(INITIAL_ORBIT, FINAL_ORBIT)

    orb = orb_prop.init_orb

    print(orb.r[0])
    ephem = orb.to_ephem()
    n_points = 10
    X = ephem.sample(ephem.epochs[:n_points]).x.value
    Y = ephem.sample(ephem.epochs[:n_points]).y.value
    
    orb2 = Orbit.from_classical(Neptune, orb.a, orb.ecc, orb.inc, orb.raan, orb.argp, 65.0 << u.deg)
    print(orb.r.value, orb2.r.value)
    ephem2 = orb2.to_ephem()
    X2 = ephem.sample(ephem.epochs[:n_points]).x.value
    Y2 = ephem.sample(ephem.epochs[:n_points]).y.value
    # orb_prop.reset_orbit()
    # print(orb_prop.positions)
    # orb_prop.acceleration = 1e-2
    # for i in range(6000):
    #     orb_prop.propagate_orbit(10, 4e-3, 50)

    # print(orb_prop.positions)

    # print('--------')

    """
    xy = orb_prop.get_orbit_points(orb_prop.init_orb)
    print(xy)
    print(xy[:,0])
    """
    # plt.plot(orb_prop.positions[:,0], orb_prop.positions[:,1])
    plt.plot(X,Y)
    plt.plot(X2, Y2)
    plt.axis('scaled')
    plt.xlim([-6700, 6700])
    plt.ylim([-6700, 6700])
    plt.show()
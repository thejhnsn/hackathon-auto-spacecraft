
import numpy as np
from numpy import cos, pi, sin
import pygame

import gymnasium as gym
from gymnasium import spaces

#from environment.propellant_tank import Propellant_Tank
#from environment.orbit_propagator import Orbit_Propagator
#from environment.thruster import Thruster
#from environment.renderer import Renderer
#from environment.battery import Battery
#from environment.DataClass import Data

from propellant_tank import Propellant_Tank
from orbit_propagator import Orbit_Propagator
from thruster import Thruster
from renderer import Renderer
from battery import Battery
from DataClass import Data

class Spacecraft(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    dt = 60 # seconds

    MASS_SPACECRAFT_INITIAL  = 12
    INITIAL_ENERGY= 15000   #15000 [J]
    INITIAL_DATA_STORAGE= 60000000        #[bits]
    POWER_CONSUMPTION= 10                 #[W]
    INITIAL_PROPELLANT_MASS  = 2 # [Kg]
    R_NEPTUNE = 24622           #[km]
    MAX_STEPS = 5000

    # --------------    SEMIMAJOR AXIS        ECCENTRICITY        INCLINATION (º)          RAAN (º)             ARGP (º)           TRUE ANOMALY (º) -------------
    COM_ORBIT = {'a': (R_NEPTUNE + 1350.0),  'e' : 0.0,           'i': 0.0,            'raan': 0.0,         'argp': 0.0,            'nu': 0.0}
    OBSERVER_ORBIT = {'a': (R_NEPTUNE + 200.0),  'e' : 0.0,        'i': 0.0,            'raan': 0.0,         'argp': 0.0,            'nu': 0.0}

    def __init__(self, render_mode = None):
        
        self.propellant_tank   = Propellant_Tank(self.INITIAL_PROPELLANT_MASS)
        self.thruster          = Thruster()
        self.orbit_propagator  = Orbit_Propagator(self.COM_ORBIT, self.OBSERVER_ORBIT)
        self.battery           = Battery(self.INITIAL_ENERGY) ##############
        self.DataClass         = Data(self.INITIAL_DATA_STORAGE) ################
        self.state = None
        self.action_space = spaces.Discrete(4)

        self._ai_action_to_spacecraft_action = {
            0: np.array([-1, 0]), # Thruster PROGRADE, Communications OFF
            1: np.array([1, 0]), # Thruster RETROGRADE, Communications OFF
            2: np.array([0, 0]),  # Thruster OFF, Communications OFF
            3: np.array([0, 1])  # Thruster OFF, Communications ON
        }

        self.mass_spacecraft = self.MASS_SPACECRAFT_INITIAL

        self.thrust_vector = []
        self.position_vector_x_com = []
        self.position_vector_y_com = []
        self.position_vector_x_obs = []
        self.position_vector_y_obs = []
        self.time_vector = [0]
        self.data_sent = 0
        self.prop_used = 0
        self.en_used = 0

        self.steps_to_truncate = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.renderer = Renderer()
            self.orbit_propagator.reset_orbits()
            self.com_orbit_points = self.orbit_propagator.get_orbit_points(self.orbit_propagator.orb_com)
            self.obs_orbit_points = self.orbit_propagator.get_orbit_points(self.orbit_propagator.orb_obs)
            self.comms = 0


    def reset(self):

        """ Resets the simulation. Sets the orbit and mass stored propellant tank to its initial values 
            Generally run at the beginning of a new episode
        """
        
        self.propellant_tank.reset_propellant_tank(self.INITIAL_PROPELLANT_MASS)
        self.battery.reset_battery(self.INITIAL_ENERGY)
        self.DataClass.reset_data_storage(self.INITIAL_DATA_STORAGE)
        self.orbit_propagator.reset_orbits()

        self.state = self.get_state()

        self.mass_spacecraft = self.MASS_SPACECRAFT_INITIAL

        self.steps_to_truncate = 0
        self.data_sent = 0
        self.prop_used = 0
        self.en_used = 0

        self.thrust_vector = []
        self.position_vector_x_com = [self.orbit_propagator.orb_com.r[0].value]
        self.position_vector_y_com = [self.orbit_propagator.orb_com.r[1].value]
        self.position_vector_x_obs = [self.orbit_propagator.orb_obs.r[0].value]
        self.position_vector_y_obs = [self.orbit_propagator.orb_obs.r[1].value]
        self.time_vector = [0]

        return self.state, {}


    def step(self, action):

        """ Takes one time step of the entire simulation """

        s = self.state
        assert s is not None, "Call reset before using Spacecraft object."

        self.thruster_action     = self._ai_action_to_spacecraft_action[action][0]
        communication_on_off      = self._ai_action_to_spacecraft_action[action][1]
        self.comms = communication_on_off
        # Propellant Tank
        propellant_used = abs(self.thruster_action) * 0.01
        self.propellant_tank.remove_mass(propellant_used)
        self.prop_used = propellant_used
        # Computing burn duration
        t_burn = 5 * abs(self.thruster_action) 
        
        # Thruster
        thrust = self.thruster_action * self.thruster.get_thrust(propellant_used, t_burn)
        acceleration = thrust / self.mass_spacecraft
        if self.propellant_tank.current_mass <= 0:
            acceleration = 0
        
        # Current distance
        dist = self.orbit_propagator.compute_distance_between_spacecraft()
        
        # Orbit_Propagator
        self.orbit_propagator.acceleration_com = acceleration
        self.orbit_propagator.propagate_orbits(self.dt, t_burn)

        self.mass_spacecraft = self.mass_spacecraft - propellant_used
        self.ignition_on = thrust > 0

        # Battery
        energy_used = self.comms*self.POWER_CONSUMPTION*self.dt ########################
        self.battery.remove_energy(energy_used)
        self.en_used = energy_used

        #Data_Storage
        self.data_sent= self.comms*self.DataClass.DataSent(dist*1000, self.dt)
        self.DataClass.DataToSend(self.data_sent)
       
        self.state = self.get_state()
        reward = self.get_reward()
        terminated = self._terminal()
        truncated = self._truncated()

        self.thrust_vector.append(getattr(thrust, "tolist", lambda: value)())
        self.position_vector_x_com.append(self.orbit_propagator.orb_com.r[0].value)
        self.position_vector_y_com.append(self.orbit_propagator.orb_com.r[1].value)
        self.position_vector_x_obs.append(self.orbit_propagator.orb_obs.r[0].value)
        self.position_vector_y_obs.append(self.orbit_propagator.orb_obs.r[1].value)
        self.time_vector.append(self.time_vector[-1]+self.dt)

        if self.render_mode == "human":
            self.render()

        self.steps_to_truncate += 1
        
        return self.state, reward, terminated, truncated, {}

    
    def get_state(self):
        return self._get_state_new()

    def get_reward(self):
        return self._get_reward3()
    
    def _get_state_new(self):
        r_com = self.orbit_propagator.orb_com.r.value / self.COM_ORBIT['a']
        r_obs = self.orbit_propagator.orb_obs.r.value / self.OBSERVER_ORBIT['a']
        
        h_com = self.orbit_propagator.orb_com.a.value / self.R_NEPTUNE
        h_obs = self.orbit_propagator.orb_obs.a.value / self.R_NEPTUNE

        ecc_com = self.orbit_propagator.orb_com.ecc.value
        ecc_obs   = self.orbit_propagator.orb_obs.ecc.value

        argp_com = self.orbit_propagator.orb_com.argp.value
        argp_obs   = self.orbit_propagator.orb_obs.argp.value

        theta_obs     = self.orbit_propagator.orb_obs.nu.value
        theta_com     = self.orbit_propagator.orb_com.nu.value

        propellant_mass = self.propellant_tank.current_mass
        energy_level = self.battery.current_energy
        data_left = self.DataClass.current_data

        return np.array([h_com ,h_obs, ecc_com, ecc_obs, cos(argp_com), sin(argp_com), cos(argp_obs), sin(argp_obs), cos(theta_com), sin(theta_com), cos(theta_obs), sin(theta_obs), propellant_mass, energy_level, data_left])

    def _get_reward3(self):

        if self.data_sent > 0:
            rew_data = 100*(self.data_sent/self.INITIAL_DATA_STORAGE)
        else:
            rew_data = 0
        
        if self.prop_used > 0:
            rew_prop_u = -self.prop_used
        else:
            rew_prop_u = 0

        if self.en_used > 0:
            rew_bat_u = -(self.en_used/self.INITIAL_ENERGY)
        else:
            rew_bat_u = 0

        if self.DataClass.current_data <= 0:
            rew_data += 1000

        if self.battery.current_energy <= 0:
            rew_bat = -10
        else:
            rew_bat = 0
        
        if self.propellant_tank.current_mass <= 0:
            rew_prop = -10
        else:
            rew_prop = 0

        reward = rew_data + rew_prop + rew_prop_u + rew_bat + rew_bat_u
        return reward

        

    def _terminal(self):
         terminal = False
         if self.propellant_tank.current_mass <= 0:
             terminal = True
         if self.battery.current_energy <= 0:
             terminal = True
         if self.DataClass.current_data <= 0:
             terminal = True
         if np.linalg.norm(self.orbit_propagator.orb_com.r.value) <= 24622:
             terminal = True
         return terminal


    def _truncated(self):
        
        truncated = False

        if self.steps_to_truncate >= self.MAX_STEPS:
            truncated = True

        return truncated


    def render(self):
        r = self.orbit_propagator.orbit_compare()
        l = len(self.orbit_propagator.positions_com)

        if len(self.orbit_propagator.positions_com) < 101:
            self.renderer.render(self.com_orbit_points, self.obs_orbit_points, self.orbit_propagator.positions_com, self.orbit_propagator.positions_obs, self.comms)
        else:
            self.renderer.render(self.com_orbit_points, self.obs_orbit_points, self.orbit_propagator.positions_com[l-100:l], self.orbit_propagator.positions_obs[l-100:l], self.comms)





if __name__ == "__main__":
    from itertools import count
    env = Spacecraft(render_mode = "human")
    # env = Spacecraft()
    env.reset()
    for t in count():
        observation, reward, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
        if done:
            break

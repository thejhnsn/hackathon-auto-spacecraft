import numpy as np
from torch.utils.hipify.hipify_python import bcolors
import gym
import spacecraft as sc
from itertools import count
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf


class PyEnvironment(object):

  def reset(self):
    """Return initial_time_step."""
    self._current_time_step = self._reset()
    return self._current_time_step

  def step(self, action):
    """Apply action and return new time_step."""
    if self._current_time_step is None:
        return self.reset()
    self._current_time_step = self._step(action)
    return self._current_time_step

  def current_time_step(self):
    return self._current_time_step

  def time_step_spec(self):
    """Return time_step_spec."""

  @abc.abstractmethod
  def observation_spec(self):
    """Return observation_spec."""

  @abc.abstractmethod
  def action_spec(self):
    """Return action_spec."""

  @abc.abstractmethod
  def _reset(self):
    """Return initial_time_step."""

  @abc.abstractmethod
  def _step(self, action):
    """Apply action and return new time_step."""





env = sc.Spacecraft()
env.reset()



##return current data
def outtputcurrent():
    energy = env.battery.current_energy
    prop = env.propellant_tank.current_mass
    comm = env.DataClass.current_data
    time = env.time_vector[-1]

    if(energy <= 0):
        return (f"Energy: {energy}, Propoltion Left: {prop}, Comms: {comm}, Time: {time} \n" + bcolors.FAIL + bcolors.BOLD + "Out of energy" + bcolors.ENDC)
    elif(prop <= 0):
        return(f"Energy: {energy}, Propoltion Left: {prop}, Comms: {comm}, Time: {time} \n" + bcolors.FAIL + bcolors.BOLD +"Out of propellant"+ bcolors.ENDC)
    elif(comm <= 0):
        return(f"Energy: {energy}, Propoltion Left: {prop}, Comms: {comm}, Time: {time} \n" + bcolors.OKGREEN + bcolors.BOLD + "Communication done! Programm ran succesfully"+ bcolors.ENDC)
    elif(time >= env.MAX_STEPS):
        return(f"Energy: {energy}, Propoltion Left: {prop}, Comms: {comm}, Time: {time} \n" + bcolors.FAIL + bcolors.BOLD + "Time is up!" + bcolors.ENDC)
    else:
        return(f"Energy: {energy}, Propoltion Left: {prop}, Comms: {comm}, Time: {time} \n" + bcolors.OKBLUE + bcolors.BOLD + "Programm is running" + bcolors.ENDC)

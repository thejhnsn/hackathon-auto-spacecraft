import numpy as np
from torch.utils.hipify.hipify_python import bcolors

import spacecraft as sc
from itertools import count
import openai
import os
import time
import tensorflow as tf





env = sc.Spacecraft()
env.reset()

openai.api_key = os.getenv("OPENAI_API_KEY")


c = 0
for t in count():
    c+= 1
    observation, reward, terminated, truncated, _ = env.step(3)

    done = terminated or truncated
    energy = env.battery.current_energy
    prop = env.propellant_tank.current_mass
    comm = env.DataClass.current_data
    time = env.time_vector[-1]

    if done:
        print(f"Numeber of steps: {c}  Energy: {energy}, Propoltion Left: {prop}, Comms: {comm}, Time: {time} \n")
        if(energy <= 0):
            print(bcolors.FAIL + bcolors.BOLD + "Out of energy" + bcolors.ENDC)
        elif(prop <= 0):
            print(bcolors.FAIL + bcolors.BOLD +"Out of propellant"+ bcolors.ENDC)
        elif(comm <= 0):
            print(bcolors.OKGREEN + bcolors.BOLD + "Communication done! Programm ran succesfully"+ bcolors.ENDC)
        else:
            print(bcolors.FAIL + bcolors.BOLD +"Time is up!"+ bcolors.ENDC)
        break
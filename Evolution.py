import numpy as np
from torch.utils.hipify.hipify_python import bcolors

import spacecraft as sc
from itertools import count
import os
import time
import tensorflow as tf


population = 10

agents = [None] * population
environments = [None] * population
rewards = [None] * population

battery = [None] * population
done = [None] * population
prop = [None] * population
comm = [None] * population
time = [None] * population


for k in range(population):
    environments[k] = sc.Spacecraft()
    environments[k].reset()

    agents[k] = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='softmax')  # give state from 0 to 3
    ])

    agents[k].set_weights([np.random.randn(*w.shape) for w in agents[k].get_weights()])
    agents[k].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""
input parameters:
        self.thrust_vector = []

        self.position_vector_x_com = []
        self.position_vector_y_com = []
        self.position_vector_x_obs = []
        self.position_vector_y_obs = []

        self.data_sent = 0
        self.prop_used = 0
        self.en_used = 0

        self.steps_to_truncate = 0

"""


c = 0
for t in count():
    c+= 1
    for k in range(population):
        if(t != 0):
            listTemp = [environments[k].thrust_vector[-1], environments[k].position_vector_x_com[-1],environments[k].position_vector_y_com[-1],environments[k].data_sent,environments[k].prop_used,environments[k].en_used,environments[k].steps_to_truncate]
        else:
            listTemp = [0, 0, 0, 0, 0, 0, 0]
        input_values = np.array(listTemp)
        observation, reward, terminated, truncated, _ = environments[k].step(agents[k].predict(input_values))



        done[k] = terminated or truncated
        battery[k] = environments[k].battery.current_energy
        prop[k] = environments[k].propellant_tank.current_mass
        comm[k] = environments[k].DataClass.current_data
        time[k] = environments[k].time_vector[-1]




        if done[k]:
            rewards[k] = reward
            print(f"AI number: {k} Numeber of steps: {c}  Energy: {battery[k]}, Propoltion Left: {prop[k]}, Comms: {comm[k]}, Time: {time[k]} \n")
            if(battery[k] <= 0):
                print(bcolors.FAIL + bcolors.BOLD + "Out of energy" + bcolors.ENDC)
            elif(prop[k] <= 0):
                print(bcolors.FAIL + bcolors.BOLD +"Out of propellant"+ bcolors.ENDC)
            elif(comm[k] <= 0):
                print(bcolors.OKGREEN + bcolors.BOLD + "Communication done! Programm ran succesfully"+ bcolors.ENDC)
            else:
                print(bcolors.FAIL + bcolors.BOLD +"Time is up!"+ bcolors.ENDC)

import numpy as np
from torch.utils.hipify.hipify_python import bcolors

import spacecraft as sc
from itertools import count
import os
import time
import tensorflow as tf


population = 100
generation = 0
threads = population

agents = [None] * population
environments = [None] * population
rewards = [None] * population

battery = [None] * population
done = [False] * population
prop = [None] * population
comm = [None] * population
timeArr = [None] * population


for k in range(population):
    environments[k] = sc.Spacecraft()
    environments[k].reset()

    agents[k] = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(8, activation='linear'),
        tf.keras.layers.Dense(4, activation='softmax')  # give probabilities of each step (4)
    ])
    #agents[k].summary()

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


for t in range(1000):
    c = 0

    while done.count(True) != population:
        for k in range(population):
            if done[k]:
                continue
            if(c != 0):
                listTemp = [environments[k].thrust_vector[-1], environments[k].position_vector_x_com[-1],environments[k].position_vector_y_com[-1],environments[k].data_sent,environments[k].prop_used,environments[k].en_used,environments[k].steps_to_truncate]
            else:
                listTemp = [0, 0, 0, 0, 0, 0, 0]

            input_values = np.array([listTemp])
            #input_values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
            temp = agents[k].predict(input_values, verbose = 0)[0]
            #print("t: ", t, ", k: ", k, ", temp: ", temp)
            # find max prob in array
            max = 0
            index = 0
            for i in range(4):
                if temp[i] > max:
                    max = temp[i]
                    index = i
            observation, reward, terminated, truncated, _ = environments[k].step(index)

            done[k] = terminated or truncated
            battery[k] = environments[k].battery.current_energy
            prop[k] = environments[k].propellant_tank.current_mass
            comm[k] = environments[k].DataClass.current_data
            timeArr[k] = environments[k].time_vector[-1]

            if done[k]:
                rewards[k] = reward
                print(f"Generation: {t} AI number: {k} Numeber of steps: {c}  Energy: {battery[k]}, Propoltion Left: {prop[k]}, Comms: {comm[k]}, Time: {timeArr[k]}, Reward: {rewards[k]} \n")
        c += 1
                #if(battery[k] <= 0):
                    #print(bcolors.FAIL + bcolors.BOLD + "Out of energy" + bcolors.ENDC)
                #elif(prop[k] <= 0):
                    #print(bcolors.FAIL + bcolors.BOLD +"Out of propellant"+ bcolors.ENDC)
                #elif(comm[k] <= 0):
                    #print(bcolors.OKGREEN + bcolors.BOLD + "Communication done! Programm ran succesfully"+ bcolors.ENDC)
                #else:
                    #print(bcolors.FAIL + bcolors.BOLD +"Time is up!"+ bcolors.ENDC)

    # get indices of the worst performing agents
    worst = np.argsort(rewards)[:int(population/2)]
    #print("worst: ", worst)

    # get indices of the best performing agents
    best = np.argsort(rewards)[int(population/2):]
    #print("best: ", best)

    # copy weights of the best performing agents to the worst performing agents
    for i in range(int(population/2)):
        agents[worst[i]].set_weights(agents[best[i]].get_weights())

    # mutate all agents by
    for k in range(population):
        # iterate through all weights
        for i in range(len(agents[k].get_weights())):
            # iterate through all weights in the layer
            for j in range(len(agents[k].get_weights()[i])):
                # iterate through all weights in the neuron
                for l in range(len(agents[k].get_weights()[i][j])):
                    # get random number between 0 and 1000 (0 - 100%)
                    rand = np.random.randint(0, 1000)
                    # 2% chance to inverse one weight
                    if(rand < 2):
                        agents[k].get_weights()[i][j][l] *= -1
                    # 4% chance to add random number between -1 and 1 to one weight
                    elif(rand < 4):
                        agents[k].get_weights()[i][j][l] += np.random.uniform(-1, 1)
                    # 6% chance to pick random number between -1 and 1 as new weight
                    elif(rand < 6):
                        agents[k].get_weights()[i][j][l] = np.random.uniform(-1, 1)

    # reset the environments
    for k in range(population):
        environments[k].reset()
        done[k] = False
        battery[k] = environments[k].battery.current_energy
        prop[k] = environments[k].propellant_tank.current_mass
        comm[k] = environments[k].DataClass.current_data
        timeArr[k] = environments[k].time_vector[-1]
        rewards[k] = 0

    generation += 1

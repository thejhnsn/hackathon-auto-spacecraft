import numpy as np
import spacecraft as sc
import NeuralNetwork as nn
import threading


population = 50
generation = 0
#threadsArray = [0] * 8

agents = [nn.NeuralNetwork] * population
environments = [sc.Spacecraft] * population
rewards = [0.0] * population

battery = [0] * population
done = [False] * population
prop = [0.0] * population
comm = [0.0] * population
timeArr = [0] * population

for k in range(population):
    environments[k] = sc.Spacecraft()
    environments[k].reset()

    agents[k] = nn.NeuralNetwork(np.array([6, 128, 128, 128, 4]))

for t in range(20):
    c = 0

    while done.count(True) != population:
        for k in range(population):
            if done[k]:
                continue
            if(c != 0):
                listTemp = [environments[k].thrust_vector[-1], environments[k].position_distance_vector[-1],environments[k].data_sent,environments[k].prop_used,environments[k].en_used,environments[k].steps_to_truncate]
            else:
                listTemp = [0, 0, 0, 0, 0, 0]

            input_values = np.array(listTemp)
            #input_values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
            temp = agents[k].feed_forward(input_values)
            #print("t: ", t, ", k: ", k, ", temp: ", temp)
            # find max prob in array
            max = 0
            index = 0
            for i in range(4):
                if temp[i] > max:
                    max = temp[i]
                    index = i

            #if(index == 3 and environments[k].position_distance_vector[-1] > 1190.834):
            #    agents[k].add_fitness(-10)
            observation, reward, terminated, truncated, _ = environments[k].step(index)

            done[k] = terminated or truncated
            battery[k] = environments[k].battery.current_energy
            prop[k] = environments[k].propellant_tank.current_mass
            comm[k] = environments[k].DataClass.current_data
            timeArr[k] = environments[k].time_vector[-1]
            agents[k].add_fitness(reward)
            rewards[k] += reward

            if done[k]:
                print(f"Generation: {t} AI number: {k} Numeber of steps: {c}  Energy: {battery[k]}, Propoltion Left: {prop[k]}, Comms: {comm[k]}, Time: {timeArr[k]}, Reward: {rewards[k]}, Fitness: {agents[k].get_fitness()} \n")
        c += 1

    agents.sort()
    CurrentLeader = agents[population - 1]
    print("And the winner has: ", CurrentLeader.get_fitness())
    print("Standardabweichung", np.std(rewards))
    if(np.std(rewards) < 2):
        modifier = 0.1
    else:
        modifier = 1

    for i in range(population // 2):
        agents[i].copy_weights(agents[i + (population // 2)].get_weights())

        agents[i].mutate(modifier)

    for k in range(population):
        environments[k].reset()
        done[k] = False
        battery[k] = environments[k].battery.current_energy
        prop[k] = environments[k].propellant_tank.current_mass
        comm[k] = environments[k].DataClass.current_data
        timeArr[k] = environments[k].time_vector[-1]
        agents[k].set_fitness(0.0)
        rewards[k] = 0

    generation += 1

f = open("weights", "x") # create file
# write best weights to file
for i in range(len(CurrentLeader.get_weights())):
    f.write(str(CurrentLeader.get_weights()[i]))
    f.write("\n")
f.close()
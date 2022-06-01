# run.py
from model import *
import matplotlib.pyplot as plt
import numpy as np

## parameters for the world/simulation
num_iterations = 300
num_ants = 30
num_food = 100
width = 100
height = 100
food_distribution = "random" # can be random, or power_law

## parameters for the ants
prob_peromones = 0.4


num_runs = 50



for prob in nest_probabilities:
    food_collected = [0] * num_iterations
    iterations = [0] * num_iterations
    for j in range(num_runs):
        # initialize model
        model = World(num_ants, num_food, width, height, prob_peromones, prob, min_dist_between_nests)
        # simulation loop
        num_food_collected = 0
        for i in range(num_iterations):
            num_food_collected = num_food - model.step()
            iterations[i] = i
            food_collected[i] += num_food_collected

    food_collected = np.array(food_collected)
    print("food collected 1 ", food_collected)
    food_collected = food_collected/50
    print("food_coollleelkjf 2", food_collected)
    plt.plot(iterations, food_collected, label=str(prob))


plt.xlabel('Time')
plt.ylabel('Food Collected')
plt.title('Food Collected Over Time, over different Probabilities of Creating a Nest')
plt.legend()
plt.show()


# display results
print(num_food_collected, "out of", num_food, "food was collected in", num_iterations, "iterations")

from mesa import Agent, Model
import numpy as np
import math
from pheromones import Pheromone

class Ant(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.model.increment_agent_count()
        self.direction = np.random.rand() * 2 * math.pi
        self.food = 0
        self.nest_location = (0,0)
        self.dx_from_nest = 0
        self.dy_from_nest = 0
        self.drop_pheromone = False
        self.food_position = False

    def set_nest_location(self, loc):
        self.nest_location = loc

    def move(self):
        if (self.food):
            self.return_to_nest()
        elif (self.food_position and self.food == 0):
            self.return_to_food()
        else:
            self.random_walk()

    def random_walk(self):
        min = self.direction - (math.pi/3)
        r = np.random.rand()
        self.direction = (r * (math.pi/2)) + min
        dx = np.cos(self.direction)
        dy = np.sin(self.direction)
        new_position = (self.pos[0] + dx, self.pos[1] + dy)
        self.model.space.move_agent(self, new_position)

    def return_to_nest(self):
        Dx = self.nest_location[0] - self.pos[0]
        Dy = self.nest_location[1] - self.pos[1]
        dist = math.hypot(Dx, Dy)
        if (dist != 0):
            dx = Dx/dist
            dy = Dy/dist
            self.model.space.move_agent(self, (self.pos[0] + dx, self.pos[1] + dy))
            if (self.drop_pheromone):
                self.model.generate_pheromones(self.pos)
        else:
            print("dist is 0")

    def return_to_food(self):
        Dx = self.food_position[0] - self.pos[0]
        Dy = self.food_position[1] - self.pos[1]
        dist = math.hypot(Dx, Dy)
        if (dist != 0):
            dx = Dx/dist
            dy = Dy/dist
            self.model.space.move_agent(self, (self.pos[0] + dx, self.pos[1] + dy))

    def pick_up_food(self, food):
        self.food = 1
        self.food_position = self.pos
        food.decrease_value()
        if (food.get_value() < 1):
            self.model.space.remove_agent(food)
            self.model.decrease_food_count()
        if (np.random.rand() < self.model.prob_pheromones):
            self.drop_pheromone = True

    # scan current location for food or nest
    def scan_area(self):
        neighbors = self.model.space.get_neighbors(self.pos, 1)
        for n in neighbors:
            agent_type = type(n).__name__

            # ant collides with a food item, pick up food item
            if (agent_type == "Food"):
                self.pick_up_food(n)

            # drop food item if it is at a nest
            elif (agent_type == "Nest"):
                self.food = 0
                self.remember_nest()
                self.drop_pheromone = False

            # site fidelity, returned to location of food
            elif (self.food_position == self.pos):
                self.food_position = False

    def remember_nest(self):
        self.nest_location = self.pos

    def randomly_generate_nest(self):
        if (np.random.rand() < self.model.prob_create_nest):
            Dx = self.nest_location[0] - self.pos[0]
            Dy = self.nest_location[1] - self.pos[1]
            dist = math.hypot(Dx, Dy)
            if (dist > self.model.min_dist_between_nests):
                self.model.generate_nest(self.pos)
                self.remember_nest()

    def step(self):
        self.scan_area()
        self.move()
        self.randomly_generate_nest()
        pass


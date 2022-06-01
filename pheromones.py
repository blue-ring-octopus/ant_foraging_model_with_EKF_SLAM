from mesa import Agent, Model

class Pheromone(Agent):
    def __init__(self, model):
        unique_id = model.get_agent_count()
        super().__init__(unique_id, model)
        self.model.increment_agent_count()
        self.model = model
        self.value = 30

    def step(self):
        self.value -=1
        if (self.value < 1):
            self.pos = None
            #self.model.space.remove_agent(self)
        pass

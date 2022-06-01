from mesa import Agent, Model

class Food(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.model.increment_agent_count()
        self.value = 1

    def decrease_value(self):
        self.value = self.value - 1

    def get_value(self):
        return self.value

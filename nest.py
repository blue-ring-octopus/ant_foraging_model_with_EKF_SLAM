from mesa import Agent, Model

class Nest(Agent):
    def __init__(self, model):
        unique_id = model.get_agent_count()
        super().__init__(unique_id, model)
        self.model.increment_agent_count()

from mesa.visualization.ModularVisualization import ModularServer
from model import World
from SimpleContinuousModule import SimpleCanvas
from mesa.visualization.UserParam import UserSettableParameter
import numpy as np
from mesa.visualization.modules import ChartModule


### MODEL PARAMETERS! update them here
model_params = {
    "num_agents": UserSettableParameter("slider","num_agents", value=10, min_value=1, max_value=50,step=1),
    "num_food": UserSettableParameter("slider","num_food", value=5, min_value=1, max_value=50,step=1),
    "pheromone_attraction": UserSettableParameter("slider","pheromone_attraction", value=0.01, min_value=0, max_value=1,step=0.01),
}

def draw(agent):

    agent_type = type(agent).__name__
    portrayal = {}

    if (agent_type == "Obstacle"):
        portrayal = {"Shape": "rect",
                     "Filled": "true",
                     "Layer": 0,
                     #"Color": "black",
                     "Color": "white",
                     "w": 1/50,
                     "h": 1/50
        }
    elif (agent_type == "Ant"):
        color="red"
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 4,
                     "Color": color,
                     "r": 4}
        
    elif (agent_type == "Pheromone"):
        if agent.type=="Food":
            color="cyan"
            size=np.clip(agent.concentration, 1, 5)

        elif agent.type=="Trail":
            color="magenta"
            size=np.clip(agent.concentration*10, 0.01, 3)

      #      size=agent.concentration
        elif agent.type=="Home":
            color="blue"
            size=np.clip(agent.concentration, 0.01, 5)
        else:
            color="grey"
            size=np.clip(agent.concentration, 5, 10)

        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 3,
                     "Color": color,
                     "r": size}
    elif (agent_type == "Home"):
        portrayal = {"Shape": "rect",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": "black",
                     "w": 1/50,
                     "h": 1/50}
    else:
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 2,
                     "Color": "blue",
                     "r": 3}

    return portrayal
#{"Label": "food_RMSE", "Color": "#FF0000"}
chart = ChartModule(
    [{"Label": "information", "Color": "#0000FF"},{"Label": "missing_food", "Color": "#00FF00"}], data_collector_name="datacollector"
)
canvas = SimpleCanvas(draw, 500, 500)

server = ModularServer(World,
                       [canvas, chart],
                       "World",
                       model_params)

server.launch()

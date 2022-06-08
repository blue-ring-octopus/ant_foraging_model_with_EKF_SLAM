from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
import numpy as np
from scipy import stats
import cv2
from mesa.datacollection import DataCollector
from mesa.batchrunner import batch_run
import matplotlib.pyplot as plt
import pickle
import colorsys
def compute_accuracy(model):
    food_pos_est=[]
    food_pos_real=[]
    rmse=0
    det_info=0
    food_count=0
    for food in model.food_list:    
        info=np.zeros((2,2))
        pos=np.zeros(2)
        for agent in model.ant_list:
            if str(food.unique_id) in agent.pheromone["Food"].keys():
                cov_inv=np.linalg.inv(agent.pheromone["Food"][str(food.unique_id)].covariance)
                pos+=cov_inv@np.asarray(agent.pheromone["Food"][str(food.unique_id)].pos)
                info+=cov_inv

        if np.trace(info)!=0:
            cov=np.linalg.inv(info)
            pos=cov@pos
            food_count+=1
            
        food_pos_est.append(pos)
        food_pos_real.append(np.asarray(food.pos))
        rmse+=1/model.num_food*(np.linalg.norm(pos-np.asarray(food.pos)))**2
        det_info+=1/model.num_food*np.linalg.det(info)

    return model.num_food-food_count,np.sqrt(rmse), det_info


class Ant(Agent):
    def __init__(self, unique_id, model,pos, levy_scale=5,levy_freq=10, gaussian_std=0.05, sight_range=40, fov=np.pi*3/4, pf_amplitude=5, range_noise=1, bearing_noise=0.2, memory_size=100,pheromone_attraction=0.1):
        super().__init__(unique_id, model)
        self.state = np.array([pos[0],pos[1],self.angle_wrapping(np.random.uniform(1/2*np.pi, 3/2*np.pi))]) #[x,y,theta]
        self.input= np.zeros(3) #[vx,vy,vtheta]
        self.levy_scale=levy_scale
        self.gaussian_std=gaussian_std
        self.fov=fov
        self.sight_range=sight_range
        self.pf_amplitude=pf_amplitude
        self.bearing_noise=bearing_noise
        self.range_noise=range_noise
        self.A=np.eye(3)
        self.covariance=np.eye(3)*99999
        self.alert=False
        self.levy_freq=levy_freq
        self.pheromone={"Food":{},"Obstacle":{}}
        self.neighbor_pheromone={}
        self.average_pheromone={}
        self.memory_size=memory_size
        self.pheromone_memory=[None for i in range(memory_size)]
        self.pheromone_attraction=pheromone_attraction 
        self.return_home=False
        
    def state_transition(self):
        dt=self.model.dt
        tf=np.array([[np.cos(self.state[2]), np.sin(self.state[2])], [-np.sin(self.state[2]), np.cos(self.state[2])]])
        v_r=tf@self.input[0:2]
    
        v_clip=np.asarray([np.clip(v_r[0],0,20),np.clip(v_r[1],-7,7)])

        v=tf.T@v_clip
        self.input[0]=v[0]
        self.input[1]=v[1]
        self.input[2]=np.clip(self.input[2],-1/4*np.pi,1/4*np.pi)
        
        self.state=self.A@self.state+dt*self.input
        
        self.state[0]=np.clip(self.state[0], 1, self.model.space.width-2)
        self.state[1]=np.clip(self.state[1],1, self.model.space.height-2)
        self.state[2]=self.angle_wrapping(self.state[2])
        new_position=(self.state[0],self.state[1] )
        self.model.space.move_agent(self, new_position)


    def levy_flight(self):
        dtheta = np.random.uniform(-np.pi/2, np.pi/2)
        dr=stats.levy.rvs(scale=self.levy_scale)
        dx=dr*np.cos(self.state[2])
        dy=dr*np.sin(self.state[2])
        
        self.input+=np.array([dx, dy,dtheta])

    # observe neighborhood 
    def scan_area(self):
        neighbors = self.model.space.get_neighbors(self.pos, radius=self.sight_range)
        in_sight=[]
        for n in neighbors:
            agent_type = type(n).__name__
            dy=n.pos[1]-self.pos[1]
            dx=n.pos[0]-self.pos[0]

            if (agent_type == "Pheromone"):
                if n.type=="Trail":
                    if np.linalg.norm([dx,dy])<=2:
                        self.pheromone_memory.append({"pheromone":n, "direction": self.state[2]})
                        self.pheromone_memory=self.pheromone_memory[-self.memory_size:]
            bearing=np.arctan2(dy,dx)-self.state[2]
            
            if bearing>=-self.fov/2 and  bearing<=self.fov/2:    
                in_sight.append(n)
                
        for n in in_sight:
            z=self.measure_neighbors(n.pos)
            agent_type = type(n).__name__
            if (agent_type == "Pheromone"):
                if n.type=="Trail":
                    self.input[0]+=self.pheromone_attraction*n.concentration*(z[0]*np.cos(z[1]))
                    self.input[1]+=self.pheromone_attraction*n.concentration*(z[0]*np.sin(z[1]))
                ##    self.input[0]+=self.pheromone_attraction*n.concentration*np.cos(n.direction)
                  #  self.input[1]+=self.pheromone_attraction*n.concentration*np.sin(n.direction)
                    self.input[2]+=5*self.pheromone_attraction*n.concentration*self.angle_wrapping((n.direction-self.state[2]))

            else:
                if (agent_type == "Obstacle"):
                    amplitude=self.pf_amplitude*40
                  #  self.update_pheromone(neighbor=n,z=z)

                elif (agent_type=="Food"):
                    amplitude=self.pf_amplitude
                    self.update_pheromone(neighbor=n,z=z)
                    self.return_home=True
                elif (agent_type=="Ant"):
                    amplitude=self.pf_amplitude*40
                elif (agent_type=="Home"):
                    amplitude=0
                    self.return_home=False

                self.add_potential_field(n.pos, amplitude)
        self.get_average_pheromone()
        for n in in_sight:
            agent_type = type(n).__name__
            if agent_type=="Food":
                concentration=np.log(1+np.linalg.det(self.average_pheromone[str(n.unique_id)]["covariance"]))
                self.pheromone_trail(concentration)        

        #    elif agent_type=="Obstacle":
     #           concentration=0.1*np.log(1+np.linalg.det(self.average_pheromone[str(n.unique_id)]["covariance"]))
       #         self.pheromone_trail(concentration)   
                
            elif agent_type=="Ant":
                self.share_informatoin(n)

                
    def add_pheromon_attraction(self):
        for item in self.pheromone:
            for pheromone in item:
                k=pheromone.concentration
                dx=pheromone.pos[0]-self.pos[0]
                dy=pheromone.pos[1]-self.pos[1]
                direction=self.angle_wrapping(np.arctan2(dy,dx)-self.state[2])

                self.input[0]+=k*dx
                self.input[1]+=k*dy
                self.input[3]+=k*direction
    def add_potential_field(self,target_pos, amplitude):
        dx=self.pos[0]-target_pos[0]
        dy=self.pos[1]-target_pos[1]
    
        r=np.linalg.norm([dx, dy])+0.0001
        f=amplitude/(abs(r)**2)
        self.input[0]+=f*dx/r
        self.input[1]+=f*dy/r
            
    def localization():
        pass

    def measure_neighbors(self, target_pos):
        dx=target_pos[0]-self.pos[0]
        dy=target_pos[1]-self.pos[1]
        target_range=np.linalg.norm([dy,dx])+np.random.normal(0,self.range_noise)-0.2
        target_bearing=np.arctan2(dy,dx)-self.state[2]+np.random.normal(0, self.bearing_noise)
        target_bearing=self.angle_wrapping(target_bearing)
        return [target_range , target_bearing]
        
    def update_pheromone(self, neighbor, z):
        neighbor_id=neighbor.unique_id
        neighbor_type=type(neighbor).__name__
        #create new pheromone for new found food
        if str(neighbor_id) in self.pheromone[neighbor_type].keys():
            a=self.pheromone[neighbor_type][str(neighbor_id)]
        else:
            sigma= np.eye(2)*99999
            a = Pheromone(self.model.agent_count, self.model,sigma, neighbor_type)
            pos=[self.pos[0]+z[0]*np.cos(z[1]+self.state[2]), 
                    self.pos[1]+z[0]*np.sin(z[1]+self.state[2])]
            
            pos[0]=np.clip(pos[0],0,self.model.width-0.01)
            pos[1]=np.clip(pos[1],0,self.model.height-0.01)
            self.model.schedule.add(a)
            self.pheromone[neighbor_type][str(neighbor_id)]=a
            self.model.space.place_agent(a, (pos[0],pos[1]))
            self.model.agent_count+=1
        
        #kalman update step
        delta=np.asarray(a.pos)-np.asarray(self.pos)
        sigma=a.covariance
        q=delta[0]**2+delta[1]**2
        q=np.clip(q, 0.0001, np.inf)
        d=np.sqrt(q)
        z_hat=np.asarray([d, np.arctan2(delta[1],delta[0])-self.state[2]])
        z_hat[1]=self.angle_wrapping(z_hat[1])
        H=np.asarray([[delta[0]/d, delta[1]/d],[-delta[1]/q,delta[0]/q]])   
        K=sigma@H.T@np.linalg.inv(H@sigma@H.T+np.array([[self.range_noise,0],[0, self.bearing_noise]]))
        pos=np.asarray(a.pos)+K@(z-z_hat)
        a.covariance=(np.eye(2)-K@H)@sigma
        
        pos[0]=np.clip(pos[0],0,self.model.width-0.01)
        pos[1]=np.clip(pos[1],0,self.model.height-0.01)
        self.model.space.move_agent(a, (pos[0],pos[1]))
    

    def pheromone_trail(self, concentration):
        for item in self.pheromone_memory:
            if item:
                item["pheromone"].direction=self.angle_wrapping((item["pheromone"].concentration*item["pheromone"].direction+concentration*item["direction"])/(item["pheromone"].concentration+concentration))
                item["pheromone"].concentration+=concentration
      
    def share_informatoin(self, neighbor):
        neighbor.neighbor_pheromone[str(self.unique_id)]=self.pheromone
        
    def get_average_pheromone(self):
        '''
        calculate the inverse covariance weighting of neighbor covariances
        '''
        for a_type in self.pheromone:
            for a_id in self.pheromone[a_type]:
                covariance=[]
                pos=[]
                covariance.append(self.pheromone[a_type][a_id].covariance)
                pos.append(np.asarray(self.pheromone[a_type][a_id].pos))
                for neighbor in self.neighbor_pheromone:
                    if a_type in self.neighbor_pheromone[neighbor]:
                        if a_id in self.neighbor_pheromone[neighbor][a_type].keys():
                            covariance.append(self.neighbor_pheromone[neighbor][a_type][a_id].covariance)
                            pos.append(np.asarray(self.neighbor_pheromone[neighbor][a_type][a_id].pos))
                if len(covariance):
                    cov_inv=[np.linalg.inv(x) for x in covariance]
                    covariance_avg=np.linalg.inv(np.sum(cov_inv, axis=0))
                    pos_avg=covariance_avg@np.sum([cov_inv[i]@x for i,x in enumerate(pos)], axis=0)  
                    self.average_pheromone[a_id]={"pos": pos_avg, "covariance": covariance_avg}
            
    def step(self):
        self.scan_area()
        if(self.model.time%self.levy_freq==0):
            self.levy_flight()
        self.add_pheromon_attraction
        self.state_transition()

    def angle_wrapping(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
    
class Obstacle(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
    def step(self):
        pass

class Food(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
    def step(self):
        pass
    
class Pheromone(Agent):
    def __init__(self, unique_id, model,covariance, pheromone_type, evaporate=False, eva_rate=0.07):
        super().__init__(unique_id, model)
        self.covariance=covariance
        self.concentration=np.log(1+np.linalg.det(covariance))
        self.type=pheromone_type
        self.direction=0
        self.eva_rate=eva_rate
        self.evaporate=evaporate
        
    def step(self):
        if self.evaporate:
            self.concentration*=np.exp(-self.eva_rate)
        else:
            self.concentration=np.log(1+np.linalg.det(self.covariance))
            
class Home(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
    def step(self):
        pass
    
class World(Model):
    def __init__(self, num_agents, num_food, dt=1/10,pheromone_attraction=0.1):
        self.rmse=0
        self.missing_food=0
        self.datacollector = DataCollector(
            model_reporters={"food_RMSE": "rmse", "missing_food": "missing_food", "information": "det_info"},
        )
        self.scale=2
        self.map_img =(cv2.imread('map.png', cv2.IMREAD_GRAYSCALE)).T
        self.width = (self.map_img.shape[0])*self.scale-1
        self.height = (self.map_img.shape[1])*self.scale-1
        self.center = (self.width/2, self.height/2)
        self.num_agents = num_agents
        self.num_food= num_food
        self.dt=dt
        self.pheromone_attraction=pheromone_attraction
        self.home=(10,self.height-10 )

        self.space = ContinuousSpace(self.width, self.height, False, 0, 0)
        self.schedule = RandomActivation(self)
        self.running = True
        self.agent_count = 0    
        self.generate_home()
        self.generate_ants()
        self.generate_obstacle()
        self.generate_food()
        self.generate_pheromon_grid()
        self.time=0

    def generate_ants(self):
        self.ant_list=[]
        for i in range(0,self.num_agents):
            loc=np.random.multivariate_normal(np.asarray(self.home), np.eye(2))

            a = Ant(self.agent_count, self, loc, pheromone_attraction=self.pheromone_attraction)
            self.schedule.add(a)
            self.ant_list.append(a)
            self.space.place_agent(a, (loc[0],loc[1]))
            self.agent_count+=1

    def generate_obstacle(self):
        map_img=self.map_img
        obstacles_idx=np.where(map_img==0)
        for i in range(len(obstacles_idx[0])):
            a=Obstacle(self.agent_count, self)
            self.schedule.add(a)
            self.space.place_agent(a, (self.scale*obstacles_idx[0][i],self.scale*obstacles_idx[1][i]))
            self.agent_count+=1


    def generate_food(self):
        self.food_list=[]
        for i in range(self.num_food):
            f = Food( self.agent_count, self)
            x = np.random.rand() * self.space.width
            y = np.random.rand() * self.space.height
            self.food_list.append(f)
            self.schedule.add(f)
            self.space.place_agent(f, (x, y))
            self.agent_count+=1
            
    def generate_pheromon_grid(self):
        for i in np.linspace(0,self.space.width-0.01, 50):
            for j in np.linspace(0,self.space.height-0.01,50):
                a = Pheromone(self.agent_count, self,np.zeros((3,3)), "Trail",evaporate=True)
                self.schedule.add(a)
                self.space.place_agent(a, (i, j))
                self.agent_count+=1
                
    def generate_home(self):
        f = Home( self.agent_count, self)
        self.schedule.add(f)
        self.space.place_agent(f, self.home)
        self.agent_count+=1
        
    def step(self):
        self.schedule.step()
        self.time+=1
        self.missing_food, self.rmse, self.det_info=compute_accuracy(self)
        self.datacollector.collect(self)

if __name__ == "__main__":
    pheromone_attraction=np.arange(0,1,0.1)
    # all_results=[]
    # for pheromone in pheromone_attraction:
    #     params = {"num_agents": 10,
    #               "num_food":5,
    #               "pheromone_attraction": pheromone}
    
    #     results = batch_run(
    #         World,
    #         parameters=params,
    #         iterations=30, #Number of replications
    #         max_steps=500,
    #         number_processes=None,
    #         data_collection_period=1,
    #         display_progress=True)
    #     all_results.append(results)
    all_results=pickle.load(open( "results.p", "rb" ) )["results"]
    for i, results in enumerate(all_results): 
        info=[[] for x in range(30)]
        for item in results:
                info[int(item["iteration"])].append(item['information'])
        info=np.asarray(info)
        plt.plot(np.average(info, axis=0), 
                 color=colorsys.hsv_to_rgb(i/len(all_results),1,1),
                 label=str(pheromone_attraction[i])[0:3])
        
        plt.title("Cumulative Food Information")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("det(information)")

    plt.figure()
    for i, results in enumerate(all_results): 
        info=[[] for x in range(30)]
        for item in results:
                info[int(item["iteration"])].append(item['food_RMSE'])
        info=np.asarray(info)
        plt.plot(np.average(info, axis=0), 
                 color=colorsys.hsv_to_rgb(i/len(all_results),1,1),
                 label=str(pheromone_attraction[i])[0:3])
        
        plt.title("Food Location RMSE")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("L2 RMSE")

    plt.figure()
    for i, results in enumerate(all_results): 
        info=[[] for x in range(30)]
        for item in results:
                info[int(item["iteration"])].append(item['missing_food'])
        info=np.asarray(info)
        plt.plot(np.average(info, axis=0), 
                 color=colorsys.hsv_to_rgb(i/len(all_results),1,1),
                 label=str(pheromone_attraction[i])[0:3])
        
        plt.title("Missing Food")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("Food Count")
        
    plt.figure()
    end_step_info=[]
    for i, results in enumerate(all_results): 
        info=[[] for x in range(30)]
        for item in results:
                info[int(item["iteration"])].append(item['food_RMSE'])
        info=np.asarray(info)
        end_step_info.append(np.average(info, axis=0)[-1] )
    plt.plot(pheromone_attraction,end_step_info            )
    
    plt.title("Missing Food")
    plt.legend()
    plt.xlabel("step")
    plt.xlabel("Food Count")

import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
#import matplotlib.pyplot as plt #used for analysis only!

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    def __init__(self, env, epsilon=.5, learning_rate = 0.6, discount_factor = 0.4):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.directions = [None, 'forward', 'left', 'right']
        self.q = {}
        #Defaults used for setting weights
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        #Used for analysis
        self.trial_rewards = [] #log of all episodes in a 2d array[[penalty, reward]]
        self.failed_trials = 0  #number of failed trials
        self.passed_trials = 0  #number of passed trials

    def reset(self, destination=None):
        '''called before each episode'''
        self.planner.route_to(destination)
        #reset values after each episode
        self.trial_penalty = 0 #points gained for single episode
        self.trial_reward = 0  #points lost for a single episode

    def get_epsilon_decay(self):
        '''decays epislon value over time. Used to promote exploration in 
           earlier episodes.'''
        return float(self.epsilon) / (len(self.trial_rewards) + float(self.epsilon))

    def random_action(self):
        '''returns a random action, which increases our explored states.'''
        random_direction = random.choice(self.directions)
        return random_direction

    def greedy_action(self, state):
        '''returns the actions with the best known q value for a given state.'''
        q_vals = [self.get_Q_val(state, action) for action in self.directions]
        max_q_val = max(q_vals)
        if q_vals.count(max_q_val) > 1: #mult existing values
            best = [i for i in range(len(self.directions)) if q_vals[i] == max_q_val]
            direction_idx = random.choice(best)
        else:
            direction_idx = q_vals.index(max_q_val)
        return self.directions[direction_idx]

    def choose_action(self, state):
        '''decide if we should take a random action or a greedy action.'''
        epsilon = self.get_epsilon_decay()
        if random.random() < epsilon: #Roll the dice
            action = self.random_action() #Used for exploring the enviroment
        else:
            action =  self.greedy_action(state) #Pick the best value for Q - Greedy action
        return action

    def get_Q_val(self, state, action):
        '''Util to get a q value for a given state and action pair.'''
        q_val = self.q.get((state, action), 0.0)
        return q_val

    def set_Q_val(self, state, action, value):
        '''Util to set a q value for a given state and action pair.'''
        self.q[(state, action)] = value

    def learned_value(self, state, reward):
        '''generates a discounted max value used to slightly update our q val look up.'''       
        #estimate of the optimal future Q-value for the new_state
        max_q_new = max([self.get_Q_val(state, direction) for direction in self.directions])
        return reward + self.discount_factor * max_q_new

    def learn_Q(self, prev_state, action, reward, new_state):
        '''determine q val and update lookup.'''
        if prev_state is not None:
            prev_val = self.q.get((prev_state, action), None)
            if prev_val is None:
                new_val = reward
            else:
                learned_val = self.learned_value(new_state, reward)
                #moves the new q val slightly in direction of new value
                new_val = prev_val + (self.learning_rate * (learned_val - prev_val))
            self.set_Q_val(prev_state, action, new_val)

    def update(self, t):
        '''main program loop called at the begining of each state.'''
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)# Gather inputs to create state
    
        '''Update state: Important variables for state are light, oncoming, and left, 
           right is not required as driver can turn right on red without penalty.'''
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])

        #Select action according to your policy
        action = self.choose_action(self.state)

        #Execute action and get reward
        reward = self.env.act(self, action)

        #get the new state after taking the action
        inputs = self.env.sense(self)
        new_state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])

        #Learn policy based on state, action, reward, new_state
        self.learn_Q(self.state, action, reward, new_state)

        '''Logging for analysis of the agent.'''
        #if the reward is negative we consider it a penalty
        if reward < 0: 
            self.trial_penalty += reward
        else:
            self.trial_reward += reward
        #determine if the agent has reached the destination within the number of alloted steps.'''
        if(self.env.done == True):
            self.passed_trials += 1
            self.trial_rewards.append([self.trial_penalty, self.trial_reward])
        elif(self.env.get_deadline(self) <= 0):
            self.failed_trials += 1
            #print("Failed Trial", len(self.trial_rewards))
            self.trial_rewards.append([self.trial_penalty, self.trial_reward])
    #end of LearningAgent
 
def draw_Q_table(agent):
    #print learned states
    for state, action in agent.q:
        print("State: (Next Waypoint: {}, Light: {}, Oncoming: {}, Left: {})  Action: {}, Reward {}").format(
            state[0], state[1], state[2], state[3], action, agent.get_Q_val(state, action))
    print("\n")

def draw_chart(chart_data_array):
    print(chart_data_array)
    plt.plot(chart_data_array)
    plt.show()
    #print("draw_reward_chart called", chart_data_array)

def mean_reward(two_d_array):
    '''Calculate the mean reward by adding together the rewards and the penalties'''
    reward_tot = 0
    for reward in two_d_array:
        reward_tot += reward[0]+reward[1]
    return reward_tot/len(two_d_array) 

def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    pass_rate = (float(a.passed_trials)/ float(len(a.trial_rewards)))*len(a.trial_rewards)
    #print(a.trial_rewards)
    #Print stats after running a finite number of trials.
    print("Learning Rate: {}, Discount Factor: {}, Epsilon {}, Pass Rate: {}%, Explored States: {}, Mean Reward: {}, Number of Trials: {}").format(
        a.learning_rate, a.discount_factor, a.epsilon, pass_rate, len(a.q), mean_reward(a.trial_rewards), len(a.trial_rewards))


if __name__ == '__main__':
    run()
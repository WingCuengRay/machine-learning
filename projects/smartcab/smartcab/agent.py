import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

#import matplotlib.pyplot as plt
#import os
#clear = lambda: os.system('clear')

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, epsilon=1, learning_rate = 0.25, discount = 0.9):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.directions = [None, 'forward', 'left', 'right']
        self.q = {}
        #Defaults used for setting weights
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        
        self.last_state = None #Memoize last state
        self.last_action = None #Memoize last action

        self.trial_reward = 0
        self.trial_steps = 0
        self.failed_trials = 0
        self.passed_trials = 0
        self.trail_rewards = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        if(self.trial_steps >= self.env.get_deadline(self) ):
            #print("\033[0;31mFAIL Steps To Complete = {}, Deadline = {} \033[0m").format(self.trial_steps, self.env.get_deadline(self))
            self.failed_trials = self.failed_trials +1
        else:
            #print("\033[0;32mPASS Steps To Complete = {}, Deadline = {} \033[0m").format(self.trial_reward, self.trial_steps, self.env.get_deadline(self))
            self.passed_trials = self.passed_trials +1
        #draw_Q_table(self)
        self.trail_rewards.append(self.trial_reward)

        self.last_state = None #Memoize last state
        self.last_action = None #Memoize last action
        
        self.trial_reward = 0
        self.trial_steps =  0
        # TODO: Prepare for a new trip; reset any variables here, if required

    def choose_action(self, state):
        epsilon = self.get_epsilon_decay()
        if random.random() < epsilon: #Shake the dice
            direction = self.random_direction() #Used for exploring the enviroment
        else:
            direction =  self.greedy_direction(state) #Pick the best value for Q - Greedy action
        return direction

    def get_epsilon_decay(self):
        return float(self.epsilon) / (self.trial_steps + float(self.epsilon))

    def random_direction(self):
        random_direction = random.choice(self.directions)
        return random_direction

    def greedy_direction(self, state):
        q = [self.get_Q_val(state, a) for a in self.directions]
        maxQ = max(q)
        if q.count(maxQ) > 1: #existing value
            best = [i for i in range(len(self.directions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)
        return self.directions[i]

    def get_Q_val(self, state, action):
        q_val = self.q.get((state, action), 0.0)
        return q_val

    def learn_Q(self, prev_state, action, reward, value):
        print("Learning:", value)
        prev_val = self.q.get((state, action), None)
        if prev_val is None:
             self.q[(state, action)] = reward
        else:
             self.q[(state, action)] = prev_val + self.learning_rate * (value - prev_val)

    def discounted_estimate_next_state(self, reward):
        max_q_new = max([self.get_Q_val(self.last_state, direction) for direction in self.directions])  
        return reward+self.discount*max_q_new

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        ''' Important variables for state are light, oncoming, and left, right is not required as 
            driver can turn right on red without penalty '''
        self.current_state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])
        self.trial_steps = t #used only for performance analysis.
        print("Current State", self.current_state)
        # TODO: Select action according to your policy
        action = self.choose_action(self.current_state)
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.trial_reward += reward
        print("Previous State:", self.last_state)
        
        # TODO: Learn policy based on state, action, reward
        if self.last_state is not None:
            self.learn_Q(self.current_state, action, reward, self.discounted_estimate_next_state(reward))

        #update memoized last state and action.
        self.last_state = self.current_state
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(
        #   deadline, inputs, action, reward)  # [debug]
        
def draw_Q_table(agent):
    #print learned states
    for state, action in agent.q:
        print("State: (Next Waypoint: {}, Light: {}, Oncoming: {}, Left: {})  Action: {}, Reward {}").format(state[0], state[1], state[2], state[3], action, agent.get_Q_val(state, action))
    print("\n")

def draw_reward_chart(chart_data_array):
    print(chart_data_array)
    plt.plot(chart_data_array)
    plt.show()
    #print("draw_reward_chart called", chart_data_array)

def draw_pass_chart(chart_data_array):
    print(chart_data_array)
    plt.plot(chart_data_array)
    plt.show()
    #print("draw_reward_chart called", chart_data_array)

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

    #draw_reward_chart(self.trail_rewards)
    pass_rate = float(a.passed_trials)/ float(len(a.trail_rewards)+1)
    print("Pass Rate:", pass_rate)

if __name__ == '__main__':
    run()
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

logging = True

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.directions = [None, 'forward', 'left', 'right']
        self.q = {}
        self.epsilon = 0.1
        self.learning_rate = 0.2
        self.discount = 0.9
        self.last_state = None #Memoize last state
        self.last_action = None #Memoize last action
        self.trial_reward = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        if logging == True:
            print("Total trial reward:", self.trial_reward)
        self.last_state = None #Memoize last state
        self.last_action = None #Memoize last action
        self.trial_reward = 0
        # TODO: Prepare for a new trip; reset any variables here, if required

    def choose_action(self, state):
        if random.random() < self.epsilon:
            direction = random.choice(self.directions)
        else:
            q = [self.get_Q(state, a) for a in self.directions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.directions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            direction = self.directions[i]
        return direction

    def get_Q(self, state, action):
        q_val = self.q.get((state, action), 0.0)
        return q_val

    def learn_Q(self, state, action, reward, value):
        prev_val = self.q.get((state, action), None)
        if prev_val is None:
             self.q[(state, action)] = reward
        else:
             self.q[(state, action)] = prev_val + self.learning_rate * (value - prev_val)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # Important variables for state are light, oncoming, and left,  right is not required
        # driver can turn right on red.
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])

        # TODO: Select action according to your policy
        action = self.choose_action(self.state);

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        #if logging == True:
        #    print("State:", self.state)
        #    for direction in self.directions:
        #        print("Direction: {}  Q Val: {}").format(direction, self.get_Q(self.state, direction))
        #    print("------------------------------")

        if self.last_state is not None:
            max_q_new = max([self.get_Q(self.last_state, direction) for direction in self.directions])
            self.learn_Q(self.state, action, reward, reward + self.discount*max_q_new)

        #update memoized last state and action.
        self.last_state = self.state
        self.last_action = action

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(
        #   deadline, inputs, action, reward)  # [debug]
        self.trial_reward += reward

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
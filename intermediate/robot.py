####################################
#      YOU MAY EDIT THIS FILE      #
# MOST OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

# Imports from this project
import constants, environment
import config

# Configure matplotlib for interactive mode
plt.ion()


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self, forward_kinematics):
        # Give the robot the forward kinematics function, to calculate the hand position from the state
        self.forward_kinematics = forward_kinematics
        # A list of visualisations which will be displayed on the right-side of the window
        self.planning_visualisation_lines = []
        self.model_visualisation_lines = []
        # The position of the robot's base
        self.robot_base_pos = np.array(constants.ROBOT_BASE_POS)
        # The goal state
        self.goal_state = 0
        self.n_episodes = 0
        self.eps_count = 0
        self.inputs = []
        self.outputs = []
        self.dyn_model = None

    # Reset the robot at the start of an episode
    def reset(self):
        self.eps_count = 0

    # Give the robot access to the goal state
    def set_goal_state(self, goal_state):
        self.goal_state = goal_state

    # Function to get the next action in the plan
    def select_action(self, state):
        # For now, a random action
        if self.n_episodes < 100:
            action = np.random.uniform(low=-constants.MAX_ACTION_MAGNITUDE, high=constants.MAX_ACTION_MAGNITUDE, size=2)
            if self.eps_count == 5:
                self.n_episodes += 1
                return action, True
            self.eps_count += 1
            return action, False
        elif len(self.inputs) == 100:
            print("start training")
            xs = torch.stack(self.inputs, 0)
            ys = torch.stack(self.outputs, 0)
            model = nn.Sequential(
                nn.Linear(4, 32, bias=True), nn.ReLU(True),
                nn.Linear(32, 16, bias=True), nn.ReLU(True),
                nn.Linear(16, 2, bias=True), nn.ReLU(True)
            )
            optimizer = optim.Adamax(model.parameters(), lr=0.001)
            while True:
                model.train()
                y_pred = model(xs)
                loss = nn.functional.mse_loss(y_pred, ys)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if loss < 1e-5:
                    break

            self.dyn_model = model
            print("end training")
            return action, True

    # Function to add a transition to the buffer
    def add_transition(self, state, action, next_state):
        self.inputs.append(torch.cat((torch.Tensor(state), torch.Tensor(action)), 0))
        self.outputs.append(torch.Tensor(next_state))

    def dynamics_train(self, environment: environment.Environment):
        n_episodes, episode_l = 100, 5
        for ep in range(n_episodes):
            environment.reset()
            for i in range(episode_l):
                state = np.copy(environment.state)
                action = np.random.uniform(-constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE, size=2)
                environment.step(action)
                self.add_transition(state, action, environment.state)
        

# The VisualisationLine class enables us to store a line segment which will be drawn to the screen
class VisualisationLine:
    # Initialise a new visualisation (a new line)
    def __init__(self, x1, y1, x2, y2, colour, width):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.colour = colour
        self.width = width

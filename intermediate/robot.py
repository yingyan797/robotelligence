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
from collections import deque

# Imports from this project
import constants, environment, config


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
        self.sequence = []
        self.replay_buffer = deque([], 65536)

    # Reset the robot at the start of an episode
    def reset(self):
        self.eps_count = 0
        self.n_episodes += 1

    # Give the robot access to the goal state
    def set_goal_state(self, goal_state):
        self.goal_state = goal_state

    # Function to get the next action in the plan
    def select_action(self, current_state):
        # For now, a random action
        if self.sequence:
            if len(self.sequence) > 1:
                return self.sequence.pop(0), False
            action = self.sequence.pop()
            return action, True
        
        if self.n_episodes < config.RAND_NEPS:
            action = np.random.uniform(low=-constants.MAX_ACTION_MAGNITUDE, high=constants.MAX_ACTION_MAGNITUDE, size=2)
            if self.eps_count == config.RAND_EPL:
                return action, True
            self.eps_count += 1
            return action, False
                
        def train_model(inputs, targets):
            self.dyn_model.train()
            # Forward pass: compute predicted next states
            predictions = self.dyn_model.forward(inputs)
            # Compute the loss
            loss = nn.functional.mse_loss(predictions, targets)
            # Backward pass and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()

        if self.n_episodes == config.RAND_NEPS:
            print("start training on random explore")
            xs = torch.stack([entry[0] for entry in self.replay_buffer], 0)
            ys = torch.stack([entry[1] for entry in self.replay_buffer], 0)
            self.dyn_model = nn.Sequential(
                nn.Linear(4, 32, bias=True), nn.ReLU(True),
                nn.Linear(32, 16, bias=True), nn.ReLU(True),
                nn.Linear(16, 2, bias=True), nn.ReLU(True)
            )
            self.optimizer = optim.Adamax(self.dyn_model.parameters(), lr=0.001)
            n_iter = 1
            while n_iter < 1000:
                train_model(xs, ys)
                n_iter += 1

            print(f"end training in {n_iter} epochs")
        elif (self.n_episodes - config.RAND_NEPS) % config.TRAIN_INTV == 0:
            import random
            for _ in range(config.N_BATCHES):
                inputs, targets = random.sample(self.replay_buffer, config.BATCH_SIZE)
                train_model(inputs, targets)

        def eucl_reward(state):
            joint_pos = self.forward_kinematics(state)
            hand_pos = joint_pos[2]
            goal = np.array(self.goal_state)
            diff = goal - hand_pos
            return -np.sqrt(np.dot(diff, diff))
        def term_reward(actions):
            state = torch.Tensor(current_state)
            for action in actions:
                state = self.dyn_model(torch.cat(state, action))
            return eucl_reward(state)
        def iterpath_visual(actions, color):
            state = current_state
            hand_pos = self.forward_kinematics(state)[2]
            for act in actions:
                state = self.dyn_model(torch.cat(state, act))
                next_pos = self.forward_kinematics(state)[2]
                self.planning_visualisation_lines.append(VisualisationLine(hand_pos[0], hand_pos[1], next_pos[0], next_pos[1], color, 3e-3))
                hand_pos = next_pos

        if eucl_reward(current_state) > -0.01:
            return np.zeros(2), True
        
        from torch.distributions import Normal
        act_mean, act_std = torch.zeros(size=(config.PATH_L, 2)), torch.ones(size=(config.PATH_L, 2))
        for i in range(config.CEM_ITER):
            policy = Normal(act_mean, act_std)
            all_actions = policy.sample(sample_shape=(config.N_PATHS,))
            all_rewards = torch.Tensor([term_reward(seq) for seq in all_actions])
            top_indices = all_rewards.topk(config.K_ELITE).indices
            elite = all_actions.index_select(0, top_indices)
            act_mean = elite.mean(0)
            act_std = elite.std(0)
            color_step = i/config.CEM_ITER
            iterpath_visual(act_mean, (50+150*color_step, 50, 255-120*color_step))
        
        self.sequence = [act.numpy() for act in act_mean]  

    # Function to add a transition to the buffer
    def add_transition(self, state, action, next_state):
        x = torch.cat((torch.Tensor(state), torch.Tensor(action)), 0)
        y = torch.Tensor(next_state)
        self.replay_buffer.append((x, y))
        

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

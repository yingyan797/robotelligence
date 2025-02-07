####################################
#      YOU MAY EDIT THIS FILE      #
# MOST OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np
from matplotlib import pyplot as plt

# Imports from this project
import constants, demonstrator, config, torch, random
import torch.nn as nn
from collections import deque

# Configure matplotlib for interactive mode
plt.ion()


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self, forward_kinematics):
        # Get the forward kinematics function from the environment
        self.forward_kinematics = forward_kinematics
        # A list of visualisations which will be displayed in the middle (planning) and right (policy) of the window
        self.planning_visualisation_lines = []
        self.policy_visualisation_lines = []
        self.n_steps = 0
        self.n_episodes = 0
        self.n_demos = 1
        self.min_dist = None
        with open("final/record.csv", "w") as f:
            f.write("")
        # self.replay_buffer = deque([], 65536)

    # Reset the robot at the start of an episode
    def reset(self):
        self.planning_visualisation_lines = []
        self.policy_visualisation_lines = []
        self.min_dist = None
        self.n_steps = 0
        self.n_episodes += 1
        if self.n_episodes == 5:
            self.n_episodes = 0
            self.n_demos += 1

    # Get the demonstrations
    def get_demos(self, demonstrator: demonstrator.Demonstrator):
        policy = nn.Sequential(
            nn.Linear(2, 16, True), nn.Sigmoid(),
            nn.Linear(16, 128, True), nn.Sigmoid(),
            nn.Linear(128, 32, True), nn.Sigmoid(),
            nn.Linear(32, 2, True),
        )
        optimizer = torch.optim.Adamax(policy.parameters(), lr=0.01)
        dataset = []
        for i in range(self.n_demos):
            print(f"demo {i+1}/{self.n_demos}")
            demos = demonstrator.generate_demonstration()
            dataset = []
            color_step = 200*i/(self.n_demos)
            for j in range(len(demos)-1):
                p1, p2 = self.forward_kinematics(demos[j][0])[2], self.forward_kinematics(demos[j+1][0])[2]
                self.planning_visualisation_lines.append(VisualisationLine(p1[0], p1[1], p2[0], p2[1], (55+color_step, 50, 255-color_step), width=3e-3))
            dataset.extend([(torch.Tensor(state), torch.Tensor(action)) for state, action in demos])
        
        for i in range(config.N_BATCHES):
            xs = torch.stack([entry[0] for entry in dataset], 0)
            ys = torch.stack([entry[1] for entry in dataset], 0)
            policy.train()
            # Forward pass: compute predicted next states
            predictions = policy.forward(xs)
            # Compute the loss
            loss = nn.functional.mse_loss(predictions, ys)
            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.policy = policy

    # Get the next action
    def select_action(self, state):
        self.n_steps += 1
        pos = self.forward_kinematics(state)
        done = False
        dist = np.linalg.norm(pos[2]-np.array(constants.GOAL_STATE))
        if self.min_dist is None or dist < self.min_dist:
            self.min_dist = dist
        if dist < 0.01:
            action = np.zeros(2)
            done = True
        else:
            self.policy.eval()
            with torch.no_grad():
                # Forward pass
                action = self.policy(torch.Tensor(state)).numpy()
            done = self.n_steps == constants.CEM_PATH_LENGTH
        if done and self.n_demos <= 10:
            with open("final/record.csv", "a") as f:
                f.write(f"{self.n_demos},{self.n_episodes},{self.min_dist}\n")
        return action, done

# The VisualisationLine class enables us to store a line segment which will be drawn to the screen
class VisualisationLine:
    # Initialise a new visualisation (a new line)
    def __init__(self, x1, y1, x2, y2, colour=(255, 255, 255), width=0.01):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.colour = colour
        self.width = width

def statistics(fn="final/record.csv"):
    with open(fn, "r") as f:
        stats, exp = [], []
        while True:
            line = f.readline()
            if not line:
                stats.append(np.array(exp))
                xs = range(1,len(stats)+1)
                ys = np.array(stats)[:,:,2]
                ym = np.mean(ys, 1)
                plt.title("Minimum distance to goal vs. demonstration number")
                plt.ylabel("min distance")
                plt.xlabel("Number of demos")
                plt.xticks(range(1,11))
                for x in xs:
                    plt.scatter([x]*ys.shape[1], ys[x-1], c="b")
                plt.plot(xs, ym, c="r")
                plt.savefig("final/stats.png")
                return xs,ys
            entry = [float(c) for c in line.strip().split(",")]
            if exp and entry[0] > exp[-1][0]:
                stats.append(np.array(exp))
                exp = []
            exp.append(entry)

# print(statistics())

####################################
#      YOU MAY EDIT THIS FILE      #
# MOST OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np
import torch, json
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
        self.planning_visualisation_lines = []
        self.model_visualisation_lines = []

    # Give the robot access to the goal state
    def set_goal_state(self, goal_state):
        self.goal_state = goal_state

    # Function to get the next action in the plan
    def select_action(self, current_state):
        # For now, a random action
        # print(len(self.sequence), self.n_episodes)
        if self.n_episodes < config.RAND_NEPS:
            if config.EXPLORE_MODE == "write":
                action = np.random.uniform(low=-constants.MAX_ACTION_MAGNITUDE, high=constants.MAX_ACTION_MAGNITUDE, size=2)
                if self.eps_count == config.RAND_EPL:
                    return action, True
                self.eps_count += 1
                return action, False
            print("Use previous random explore")
            self.n_episodes = config.RAND_NEPS
            with open("intermediate/explore.json", 'r') as f:
                explore = json.load(f)
                self.replay_buffer.extend([(torch.Tensor(trans[0]), torch.Tensor(trans[1])) for trans in explore])
                
        def train_model():
            import random
            entries = random.sample(self.replay_buffer, k=min(config.BATCH_SIZE, len(self.replay_buffer)))
            xs = torch.stack([entry[0] for entry in entries], 0)
            ys = torch.stack([entry[1] for entry in entries], 0)
            self.dyn_model.train()
            # Forward pass: compute predicted next states
            predictions = self.dyn_model.forward(xs)
            # Compute the loss
            loss = nn.functional.mse_loss(predictions, ys)
            # Backward pass and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        def dyn_predict(state, action):
            self.dyn_model.eval()
            with torch.no_grad():
                # Forward pass
                return self.dyn_model(torch.cat((state, action)))
        def eucl_reward(state):
            joint_pos = self.forward_kinematics(state)
            hand_pos = joint_pos[2]
            goal = np.array(self.goal_state)
            diff = goal - hand_pos
            return -np.sqrt(np.dot(diff, diff))
        def term_reward(actions):
            state = torch.Tensor(current_state)
            for action in actions:
                state = dyn_predict(state, action)
            return eucl_reward(state)
        def iterpath_visual(actions, color):
            state = torch.Tensor(current_state)
            hand_pos = self.forward_kinematics(state)[2]
            for act in actions:
                state = dyn_predict(state, act)
                next_pos = self.forward_kinematics(state)[2]
                self.planning_visualisation_lines.append(VisualisationLine(hand_pos[0], hand_pos[1], next_pos[0], next_pos[1], color, 3e-3))
                hand_pos = next_pos

        if eucl_reward(current_state) > -0.01:
            action = np.zeros(2)
            done = True
        else:
            def cem_plan(pl):
                from torch.distributions import Normal
                std = constants.MAX_ACTION_MAGNITUDE/2
                act_mean, act_std = torch.zeros(size=(pl, 2)), torch.tensor([[std, std] for _ in range(pl)])
                for i in range(config.CEM_ITER):
                    policy = Normal(act_mean, act_std)
                    all_actions = policy.sample(sample_shape=(config.N_PATHS,)).clamp(-constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
                    all_rewards = torch.Tensor([term_reward(seq) for seq in all_actions])
                    top_indices = all_rewards.topk(config.K_ELITE).indices
                    elite = all_actions.index_select(0, top_indices)
                    act_mean = elite.mean(0)
                    act_std = torch.maximum(elite.std(0), torch.tensor(1e-3))
                    color_step = i/config.CEM_ITER
                    iterpath_visual(act_mean, (50+150*color_step, 50, 255-120*color_step))
                self.sequence = [act.numpy() for act in act_mean]

            def draw_loss(loss, mode="init"):
                plt.clf()
                plt.title("MSE training loss over minibatches")
                plt.xlabel("Batch number")
                plt.ylabel("MSE loss")
                plt.plot(range(len(loss)), loss)
                plt.savefig(f"intermediate/{mode}_loss.png")

            action_i = self.eps_count
            if self.eps_count == 0:
                if self.n_episodes == config.RAND_NEPS:
                    # Create dynamic network after random explore
                    if config.EXPLORE_MODE == "write":
                        print(f"Random explore saved as file: {len(self.replay_buffer)} transitions")
                        with open("intermediate/explore.json", 'w') as f:
                            json.dump([(x.numpy().tolist(), y.numpy().tolist()) for x,y in self.replay_buffer], f)

                    print("start training on random explore")
                    self.dyn_model = nn.Sequential(
                        nn.Linear(4, 32, bias=True), nn.ReLU(True),
                        nn.Linear(32, 128, bias=True), nn.ReLU(True),
                        nn.Linear(128, 64, bias=True), nn.ReLU(True),
                        nn.Linear(64, 16, bias=True), nn.ReLU(True),
                        nn.Linear(16, 2, bias=True)
                    )
                    self.optimizer = optim.Adamax(self.dyn_model.parameters(), lr=5e-3)
                    if False:
                        n_iter = 1
                        all_loss = []
                        for i in range(config.N_BATCHES):
                            l = train_model()
                            if l < 1e-5:
                                break
                            all_loss.append(l)
                            n_iter += 1
                        print(f"Training end at loss {all_loss[-1]}")
                        draw_loss(all_loss)
                        print(f"end training in {n_iter} epochs")
                    # self.replay_buffer.clear()
                elif (self.n_episodes - config.RAND_NEPS) % config.TRAIN_INTV == 0:
                    # print("Retrain model")
                    all_loss = []
                    for i in range(config.N_BATCHES):
                        l = train_model()
                        if l < 1e-5:
                            break
                        all_loss.append(l)
                    # draw_loss(all_loss, self.n_episodes)

                cem_plan(config.PATH_L)
            elif config.LOOP_MODE == "closed":
                self.planning_visualisation_lines = []
                cem_plan(config.PATH_L - self.eps_count)
                action_i = 0
            
            print(len(self.sequence), action_i)
            action = self.sequence[action_i]
            done = self.eps_count == config.PATH_L-1
        if done:
            dist = -eucl_reward(current_state)
            with open(f"intermediate/{config.LOOP_MODE}_record.csv", "a") as f:
                f.write(str(dist)+"\n")
                
        self.eps_count += 1
        return action, done

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

def statistics():
    stats = ([], [])
    suf = ["open", "closed"]
    plt.title("Final distance to goal vs. episodes")
    plt.xlabel("Episode number")
    plt.ylabel("Final distance")
    for i in [0,1]:
        with open(f"intermediate/{suf[i]}_record.csv", "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                dist = float(line.strip())
                stats[i].append(dist)

        plt.plot(range(len(stats[i])), stats[i], label=f"{suf[i]} planning")
    plt.legend()
    
    plt.savefig("intermediate/perf.png")
    
statistics()
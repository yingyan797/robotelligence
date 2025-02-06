####################################
#      YOU MAY EDIT THIS FILE      #
# MOST OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np

# Imports from this project
import constants, environment, config

# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self):
        # The environment
        self.environment:environment.Environment = None
        # A list of visualisations which will be displayed on the right-side of the window
        self.visualisation_lines = []
        self.visualisation_circles = []

    # Reset the robot at the start of an episode
    def reset(self):
        self.sequence = []
        self.visualisation_lines = []

    # Give the robot access to the environment's dynamics
    def give_environment_access(self, environment):
        self.environment = environment

    # Function to get the next action in the plan
    def select_action(self, current_state):
        # Initially, we just return a random action, but you should implement a planning algorithm
        # action = np.random.uniform(low=-constants.MAX_ACTION_MAGNITUDE, high=constants.MAX_ACTION_MAGNITUDE, size=2)
        # episode_done = False
        # return action, episode_done
        import torch
        from torch.distributions.normal import Normal

        def eucl_reward(state):
            joint_pos = self.environment.get_joint_pos_from_state(state)
            hand_pos = joint_pos[2]
            goal = np.array(self.environment.goal_state)
            diff = goal - hand_pos
            return -np.sqrt(np.dot(diff, diff))
        
        if eucl_reward(current_state) > -0.01:
            return np.zeros(2), True
        
        def term_reward(actions):
            state = current_state
            for action in actions:
                state = self.environment.dynamics(state, action.numpy())
            return eucl_reward(state)     
               
        def iterpath_visual(actions, color):
            state = current_state
            hand_pos = self.environment.get_joint_pos_from_state(state)[2]
            for act in actions:
                state = self.environment.dynamics(state, act.numpy())
                next_pos = self.environment.get_joint_pos_from_state(state)[2]
                self.visualisation_lines.append(VisualisationLine(hand_pos[0], hand_pos[1], next_pos[0], next_pos[1], color, 3e-3))
                hand_pos = next_pos
        
        if not self.sequence:
            act_mean, act_std = torch.zeros(size=(config.PATH_L, 2)), torch.ones(size=(config.PATH_L, 2))
            for i in range(config.N_ITER):
                policy = Normal(act_mean, act_std)
                all_actions = policy.sample(sample_shape=(config.N_PATHS,)).clamp(-constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
                all_rewards = torch.Tensor([term_reward(seq) for seq in all_actions])
                top_indices = all_rewards.topk(config.K_ELITE).indices
                elite = all_actions.index_select(0, top_indices)
                act_mean = elite.mean(0)
                act_std = elite.std(0)
                color_step = i/config.N_ITER
                iterpath_visual(act_mean, (50+150*color_step, 50, 255-120*color_step))
            
            self.sequence = [act.numpy() for act in act_mean]

    
        if len(self.sequence) > 1:
            return self.sequence.pop(0), False
        action =  self.sequence.pop()
        return action, True

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


# The VisualisationCircle class enables us to a circle which will be drawn to the screen
class VisualisationCircle:
    # Initialise a new visualisation (a new circle)
    def __init__(self, x, y, radius, colour=(255, 255, 255)):
        self.x = x
        self.y = y
        self.radius = radius
        self.colour = colour
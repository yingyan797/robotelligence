#########################
# DO NOT EDIT THIS FILE #
#########################

# Graphics
WINDOW_MARGIN = 10
WINDOW_HEADER = 50
ROBOT_BASE_COLOUR = (200, 200, 200)
ROBOT_JOINT_COLOUR = (50, 100, 200)
ROBOT_LINK_COLOUR = (100, 130, 170)
ROBOT_HAND_COLOUR = (50, 180, 50)
GOAL_COLOUR = (200, 50, 50)
OBSTACLE_COLOUR = (160, 140, 150)

# Environment
ROBOT_LINK_LENGTHS = [0.35, 0.27, 0.23]
ROBOT_LINK_WIDTHS = [0.02, 0.015, 0.01]
ROBOT_BASE_LENGTH = 0.1
ROBOT_BASE_WIDTH = 0.03
ROBOT_JOINT_RADIUS = 0.02
ROBOT_HAND_RADIUS = 0.02
GOAL_RADIUS = 0.02
OBSTACLE_POS = [0.65, 0.7]
OBSTACLE_RADIUS = 0.1
INIT_STATE = [0.0, -0.6]
ROBOT_BASE_POS = [0.3, 0.1]
GOAL_STATE = [0.6, 0.5]
MAX_ACTION_MAGNITUDE = 0.05

# CEM parameters for the demonstrations
CEM_NUM_ITER = 10
CEM_NUM_PATHS = 500
CEM_PATH_LENGTH = 60
CEM_NUM_ELITES = 30

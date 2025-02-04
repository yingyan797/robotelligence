##########################
# YOU MAY EDIT THIS FILE #
##########################

# The window width and height in pixels, for both the "environment" window and the "planning" window.
# If you wish, you can modify this according to the size of your screen.
WINDOW_SIZE = 600

# The frame rate for pygame, which determines how quickly the program runs.
# Specifically, this is the number of time steps per second that the robot will execute an action in the environment.
# You may wish to slow this down to observe the robot's movement, or speed it up to run large-scale experiments.
FRAME_RATE = 10

# You may want to add your own configuration variables here, depending on the algorithm you implement.
TRAIN_ITER = 400
EXPLORE_MODE = "read"
LOOP_MODE = "closed"
RAND_EPL = 5
RAND_NEPS = 100
TRAIN_INTV = 1
CEM_ITER = 10
N_PATHS = 15
PATH_L = 30
K_ELITE = 3
N_BATCHES = 500
BATCH_SIZE = 256
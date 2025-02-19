#########################
# DO NOT EDIT THIS FILE #
#########################

# Imports from external libraries
import time
import numpy as np
import pygame

# Imports from this project
from environment import Environment
from robot import Robot
from graphics import Graphics


# Set the numpy random seed
seed = int(time.time())
np.random.seed(seed)
# Initialize Pygame
pygame.init()
# Create an environment (the "physical" world) and reset the episode
environment = Environment()
environment.reset()
# Create a robot (the robot's "brain" making the decisions) and reset the episode
robot = Robot()
robot.reset()
# In this exercise, we give the robot access to the environment
robot.give_environment_access(environment)
# Create a graphics (this will create a window and draw on the window)
graphics = Graphics()

# Main loop
running = True
while running:
    # Check for any user input
    for event in pygame.event.get():
        # Closing the window
        if event.type == pygame.QUIT:
            running = False
    # Robot selects an action, and decides whether it has finished the episode
    action, episode_done = robot.select_action(environment.state)
    # Robot executes the action in the environment
    environment.step(action)
    # Draw the environment, and any visualisations, on the window
    graphics.draw(environment, robot.visualisation_lines, robot.visualisation_circles)
    # Check if the robot has finished its episode
    if episode_done:
        # If the episode has finished, create a new environment and reset the robot's episode
        environment = Environment()
        robot.reset()
        robot.give_environment_access(environment)
        episode_done = False
# If we have broken out of the main loop, quite pygame and end the program
pygame.quit()

import gymnasium as gym
import numpy as np
import pygame
import argparse
import time
from car_racing import CustomCarRacing
import scipy
import random
import warnings
from matplotlib import pyplot as plt
from utils import get_state_pose_surface, get_state_position_surface, transform_position, STATE_FROM_OCCUPANCY, OCCUPANCY_FROM_STATE, OCCUPANCY_FROM_STATE_FACTOR

import threading
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")
warnings.filterwarnings('ignore', message='RuntimeWarning: divide by zero encountered in true_divide')
import profile
from perception import Perception
from controllers import HumanController, MPCController

NULL_ACTION = np.zeros((3))

def play_single_game(controller, environment, max_iterations=None):
    quit = False

    environment.reset()
    controller.reset()
    perception = Perception()

    # Burn startup frames
    for ii in range(6*10):
        state, reward, terminated, truncated, info = environment.step(NULL_ACTION)
        if environment.render_mode == 'human':
            environment.show_graphics()

    iteration_counter = 0
    game_score = 0
    while max_iterations == None or iteration_counter < max_iterations:
        perception.compute_perception(state)
        controller.set_perception(perception)

        action = controller.get_next_action()
        quit = controller.quit
        restart = controller.restart
        state, reward, terminated, truncated, info = environment.step(action)
        game_score += reward
        
        # Display things
        if environment.render_mode == 'human':
            if type(controller) == MPCController:
                all_states, u = controller.model_system_output(controller.action_sequence)
                colors = [(int(255 - ii * 245 / all_states.shape[0]), int(255 - ii * 245 / all_states.shape[0]), 255, 200) for ii in range(all_states.shape[0])]
                surfaces = [get_state_pose_surface(all_states[ii][:3], colors[ii]) for ii in range(all_states.shape[0])]
                for surface in surfaces:
                    environment.draw_surface(surface)
            environment.draw_surface(get_state_position_surface(transform_position(STATE_FROM_OCCUPANCY, perception.target_pixel_far)))
            environment.draw_surface(get_state_position_surface(transform_position(STATE_FROM_OCCUPANCY, perception.road_mean)))
            environment.draw_surface(get_state_position_surface(transform_position(STATE_FROM_OCCUPANCY, perception.road_middle)))
            environment.show_graphics()
        
        if terminated or truncated or restart or quit:
            break
        iteration_counter += 1
    return quit, game_score
    

def play_multiple_games(controller, environment, max_games=None, max_iterations=None):
    quit = False
    game_counter = 0
    while not quit and (max_games == None or game_counter < max_games):
        quit, game_score  = play_single_game(controller, environment, max_iterations)
        print(game_score)
        game_counter += 1
    environment.close()

def run_and_visualize_model_control_sequence(controller, environment, control_sequence):
    perception = Perception()
    quit = False
    environment.reset()
    controller.reset()
    # Burn startup frames
    for ii in range(6*10):
        state, reward, terminated, truncated, info = environment.step(NULL_ACTION)
        perception.compute_perception(state)
        environment.show_graphics()

    while len(control_sequence) > 1:
        controller.set_perception(perception)
        all_states, u = controller.model_system_output(control_sequence)
        colors = [(int(255 - ii * 245 / all_states.shape[0]), int(255 - ii * 245 / all_states.shape[0]), 255, 200) for ii in range(all_states.shape[0])]
        surfaces = [get_state_pose_surface(all_states[ii][:3], colors[ii]) for ii in range(all_states.shape[0])]
        for surface in surfaces:
            environment.draw_surface(surface)
        environment.show_graphics()
        action = control_sequence[0]
        control_sequence = control_sequence[1:]

        state, reward, terminated, truncated, info = environment.step(action)
        
        perception.compute_perception(state)
    input()
    environment.close()

def profile_main_control_loop(controller, environment, max_games, max_iterations):
    profile.runctx('main_control_loop(controller, environment, max_games, max_iterations)', globals(), locals())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', choices=['human', 'MPC'], default='human', help='Select the control for which to run the car.')
    parser.add_argument('--no_gui', action='store_true', help='Do not render the game.')
    
    args = parser.parse_args()
    controller = None
    if args.controller == 'human':
        controller = HumanController()
    if args.controller == 'MPC':
        controller = MPCController(0.02)

    render_mode = 'state_pixels' if args.no_gui else 'human'
    environment = CustomCarRacing(domain_randomize=False, continuous=True, render_mode=render_mode)
    # Sim Anneal High Temp
    # controller.unpack_parameters({"horizon": 19.238501832829535, "recompute_iterations": 7.647025347497353, "optimal_speed": 60, "optimal_speed_cost_factor": 46.66663923905215, "turning_cost_factor": 0.4434195800385791, "angled_towards_far_target_cost_factor": 45.18820097564773, "lateral_displacement_from_road_middle_cost_factor": 7.741714306874302, "distance_road_mean_cost_factor": 50, "distance_road_road_cost_factor": 50})
    # Sim Anneal Low Temp
    # controller.unpack_parameters({"horizon": 21.588459631075054, "recompute_iterations": 8.769030553285486, "optimal_speed": 49.576350534251546, "optimal_speed_cost_factor": 20.08912881278177, "turning_cost_factor": 0.3892679628438823, "angled_towards_far_target_cost_factor": 27.22413481578229, "lateral_displacement_from_road_middle_cost_factor": 4.076762294779184, "distance_road_mean_cost_factor": 31.93413601124484, "distance_road_road_cost_factor": 43.59849305096547})
    # Hill Climb
    # controller.unpack_parameters({"horizon": 28.59906812777642, "recompute_iterations": 6.614257510928882, "optimal_speed": 56.990774880513726, "optimal_speed_cost_factor": 7.507178675178933, "turning_cost_factor": 0.31580689429913594, "angled_towards_far_target_cost_factor": 6.791812014089963, "lateral_displacement_from_road_middle_cost_factor": 0, "distance_road_mean_cost_factor": 15.984073066904992, "distance_road_road_cost_factor": 0})
    # Genetic
    controller.unpack_parameters({"horizon": 15.11264034207368, "recompute_iterations": 6.85010427120707, "optimal_speed": 60, "optimal_speed_cost_factor": 10.098019388299235, "turning_cost_factor": 0.3559438168543909, "angled_towards_far_target_cost_factor": 21.389980817773143, "lateral_displacement_from_road_middle_cost_factor": 2.778813150661125, "distance_road_mean_cost_factor": 37.70884257299245, "distance_road_road_cost_factor": 39.700362426818174})
    environment.np_random = np.random.default_rng(521347812374)
    play_multiple_games(controller, environment, 1, 750)
    # profile_main_control_loop(controller, environment, 1, 100)
    # run_and_visualize_model_control_sequence(controller, environment, [(0, 1, 0)] * 20 + [(-1, 0, 0)] * 30 + [(1, 0, 0)] * 15 + [(0, 0, 0.1)] * 30)

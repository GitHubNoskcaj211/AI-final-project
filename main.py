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
import threading
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")
warnings.filterwarnings('ignore', message='RuntimeWarning: divide by zero encountered in true_divide')

NULL_ACTION = np.zeros((3))
MINIMUM_ROAD_COLOR = np.array([99, 99, 99])
MAXIMUM_ROAD_COLOR = np.array([110, 110, 110])
START_CAR_INDEX_II = 66
END_CAR_INDEX_II = 77
START_CAR_INDEX_JJ = 45
END_CAR_INDEX_JJ = 51

OCCUPANCY_FROM_STATE_FACTOR = 2
OCCUPANCY_FROM_STATE = np.array([[0, -OCCUPANCY_FROM_STATE_FACTOR, 71.5], [OCCUPANCY_FROM_STATE_FACTOR, 0, 48], [0, 0, 1]])
STATE_FROM_OCCUPANCY = np.linalg.inv(OCCUPANCY_FROM_STATE)

def plot_all_states(all_states):
    plt.plot(all_states[:, 0], all_states[:, 1])
    ax = plt.gca()
    ax.set_xlim([-24, 24])
    ax.set_ylim([-12.25, 37.75])
    plt.show()

def transform_position(matrix, position):
    return (matrix @ np.array([[position[0]], [position[1]], [1]])).reshape(-1)[0:2]

def get_state_position_surface(position):
    occupancy_position = transform_position(OCCUPANCY_FROM_STATE, position)
    ii = int(occupancy_position[0] * 800 / 96)
    jj = int(occupancy_position[1] * 1000 / 96)
    if ii < 0 or ii >= 800 or jj < 0 or jj >= 1000:
        print('out of bounds!')
        return None
    surf = pygame.Surface((1000, 800), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    pygame.draw.rect(surf, (255, 255, 255, 125), pygame.Rect(jj - 2, ii - 2, 4, 4), 4)
    # surf = pygame.transform.flip(surf, False, True)
    return surf

def compute_road_grid(screen):
    # num_rows = 16
    # num_cols = 16
    # assert screen.shape[0] % num_rows == 0, f'Need to have an occupancy grid that fits evently to the number of rows: {screen.shape[0]}'
    # assert screen.shape[1] % num_cols == 0, f'Need to have an occupancy grid that fits evently to the number of columns: {screen.shape[1]}'
    
    valid_road_pixels = np.full((screen.shape[0], screen.shape[1]), False, dtype=np.bool8)
    valid_road_pixels[np.all(screen >= MINIMUM_ROAD_COLOR, axis=2) & np.all(screen <= MAXIMUM_ROAD_COLOR, axis=2)] = True
    valid_road_pixels[START_CAR_INDEX_II:END_CAR_INDEX_II, START_CAR_INDEX_JJ:END_CAR_INDEX_JJ] = True

    # road_grid_blocks = valid_road_pixels.reshape(num_rows, valid_road_pixels.shape[0] // num_rows, num_cols, valid_road_pixels.shape[1] // num_cols)
    # road_grid = np.mean(road_grid_blocks.astype(int), axis=(1,-1)) > 0.5
    # np.full((num_rows, num_cols), False, dtype=np.bool8)
    # disp_road_grid = np.repeat(np.repeat(road_grid.reshape((-1)), 96 / num_rows).reshape((-1, 96)), 96 / num_cols)

    # surf = pygame.surfarray.make_surface(valid_road_pixels.astype(int) * 255)
    # display.blit(surf, (0, 0))
    # pygame.display.update()

    return valid_road_pixels

def compute_speed(hull):
    return (hull.linearVelocity[0] ** 2 + hull.linearVelocity[1] ** 2) ** 0.5

def compute_angular_velocity(hull):
    return hull.angularVelocity

# def compute_target_pixel(road):
#     mean_road = np.mean(np.where(road == True), axis=1)
#     return mean_road

def compute_target_pixel(road):
    occupancy_car_position = transform_position(OCCUPANCY_FROM_STATE, np.array([0, 0]))
    road_positions = np.transpose(np.where(road == True))
    differences = road_positions - occupancy_car_position
    farthest_road_index = np.argmax(differences[0] ** 2 + differences[1] ** 2)
    farthest_road_position = road_positions[farthest_road_index]
    return farthest_road_position

class Perception:
    def __init__(self):
        self.road = None
        self.car_angular_velocity = None
        self.car_speed = None

    # This will give perception of the current system.
    def compute_perception(self, state):
        screen, car = state
        self.road = compute_road_grid(screen)
        self.car_angular_velocity = compute_angular_velocity(car.hull)
        self.car_speed = compute_speed(car.hull)
        self.target_pixel = compute_target_pixel(self.road)

class BaseController:
    def __init__(self):
       self.reset()

    def reset(self):
        self.restart = False
        self.quit = False
        self.perception = None

    def set_perception(self, perception: Perception):
        self.perception = perception

    def get_next_action(self) -> np.ndarray:
        raise NotImplementedError()

class HumanController(BaseController):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self.action = np.zeros((3))
        
    def register_input(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.action[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    self.action[0] = +1.0
                if event.key == pygame.K_UP:
                    self.action[1] = +1.0
                if event.key == pygame.K_DOWN:
                    self.action[2] = +1.0
                if event.key == pygame.K_RETURN:
                    self.restart = True
                if event.key == pygame.K_ESCAPE:
                    self.quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    self.action[0] = 0
                if event.key == pygame.K_RIGHT:
                    self.action[0] = 0
                if event.key == pygame.K_UP:
                    self.action[1] = 0
                if event.key == pygame.K_DOWN:
                    self.action[2] = 0

            if event.type == pygame.QUIT:
                self.quit = True
    
    def get_next_action(self) -> np.ndarray:
        self.register_input()
        return self.action

class RandomController(BaseController):
    def __init__(self, sequence_length: int, seed: int):
        pass # TODO

class MPCController(BaseController):
    def __init__(self, horizon: int, sample_time: float):
        super().__init__()
        self.horizon = horizon
        self.sample_time = sample_time
        self.reset()

    def reset(self):
        super().reset()
        self.c = 0
        self.action_sequence = np.zeros((10, 3)) #np.zeros((self.horizon, 3)) # TODO Fix this
    
    def model_system_output(self, u):
        all_states = np.empty((self.horizon, 5))
         # x position, y position, x velocity, y velocity, angular velocity
        current_state = np.array([0, 0, 0, self.perception.car_speed, self.perception.car_angular_velocity])
        for ii, u_t in enumerate(u):
            turn_control = u_t[0]
            throttle_control = u_t[1]
            brake_control = u_t[2]
            all_states[ii] = current_state
            is_state_on_road = self.is_state_on_road(current_state)
            input_speed_factor = 1 if is_state_on_road is not None is not is_state_on_road == True else 0.9
            vehicle_speed_factor = 1 if is_state_on_road is not None is not is_state_on_road == True else 0.9
            braking_factor = 0.9 * (1 - brake_control) + 0.1
            current_state[0] += current_state[2] * self.sample_time
            current_state[1] += current_state[3] * self.sample_time
            current_speed = (current_state[2] ** 2 + current_state[3] ** 2) ** 0.5 * braking_factor
            next_speed = current_speed * vehicle_speed_factor
            current_state[2] = next_speed * -np.sin(current_state[4])
            current_state[3] = next_speed * np.cos(current_state[4]) + throttle_control * input_speed_factor * self.sample_time
            current_state[4] = current_speed * -turn_control
        return all_states, u

    def is_state_on_road(self, state):
        occupancy_position = transform_position(OCCUPANCY_FROM_STATE, state[0:2])
        ii = int(occupancy_position[0])
        jj = int(occupancy_position[1])
        if not (ii >= 0 and ii < self.perception.road.shape[0] and jj >= 0 and jj < self.perception.road.shape[1]):
            return None
        return self.perception.road[ii, jj]

    def compute_min_distances_from_road(self, states):
        min_distances = np.empty(states.shape[0])
        for ii, state in enumerate(states):
            is_state_on_road = self.is_state_on_road(state)
            if is_state_on_road == None or is_state_on_road == True:
                min_distances[ii] = 0
                continue
            occupancy_car_position = transform_position(OCCUPANCY_FROM_STATE, state[0:2])
            road = self.perception.road
            differences = np.transpose(np.where(road == True)) - occupancy_car_position
            distances = differences[:, 0] ** 2 + differences[:, 1]
            min_distance = np.min(distances) / OCCUPANCY_FROM_STATE_FACTOR
            min_distances[ii] = min_distance
        return min_distances

    def system_cost(self, states, u):
        summed_distance_from_start = np.sum(np.power(np.sum(np.power(states[:, 0:2], 2), axis=1), 0.5))
        poor_progress_cost = 1 / (summed_distance_from_start + 1)
        
        distance_traveled = np.sum(np.power(np.sum(np.power(states[:-1, 0:2] - states[1:, 0:2], 2), axis=1), 0.5))
        immobile_cost = 1 / (distance_traveled + 1)
        
        summed_linear_speed = np.sum(np.power(np.sum(np.power(states[:, 0:2], 2), axis=1), 0.5))
        slow_cost = 1 / (summed_linear_speed + 1)

        summed_angular_speed = np.sum(np.abs(states[:, 4]))
        unstable_cost = 1 / (summed_angular_speed)

        summed_distances_from_road = np.sum(self.compute_min_distances_from_road(states))
        stay_on_road_cost = summed_distances_from_road

        actuation_cost = np.sum(np.power(u, 2))

        target_pixel = self.perception.target_pixel
        target_position = transform_position(STATE_FROM_OCCUPANCY, target_pixel)

        summed_distance_from_target = np.sum(np.power(np.sum(np.power(states[:, 0:2] - target_position, 2), axis=1), 0.5))
        target_cost = summed_distance_from_target
        return stay_on_road_cost + target_cost #1e-3 * actuation_cost + immobile_cost + unstable_cost + slow_cost + poor_progress_cost + stay_on_road_cost + target_cost

    def get_optimal_data_input_sequence(self):
        def optimization_function(u) -> float:
            u = u.reshape((self.horizon, 3))
            states, u = self.model_system_output(u)
            return self.system_cost(states, u)
        u_guess = np.zeros((self.horizon * 3))
        # Constrain every 3 values of u to be between -1 and 1 and all other values to be between 0 and 1
        constraints = scipy.optimize.LinearConstraint(np.eye(u_guess.shape[0]), np.tile([-1, 0, 0], self.horizon), np.ones(u_guess.shape[0]))
        result = scipy.optimize.minimize(optimization_function, u_guess, method='trust-constr', constraints=constraints, options={'maxiter': 100}) # TODO Make this faster to not limit
        return result.x.reshape((self.horizon, 3))

    def get_next_action(self) -> np.ndarray:
        if self.c > 5: # TODO Switch to multithreaded # TODO This relates to the sample_time = this / FPS (50)
            self.c = 0
            u = self.get_optimal_data_input_sequence()
            self.action_sequence = u
            # print(self.current_action)
            # plot_all_states(self.model_system_output(u)[0])
        current_action = self.action_sequence[self.c]
        self.c += 1
        return current_action

def main_control_loop(controller, environment):
    perception = Perception()
    quit = False
    while not quit:
        environment.reset()
        controller.reset()
        total_distance = 0
        # total_reward = 0.0
        # steps = 0
        # Burn startup frames
        for ii in range(6*10):
            state, reward, terminated, truncated, info = environment.step(NULL_ACTION)
            perception.compute_perception(state)
        while True:
            controller.set_perception(perception)
            action = controller.get_next_action()
            quit = controller.quit
            restart = controller.restart
            state, reward, terminated, truncated, info = environment.step(action)
            # print(perception.target_pixel)
            # print(transform_position(STATE_FROM_OCCUPANCY, perception.target_pixel))
            environment.draw_surface(get_state_position_surface(transform_position(STATE_FROM_OCCUPANCY, perception.target_pixel)))
            perception.compute_perception(state)
            # total_distance += perception.car_speed * 1/50   
            # print(total_distance)
            
            # total_reward += reward
            # if steps % 200 == 0 or terminated or truncated:
            #     print("\naction " + str([f"{x:+0.2f}" for x in action]))
            #     print(f"step {steps} total_reward {total_reward:+0.2f}")
            # steps += 1
            if terminated or truncated or restart or quit:
                break
    environment.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', choices=['human', 'MPC'], default='human', help='Select the control for which to run the car.')
    parser.add_argument('--render', type=bool, default=True, help='Whether or not to render the game.')
    parser.add_argument('--record-dir', type=str, default=None, help='A directory to record the run to.') # TODO Implement recording. TODO Implement playback.
    # env = gym.wrappers.RecordVideo(env, "videos", step_trigger=lambda x: x % 100 == 0)
    args = parser.parse_args()
    controller = None
    if args.controller == 'human':
        controller = HumanController()
    if args.controller == 'MPC':
        controller = MPCController(15, 0.1)

    render_mode = 'human' if args.render else None
    
    environment = CustomCarRacing(domain_randomize=False, continuous=True, render_mode=render_mode)#gym.make('car_racing/CustomCarRacing-v0', )

    main_control_loop(controller, environment)

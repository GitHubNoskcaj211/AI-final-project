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
from scipy.spatial.transform import Rotation as R
import threading
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")
warnings.filterwarnings('ignore', message='RuntimeWarning: divide by zero encountered in true_divide')
import profile

NULL_ACTION = np.zeros((3))
MINIMUM_ROAD_COLOR = np.array([99, 99, 99])
MAXIMUM_ROAD_COLOR = np.array([110, 110, 110])
START_CAR_INDEX_II = 66
END_CAR_INDEX_II = 77
START_CAR_INDEX_JJ = 45
END_CAR_INDEX_JJ = 51

OCCUPANCY_FROM_STATE_FACTOR = 2
OCCUPANCY_FROM_STATE = np.array([[-OCCUPANCY_FROM_STATE_FACTOR, 0, 71.5], [0, -OCCUPANCY_FROM_STATE_FACTOR, 48], [0, 0, 1]])
STATE_FROM_OCCUPANCY = np.linalg.inv(OCCUPANCY_FROM_STATE)

def plot_all_states(all_states):
    plt.plot(all_states[:, 0], all_states[:, 1])
    ax = plt.gca()
    ax.set_xlim([-24, 24])
    ax.set_ylim([-12.25, 37.75])
    plt.show()

def transform_position(matrix, position):
    return (matrix @ np.array([[position[0]], [position[1]], [1]])).reshape(-1)[0:2]

def transform_pose(matrix, pose):
    matrix_pose = np.array([[np.cos(pose[2]), -np.sin(pose[2]), pose[0]], [np.sin(pose[2]), np.cos(pose[2]), pose[1]], [0, 0, 1]])
    transformed_mat = matrix @ matrix_pose
    expanded_rot = np.array([[transformed_mat[0,0], transformed_mat[0,1], 0], [transformed_mat[1,0], transformed_mat[1,1], 0], [0, 0, 1]])
    r = R.from_matrix(expanded_rot)
    return np.array([transformed_mat[0,2], transformed_mat[1,2], r.as_euler('zyx')[0]])

def get_state_position_surface(position, color=(255, 255, 255, 125)):
    occupancy_position = transform_position(OCCUPANCY_FROM_STATE, position)
    ii = int(occupancy_position[0] * 800 / 96)
    jj = int(occupancy_position[1] * 1000 / 96)
    if ii < 0 or ii >= 800 or jj < 0 or jj >= 1000:
        print('out of bounds!')
        return None
    surf = pygame.Surface((1000, 800), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    pygame.draw.rect(surf, color, pygame.Rect(jj - 2, ii - 2, 4, 4), 4)
    # surf = pygame.transform.flip(surf, False, True)
    return surf

def get_state_pose_surface(pose, color=(255, 255, 255, 125)):
    occupancy_pose = transform_pose(OCCUPANCY_FROM_STATE, pose)
    ii = int(occupancy_pose[0] * 800 / 96)
    jj = int(occupancy_pose[1] * 1000 / 96)
    if ii < 0 or ii >= 800 or jj < 0 or jj >= 1000:
        print('out of bounds!')
        return None
    surf = pygame.Surface((1000, 800), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    pygame.draw.line(surf, color, (jj, ii), (jj + 10*np.sin(occupancy_pose[2]), ii + 10*np.cos(occupancy_pose[2])), 4)
    # surf = pygame.transform.flip(surf, False, True)
    return surf

def compute_road_grid(screen):
    valid_road_pixels = np.full((screen.shape[0], screen.shape[1]), False, dtype=np.bool_)
    valid_road_pixels[np.all(screen >= MINIMUM_ROAD_COLOR, axis=2) & np.all(screen <= MAXIMUM_ROAD_COLOR, axis=2)] = True
    valid_road_pixels[START_CAR_INDEX_II:END_CAR_INDEX_II, START_CAR_INDEX_JJ:END_CAR_INDEX_JJ] = True
    return valid_road_pixels

def compute_speed(hull):
    return (hull.linearVelocity[0] ** 2 + hull.linearVelocity[1] ** 2) ** 0.5

def compute_angular_velocity(hull):
    return hull.angularVelocity

def compute_wheel_angle(car):
    return car.wheels[0].joint.angle

# def compute_target_pixel(road):
#     mean_road = np.mean(np.where(road == True), axis=1)
#     return mean_road

def compute_target_pixel(road):
    occupancy_car_position = transform_position(OCCUPANCY_FROM_STATE, np.array([0, 0]))
    road_positions = np.transpose(np.where(road == True))
    differences = road_positions - occupancy_car_position
    distances = differences[0] ** 2 + differences[1] ** 2
    farthest_road_index = np.argmax(distances)
    if distances[farthest_road_index] < 70:
        return (50, 2) # Goal to the left when can't find road
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
        self.car_wheel_angle = compute_wheel_angle(car)
        self.car_speed = compute_speed(car.hull)
        self.target_pixel = compute_target_pixel(self.road)

def generate_random_control_sequence(sequence_length):
    return np.hstack((np.random.rand(sequence_length, 1) * 2 - 1, np.random.rand(sequence_length, 1), np.random.rand(sequence_length, 1)))

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
    def __init__(self, horizon: int, sample_time: float, recompute_iterations: int):
        super().__init__()
        self.horizon = horizon
        self.sample_time = sample_time
        self.recompute_iterations = recompute_iterations
        self.reset()

    def reset(self):
        super().reset()
        self.c = 0
        self.action_sequence = np.zeros((0, 3)) #np.zeros((self.horizon, 3)) # TODO Fix this
    
    def model_system_output(self, u):
        all_states = np.empty((len(u) + 1, 6))
         # x position (forward), y position (left), angle, speed, angular velocity, wheel angle
        current_state = np.array([0, 0, 0, self.perception.car_speed, self.perception.car_angular_velocity, self.perception.car_wheel_angle])
        for ii, u_t in enumerate(u):
            turn_control = u_t[0]
            throttle_control = u_t[1]
            brake_control = u_t[2]
            all_states[ii] = current_state
            braking_factor = 0.25 * (1 - brake_control) + 0.75
            current_state[0] += current_state[3] * np.cos(current_state[2]) * self.sample_time
            current_state[1] += current_state[3] * np.sin(current_state[2]) * self.sample_time
            current_state[2] += current_state[4] * self.sample_time
            current_state[3] = current_state[3] * braking_factor + throttle_control * 1
            current_state[4] = current_state[3] * np.sin(current_state[5]) / 4
            current_state[5] = np.clip(current_state[5] - turn_control * 0.07, -0.42, 0.42)
        all_states[-1] = current_state
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
        
        non_optimal_speed = np.sum(np.power(states[:, 3] - 30, 2))

        summed_angular_speed = np.sum(np.abs(states[:, 4]))
        unstable_cost = 1 / (summed_angular_speed)

        summed_distances_from_road = np.sum(self.compute_min_distances_from_road(states))
        stay_on_road_cost = summed_distances_from_road

        actuation_cost = np.sum(np.power(u, 2))

        target_pixel = self.perception.target_pixel
        target_position = transform_position(STATE_FROM_OCCUPANCY, target_pixel)

        summed_distance_from_target = np.sum(np.power(np.sum(np.power(states[:, 0:2] - target_position, 2), axis=1), 0.5))
        target_cost = summed_distance_from_target

        vector_to_target = target_position - states[:, 0:2]
        angled_towards_target = np.sum(np.power(np.arctan2(vector_to_target[:,1], vector_to_target[:,0]) - states[:,2], 2))
        return target_cost + non_optimal_speed * 20 + angled_towards_target * 1e6# + stay_on_road_cost + 1e-3 * actuation_cost + immobile_cost + unstable_cost + poor_progress_cost + stay_on_road_cost

    # def get_optimal_data_input_sequence(self):
    #     def optimization_function(u) -> float:
    #         u = u.reshape((self.horizon, 3))
    #         states, u = self.model_system_output(u)
    #         return self.system_cost(states, u)
    #     u_guess = np.zeros((self.horizon * 3))
    #     # Constrain every 3 values of u to be between -1 and 1 and all other values to be between 0 and 1
    #     constraints = scipy.optimize.LinearConstraint(np.eye(u_guess.shape[0]), np.tile([-1, 0, 0], self.horizon), np.ones(u_guess.shape[0]))
    #     result = scipy.optimize.minimize(optimization_function, u_guess, method='trust-constr', constraints=constraints, options={'maxiter': 100}) # TODO Make this faster to not limit
    #     control_input = result.x.reshape((self.horizon, 3))
    #     return control_input
    
    def get_optimal_data_input_sequence(self):
        discrete_control_space = [np.array([1, 0, 0]), np.array([0, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        control_input = np.empty((self.horizon, 3))
        for ii in range(self.horizon):
            optimal_cost = np.inf
            optimal_control = np.array([0, 0, 0])
            for control_possibility in discrete_control_space:
                new_control_sequence = np.vstack((control_input[:ii, :], np.tile(control_possibility, 5).reshape(5, -1)))
                states, u = self.model_system_output(new_control_sequence)
                cost = self.system_cost(states, u)
                if cost < optimal_cost:
                    optimal_cost = cost
                    optimal_control = control_possibility
            control_input[ii] = optimal_control
        return control_input

    def get_next_action(self) -> np.ndarray:
        if self.action_sequence.shape[0] == 0 or self.c >= self.recompute_iterations: # TODO Switch to multithreaded # TODO This relates to the sample_time = this / FPS (50)
            self.c = 0
            u = self.get_optimal_data_input_sequence()
            self.action_sequence = u
            # print(self.current_action)
            # plot_all_states(self.model_system_output(u)[0])
        current_action = self.action_sequence[0]
        self.action_sequence = self.action_sequence[1:]
        self.c += 1
        return current_action

def main_control_loop(controller, environment, max_games=None, max_iterations=None):
    perception = Perception()
    quit = False
    game_counter = 0
    while not quit and (max_games == None or game_counter < max_games):
        environment.reset()
        controller.reset()
        total_distance = 0
        # total_reward = 0.0
        # steps = 0
        # Burn startup frames
        for ii in range(6*10):
            state, reward, terminated, truncated, info = environment.step(NULL_ACTION)
            perception.compute_perception(state)
            environment.show_graphics()
        
        iteration_counter = 0
        while True and (max_iterations == None or iteration_counter < max_iterations):
            controller.set_perception(perception)

            action = controller.get_next_action()
            # all_states, u = controller.model_system_output(controller.action_sequence)
            # colors = [(int(255 - ii * 245 / all_states.shape[0]), int(255 - ii * 245 / all_states.shape[0]), 255, 200) for ii in range(all_states.shape[0])]
            # surfaces = [get_state_pose_surface(all_states[ii][:3], colors[ii]) for ii in range(all_states.shape[0])]
            # for surface in surfaces:
            #     environment.draw_surface(surface)
            quit = controller.quit
            restart = controller.restart
            state, reward, terminated, truncated, info = environment.step(action)
            
            # Display target pixel
            environment.draw_surface(get_state_position_surface(transform_position(STATE_FROM_OCCUPANCY, perception.target_pixel)))

            perception.compute_perception(state)
            # total_distance += perception.car_speed * 1/50   
            # print(total_distance)
            
            # total_reward += reward
            # if steps % 200 == 0 or terminated or truncated:
            #     print("\naction " + str([f"{x:+0.2f}" for x in action]))
            #     print(f"step {steps} total_reward {total_reward:+0.2f}")
            # steps += 1
            # environment.show_graphics()
            if terminated or truncated or restart or quit:
                break
            iteration_counter += 1
            # time.sleep(1)

        game_counter += 1
    environment.close()

def run_and_visualize_model_single_input(controller, environment, control_sequence):
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
        # print(all_states)
        colors = [(int(255 - ii * 245 / all_states.shape[0]), int(255 - ii * 245 / all_states.shape[0]), 255, 200) for ii in range(all_states.shape[0])]
        surfaces = [get_state_pose_surface(all_states[ii][:3], colors[ii]) for ii in range(all_states.shape[0])]
        for surface in surfaces:
            environment.draw_surface(surface)
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
    parser.add_argument('--render', type=bool, default=True, help='Whether or not to render the game.')
    parser.add_argument('--record-dir', type=str, default=None, help='A directory to record the run to.') # TODO Implement recording. TODO Implement playback.
    # env = gym.wrappers.RecordVideo(env, "videos", step_trigger=lambda x: x % 100 == 0)
    args = parser.parse_args()
    controller = None
    if args.controller == 'human':
        controller = HumanController()
    if args.controller == 'MPC':
        controller = MPCController(30, 0.02, 10)

    render_mode = 'human' if args.render else None
    
    environment = CustomCarRacing(domain_randomize=False, continuous=True, render_mode=render_mode)

    main_control_loop(controller, environment)
    # profile_main_control_loop(controller, environment, 1, 100)
    # run_and_visualize_model_single_input(controller, environment, [(0, 1, 0)] * 20 + [(-1, 0, 0)] * 30 + [(1, 0, 0.1)] * 30)#+ [(0, 0, 1)] * 15)
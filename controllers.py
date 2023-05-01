import numpy as np
import pygame
import scipy
from utils import transform_position, STATE_FROM_OCCUPANCY
from perception import Perception

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

class MPCController(BaseController):
    def __init__(self, sample_time: float, parameters: dict = {}):
        super().__init__()
        self.sample_time = sample_time
        self.unpack_parameters(parameters)
        self.reset()

    def unpack_parameters(self, parameters: dict):
        self.parameters = parameters

        self.horizon = int(self.parameters.get('horizon', 25))
        self.recompute_iterations = int(self.parameters.get('recompute_iterations', 5))
        self.optimal_speed = self.parameters.get('optimal_speed', 30)
        self.optimal_speed_cost_factor = self.parameters.get('optimal_speed_cost_factor', 1)
        self.turning_cost_factor = self.parameters.get('turning_cost_factor', 0.05)
        self.angled_towards_far_target_cost_factor = self.parameters.get('angled_towards_far_target_cost_factor', 2)
        self.lateral_displacement_from_road_middle_cost_factor = self.parameters.get('lateral_displacement_from_road_middle_cost_factor', 1)
        self.distance_road_mean_cost_factor = self.parameters.get('distance_road_mean_cost_factor', 5)
        self.distance_road_road_cost_factor = self.parameters.get('distance_road_road_cost_factor', 20)
        # TODO Parameterize a element product on each cost function 
    
    def reset(self):
        super().reset()
        self.c = 0
        self.action_sequence = np.zeros((0, 3))
    
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

    def system_cost(self, states, u):
        turning_cost = np.sum(np.abs(u[:,0]))
        
        optimal_speed_cost = np.sum(np.power(states[:, 3] - self.optimal_speed, 2))

        target_position = transform_position(STATE_FROM_OCCUPANCY, self.perception.target_pixel_far)
        distance_road_road_cost = np.sum(np.power(np.sum(np.power(states[:, 0:2] - target_position, 2), axis=1), 0.5))

        vector_to_target = target_position - states[:, 0:2]
        angled_towards_far_target_cost = np.sum(np.power(np.arctan2(vector_to_target[:,1], vector_to_target[:,0]) - states[:,2], 2))

        target_position = transform_position(STATE_FROM_OCCUPANCY, self.perception.road_mean)
        summed_distance_from_target = np.sum(np.power(np.sum(np.power(states[:, 0:2] - target_position, 2), axis=1), 0.5))
        distance_road_mean_cost = summed_distance_from_target

        target_position = transform_position(STATE_FROM_OCCUPANCY, self.perception.road_middle)
        summed_distance_from_target = np.sum(np.power(states[:, 0:2] - target_position, 2)[:,1])
        lateral_displacement_from_road_middle_cost = summed_distance_from_target

        return distance_road_road_cost * self.distance_road_road_cost_factor + distance_road_mean_cost * self.distance_road_mean_cost_factor + lateral_displacement_from_road_middle_cost * self.lateral_displacement_from_road_middle_cost_factor + angled_towards_far_target_cost * self.angled_towards_far_target_cost_factor + optimal_speed_cost * self.optimal_speed_cost_factor + turning_cost * self.turning_cost_factor
    
    def scipy_get_optimal_data_input_sequence(self):
        def optimization_function(u) -> float:
            u = u.reshape((self.horizon, 3))
            states, u = self.model_system_output(u)
            return self.system_cost(states, u)
        u_guess = np.zeros((self.horizon * 3))
        # Constrain every 3 values of u to be between -1 and 1 and all other values to be between 0 and 1
        constraints = scipy.optimize.LinearConstraint(np.eye(u_guess.shape[0]), np.tile([-1, 0, 0], self.horizon), np.ones(u_guess.shape[0]))
        result = scipy.optimize.minimize(optimization_function, u_guess, method='trust-constr', constraints=constraints, options={'maxiter': 100})
        control_input = result.x.reshape((self.horizon, 3))
        return control_input
    
    def greedy_get_optimal_data_input_sequence(self):
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
        if self.action_sequence.shape[0] == 0 or self.c >= self.recompute_iterations:
            self.c = 0
            u = self.greedy_get_optimal_data_input_sequence()
            self.action_sequence = u
        current_action = self.action_sequence[0]
        self.action_sequence = self.action_sequence[1:]
        self.c += 1
        return current_action
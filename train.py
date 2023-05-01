from main import play_single_game
import numpy as np
from car_racing import CustomCarRacing
import warnings
from utils import get_state_pose_surface, get_state_position_surface, transform_position, STATE_FROM_OCCUPANCY, OCCUPANCY_FROM_STATE, OCCUPANCY_FROM_STATE_FACTOR
from tqdm import tqdm
from main import play_single_game
import profile
from perception import Perception
from controllers import HumanController, MPCController
import os
import json
from pathlib import Path
from simanneal import Annealer
from alive_progress import alive_bar
import random
import jsonpickle

def score_controller(environment, evaluating_seeds, controller, max_game_iterations, progress_bar) -> dict:
    game_scores = np.empty(len(evaluating_seeds))
    for ii, seed in list(enumerate(evaluating_seeds)):
        environment.np_random = np.random.default_rng(seed)
        _, game_score = play_single_game(controller, environment, max_game_iterations)
        game_scores[ii] = game_score
        progress_bar()
    return {'avg game score': np.average(game_scores), 'std game score': np.std(game_scores)}

def grid_search_incremement_parameters(parameters, parameter_bounds):
    # Move parameters through the grid search
    increment_complete = False
    for key in parameter_bounds.keys():
        start, stop, increment = parameter_bounds[key]
        if parameters[key] + increment > stop:
            parameters[key] = start
        else:
            parameters[key] += increment
            increment_complete = True
            break
    return increment_complete, parameters

def find_optimal_controller_parameters_grid_search(environment, controller, eval_seeds, test_seeds, parameter_range, output_file_path, number_controller_evals, max_game_iterations):
    assert len(set(eval_seeds).intersection(test_seeds)) == 0, 'Evaluation and test sets contain overlap.'
    # print(np.array([(stop - start) // increment + 1 for start, stop, increment in parameter_range.values()]))
    expected_number_controller_evals = np.prod(np.array([(stop - start) // increment + 1 for start, stop, increment in parameter_range.values()]))
    assert expected_number_controller_evals == number_controller_evals, f'Parameter bounds expected controller evaluations {expected_number_controller_evals} must be equal to the max number of game played {number_controller_evals}'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    assert not Path(output_file_path).is_file(), 'Must write to a new file.'
    output_file = open(output_file_path, 'w')

    increment_complete = True
    parameters = {key: start for key, (start, stop, increment) in parameter_range.items()}
    with alive_bar(number_controller_evals * (len(eval_seeds) + len(test_seeds))) as progress_bar:
        for controller_evaluation in tqdm(range(number_controller_evals)):
            assert increment_complete, 'Increment did not complete. Looped back to original state.'
            controller.unpack_parameters(parameters)
            eval_metrics = score_controller(environment, eval_seeds, controller, max_game_iterations, progress_bar)
            test_metrics = score_controller(environment, test_seeds, controller, max_game_iterations, progress_bar)
            output_string = f'{json.dumps({"Eval Metrics": eval_metrics, "Test Metrics": test_metrics, "Parameters": parameters})}\n'
            output_file.write(output_string)
            output_file.flush()
            increment_complete, parameters = grid_search_incremement_parameters(parameters, parameter_range)

    output_file.close()

class ParameterTuningProblem(Annealer):
    # start, stop per parameter
    def __init__(self, parameter_bounds, parameter_additive_factors, environment, controller, eval_seeds, test_seeds, output_file_path, max_game_iterations, progress_bar):
        self.parameters = {key: np.random.uniform(start, stop) for key, (start, stop) in parameter_bounds.items()}
        super().__init__(self.parameters)
        self.parameter_bounds = parameter_bounds
        self.parameter_additive_factors = parameter_additive_factors
        self.environment = environment
        self.controller = controller
        self.eval_seeds = eval_seeds
        self.test_seeds = test_seeds
        self.progress_bar = progress_bar
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        assert not Path(output_file_path).is_file(), 'Must write to a new file.'
        self.output_file = open(output_file_path, 'w')
        self.max_game_iterations = max_game_iterations

    def move(self):
        for key in self.parameters.keys():
            self.parameters[key] += np.random.uniform(self.parameter_additive_factors[key][0], self.parameter_additive_factors[key][1])
            self.parameters[key] = max(min(self.parameters[key], self.parameter_bounds[key][1]), self.parameter_bounds[key][0])

    def energy(self):
        controller.unpack_parameters(self.parameters)
        eval_metrics = score_controller(self.environment, self.eval_seeds, self.controller, self.max_game_iterations, self.progress_bar)
        test_metrics = score_controller(self.environment, self.test_seeds, self.controller, self.max_game_iterations, self.progress_bar)
        output_string = f'{json.dumps({"Eval Metrics": eval_metrics, "Test Metrics": test_metrics, "Parameters": self.parameters})}\n'
        self.output_file.write(output_string)
        self.output_file.flush()
        return eval_metrics['avg game score']

def find_optimal_controller_parameters_simulated_annealing(environment, controller, eval_seeds, test_seeds, parameter_bounds, relative_parameter_movement_size, output_file_path, number_controller_evals, max_game_iterations):
    parameter_additive_factors = {}
    for key, (start, stop) in parameter_bounds.items():
        relative_step_size = (stop - start) * relative_parameter_movement_size
        parameter_additive_factors[key] = (-relative_step_size, relative_step_size)

    with alive_bar(number_controller_evals * (len(eval_seeds) + len(test_seeds))) as progress_bar:
        annealer = ParameterTuningProblem(parameter_bounds, parameter_additive_factors, environment, controller, eval_seeds, test_seeds, output_file_path, max_game_iterations, progress_bar)
        annealer.Tmax = 1e-3#1000
        annealer.Tmin = 1e-6#2.5
        annealer.steps = number_controller_evals - 1
        annealer.updates = 0

        output, average_game_score = annealer.anneal()
        print(output, average_game_score)

class GeneticPopulationMember():
    def __init__(self, parameters):
        self.parameters = parameters
        self.eval_metrics = None
        self.test_metrics = None

    def get_metrics(self, environment, controller, eval_seeds, test_seeds, output_file, max_game_iterations, progress_bar):
        if self.eval_metrics == None:
            controller.unpack_parameters(self.parameters)
            eval_metrics = score_controller(environment, eval_seeds, controller, max_game_iterations, progress_bar)
            test_metrics = score_controller(environment, test_seeds, controller, max_game_iterations, progress_bar)
            output_string = f'{json.dumps({"Eval Metrics": eval_metrics, "Test Metrics": test_metrics, "Parameters": self.parameters})}\n'
            output_file.write(output_string)
            output_file.flush()
            self.eval_metrics = eval_metrics
            self.test_metrics = test_metrics
        return self.eval_metrics
    
    def to_dict(self):
        return {"Parameters": self.parameters, "Eval Metrics": self.eval_metrics, "Test Metrics": self.test_metrics}


class GeneticAlgorithm():
    def __init__(self, parameter_bounds, relative_parameter_movement_size, num_population, environment, controller, eval_seeds, test_seeds, output_file_path, debug_file_path, max_game_iterations, progress_bar):
        self.environment = environment
        self.controller = controller
        self.eval_seeds = eval_seeds
        self.test_seeds = test_seeds
        self.max_game_iterations = max_game_iterations
        self.progress_bar = progress_bar
        self.num_population = num_population

        self.parameter_bounds = parameter_bounds
        self.parameter_additive_factors = {}
        for key, (start, stop) in parameter_bounds.items():
            relative_step_size = (stop - start) * relative_parameter_movement_size
            self.parameter_additive_factors[key] = (-relative_step_size, relative_step_size)

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        assert not Path(output_file_path).is_file(), 'Must write to a new file.'
        self.output_file = open(output_file_path, 'w')

        os.makedirs(os.path.dirname(debug_file_path), exist_ok=True)
        assert not Path(debug_file_path).is_file(), 'Must write to a new file.'
        self.debug_file = open(debug_file_path, 'w')

        self.population = []
        for ii in range(num_population):
            random_parameters = {key: np.random.uniform(start, stop) for key, (start, stop) in self.parameter_bounds.items()}
            self.population.append(GeneticPopulationMember(random_parameters))

    def generate_new_population_member(self, population_members):
        new_parameters = {}
        for key, (start, stop) in self.parameter_bounds.items():
            if random.random() < 0.5:
                # Select 1 parent gene
                selected_pop_member = random.randint(0, len(population_members) - 1)
                new_parameters[key] = population_members[selected_pop_member].parameters[key]
            else:
                # Average all parent genes
                new_parameters[key] = np.average(np.array([population_member.parameters[key] for population_member in population_members]))
            new_parameters[key] += np.random.uniform(self.parameter_additive_factors[key][0], self.parameter_additive_factors[key][1])
            new_parameters[key] = max(min(new_parameters[key], self.parameter_bounds[key][1]), self.parameter_bounds[key][0])
        return GeneticPopulationMember(new_parameters)
    
    def get_random_parent_indices(self):
        new_population_indices = np.random.choice(len(self.population), int(random.randint(1, int(len(self.population) / 2))), False)
        return new_population_indices

    def repopulate(self):
        while len(self.population) < self.num_population:
            parent_indices = self.get_random_parent_indices()
            parents = [self.population[ii] for ii in parent_indices]
            self.population.append(self.generate_new_population_member(parents))

    def rank_population_with_metrics(self, population_with_metrics):
        population_with_metrics.sort(key = lambda population_member_with_metrics: population_member_with_metrics[1]['avg game score'], reverse=True)
        return population_with_metrics

    def get_new_population_indices(self, ranked_population):
        new_population_indices = []
        final_population_size = int(self.num_population / 2)
        
        for ii in range(len(ranked_population)):
            if len(new_population_indices) == final_population_size:
                break
            if len(ranked_population) - 1 - ii == final_population_size - len(new_population_indices):
                new_population_indices.append(ii)
                continue
            if random.random() < 1 - (ii - 2) / len(ranked_population):
                new_population_indices.append(ii)
        return new_population_indices
    
    def save_population(self):
        self.debug_file.write(f'{json.dumps([pop_member.to_dict() for pop_member in self.population])}\n')
        self.debug_file.flush()

    def evolve_population(self):
        population_with_metrics = [(population_member, population_member.get_metrics(self.environment, self.controller, self.eval_seeds, self.test_seeds, self.output_file, self.max_game_iterations, self.progress_bar)) for population_member in self.population]
        self.save_population()
        ranked_population = self.rank_population_with_metrics(population_with_metrics)
        new_population_indices = self.get_new_population_indices(ranked_population)
        self.population = [ranked_population[ii][0] for ii in new_population_indices]
        self.repopulate()

def find_optimal_controller_parameters_genetic(environment, controller, eval_seeds, test_seeds, parameter_bounds, relative_parameter_movement_size, num_population, output_file_path, debug_file_path, number_controller_evals, max_game_iterations):
    with alive_bar(number_controller_evals * (len(eval_seeds) + len(test_seeds))) as progress_bar:
        genetic_algorithm = GeneticAlgorithm(parameter_bounds, relative_parameter_movement_size, num_population, environment, controller, eval_seeds, test_seeds, output_file_path, debug_file_path, max_game_iterations, progress_bar)
        for ii in range(int((number_controller_evals - num_population) / int(num_population / 2)) + 1):
            genetic_algorithm.evolve_population()

if __name__=='__main__':
    eval_seeds = np.arange(0, 5)
    test_seeds = np.arange(5, 10)
    max_iterations_per_game = 750

    controller = controller = MPCController(0.02)
    environment = CustomCarRacing(domain_randomize=False, continuous=True, render_mode='state_pixels')

    # start, stop, increment per parameter
    # parameter_range = {'horizon': (10, 30, 5), 'recompute_iterations': (5, 9, 2), 'optimal_speed': (20, 50, 5), 'optimal_speed_cost_factor': (0, 25, 5), 'turning_cost_factor': (0, 0.26, 0.05), 'angled_towards_far_target_cost_factor': (0, 25, 5), 'lateral_displacement_from_road_middle_cost_factor': (0, 25, 5), 'distance_road_mean_cost_factor': (0, 25, 5), 'distance_road_road_cost_factor': (0, 25, 5)}
    # output_file_path = 'runs/grid_search.txt'
    # find_optimal_controller_parameters_grid_search(environment, controller, eval_seeds, test_seeds, parameter_range, output_file_path, 4898880, max_iterations_per_game)

    # parameter_bounds = {'horizon': (10, 30), 'recompute_iterations': (5, 9), 'optimal_speed': (20, 60), 'optimal_speed_cost_factor': (0, 50), 'turning_cost_factor': (0, 0.5), 'angled_towards_far_target_cost_factor': (0, 50), 'lateral_displacement_from_road_middle_cost_factor': (0, 50), 'distance_road_mean_cost_factor': (0, 50), 'distance_road_road_cost_factor': (0, 50)}
    # output_file_path = 'runs/hill_climb_1.txt'
    # find_optimal_controller_parameters_simulated_annealing(environment, controller, eval_seeds, test_seeds, parameter_bounds, 0.15, output_file_path, 250, max_iterations_per_game)

    parameter_bounds = {'horizon': (10, 30), 'recompute_iterations': (5, 9), 'optimal_speed': (20, 60), 'optimal_speed_cost_factor': (0, 50), 'turning_cost_factor': (0, 0.5), 'angled_towards_far_target_cost_factor': (0, 50), 'lateral_displacement_from_road_middle_cost_factor': (0, 50), 'distance_road_mean_cost_factor': (0, 50), 'distance_road_road_cost_factor': (0, 50)}
    output_file_path = 'runs/genetic.txt'
    debug_file_path = 'runs/genetic_populations.txt'
    find_optimal_controller_parameters_genetic(environment, controller, eval_seeds, test_seeds, parameter_bounds, 0.15, 25, output_file_path, debug_file_path, 250, max_iterations_per_game)
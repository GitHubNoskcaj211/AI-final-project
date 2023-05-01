import umap
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

def get_parameters(file_path):
    number_of_parameters = 9
    file = open(file_path, 'r')
    lines = file.readlines()
    paraemters_per_instance = np.empty((len(lines), number_of_parameters))
    for ii, line in enumerate(lines):
        d = json.loads(line)
        parameters = d['Parameters']
        for jj, parameter in enumerate(parameters.values()):
            paraemters_per_instance[ii, jj] = parameter
    file.close()

    return paraemters_per_instance


reducer = umap.UMAP()
genetic_params = get_parameters('runs/genetic.txt')
sim_anneal_high_temp_params = get_parameters('runs/simulated_annealing_250_controller_evals.txt')
sim_anneal_low_temp_params = get_parameters('runs/simulated_annealing_250_controller_evals_low_temp.txt')
hill_climb_params = get_parameters('runs/hill_climb_1.txt')
all_data = np.vstack((genetic_params, sim_anneal_high_temp_params, sim_anneal_low_temp_params, hill_climb_params))

embedding = reducer.fit_transform(all_data)

palette = sns.color_palette(n_colors = 4)
a = genetic_params.shape[0]
b = a + sim_anneal_high_temp_params.shape[0]
c = b + sim_anneal_low_temp_params.shape[0]
plt.scatter(embedding[:a, 0], embedding[:a, 1], color=palette[0], marker='x', label='Genetic', s=15)
plt.scatter(embedding[a:b, 0], embedding[a:b, 1], color=palette[1], marker='x', label='High Temp Simulated Annealing', s=15)
plt.scatter(embedding[b:c, 0], embedding[b:c, 1], color=palette[2], marker='x', label='Low Temp Simulated Annealing', s=15)
plt.scatter(embedding[c:, 0], embedding[c:, 1], color=palette[3], marker='x', label='Hill Climbing', s=15)
plt.gca().set_aspect('equal', 'datalim')
plt.legend()
plt.title('UMAP Projection of Parameters')
plt.show()
input()
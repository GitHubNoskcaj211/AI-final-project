from matplotlib import pyplot as plt
import json

def get_metric_data_points(file_path, metric):
    eval_data_points = []
    test_data_points = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            d = json.loads(line)
            eval_data_points.append(d['Eval Metrics'][metric])
            test_data_points.append(d['Test Metrics'][metric])

    return eval_data_points, test_data_points

def create_scatter_plot(file_path, x_metric, y_metric, title):
    fig, axis = plt.subplots(2, 1)
    fig.suptitle(title, fontsize=30)

    axis[0].set_title('Evaluation Metrics', fontsize=25)
    axis[0].set_xlabel(x_metric, fontsize=15)
    axis[0].set_ylabel(y_metric, fontsize=15)
    axis[1].set_title('Test Metrics', fontsize=25)
    axis[1].set_xlabel(x_metric, fontsize=15)
    axis[1].set_ylabel(y_metric, fontsize=15)
    axis[0].set_xlim(-100, 800)
    axis[0].set_ylim(0, 500)
    axis[1].set_xlim(-100, 800)
    axis[1].set_ylim(0, 500)


    eval_x, test_x = get_metric_data_points(file_path, x_metric)
    eval_y, test_y = get_metric_data_points(file_path, y_metric)
    
    axis[0].scatter(eval_x, eval_y, marker='x', color=(0,0,0), s=15)
    axis[1].scatter(test_x, test_y, marker='x', color=(0,0,0), s=15)
    
    return fig

create_scatter_plot('runs/genetic.txt', 'avg game score', 'std game score', 'Metrics from Genetic Parameters').show()
create_scatter_plot('runs/simulated_annealing_250_controller_evals.txt', 'avg game score', 'std game score', 'Metrics from Simulated Annealing High Temperature').show()
create_scatter_plot('runs/simulated_annealing_250_controller_evals_low_temp.txt', 'avg game score', 'std game score', 'Metrics from Simulated Annealing Low Temperature').show()
create_scatter_plot('runs/hill_climb_1.txt', 'avg game score', 'std game score', 'Metrics from Hill Climbing').show()

input()
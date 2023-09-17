import torch
from src.progress_bar import ProgressBar
import numpy as np
import nn_ext
import time
from test.perf import pickler
import matplotlib.pyplot as plt
from test.perf.performance_unit import PerformanceUnit
import itertools


class CorrelateAllUnit(PerformanceUnit):
    short_name = 'corrall'
    description = 'correlation with input activations and kernels'
    f_vals = range(8, 65, 8)
    mb_vals = range(8, 17, 4)

    def interactive_input(self):
        descriptions = ['Number of filters', 'Minibatch size']
        ranges = super(CorrelateAllUnit, CorrelateAllUnit).interactive_input_helper(descriptions)
        self.f_vals = ranges[0]
        self.mb_vals = ranges[1]

    def run(self):
        torch_times = np.zeros((len(self.f_vals), len(self.mb_vals)))
        haida_times = np.zeros((len(self.f_vals), len(self.mb_vals)))
        board_size = 19
        progress_bar = ProgressBar(len(self.f_vals) * len(self.mb_vals))
        progress_bar.start()
        for f, mb in itertools.product(range(len(self.f_vals)), range(len(self.mb_vals))):
            activations = np.random.randn(self.mb_vals[mb], self.f_vals[f], board_size, board_size)
            kernels = np.random.randn(self.f_vals[f], self.f_vals[f], 3, 3)

            t = time.time()
            haida_result = nn_ext.correlate_all(activations, kernels, False)
            t = time.time() - t
            haida_times[f, mb] = t

            torch_activations = torch.tensor(activations, dtype=torch.float64)
            torch_kernels = torch.transpose(torch.tensor(kernels, dtype=torch.float64), 0, 1)
            t = time.time()
            torch_result = torch.nn.functional.conv2d(torch_activations, torch_kernels, padding=1)
            t = time.time() - t
            torch_times[f, mb] = t

            if not np.allclose(haida_result, torch_result.detach().numpy()):
                raise Exception
            progress_bar.increment()
        progress_bar.end()

        axis_labels = ['filters', 'minibatch size']
        axis_vals = [self.f_vals, self.mb_vals]
        times = {'torch': torch_times, 'haida': haida_times}
        data = pickler.PerformanceData(self.short_name, axis_labels, axis_vals, times)
        filename = pickler.save(data)
        print('saved to {}'.format(filename))

    @staticmethod
    def plot(data: pickler.PerformanceData):
        fig = plt.figure(figsize=plt.figaspect(0.5))
        fig.suptitle(data.unit)

        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel(data.axis_labels[0])
        ax.set_ylabel(data.axis_labels[1])
        ax.set_zlabel('time', rotation='vertical')

        x, y = np.meshgrid(data.axis_vals[0], data.axis_vals[1])
        colors = itertools.cycle(['r', 'c', 'g'])
        for label in data.times:
            ax.plot_wireframe(x, y, np.transpose(data.times[label]), label=label, color=colors.__next__())
        ax.legend()
        plt.show()

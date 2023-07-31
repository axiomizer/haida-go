from src.nn.conv_block import ConvolutionalBlock
from test.mse_stub import MseStub
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch
import time
from test.perf.progress_bar import ProgressBar
from test.perf import pickler
from test.perf.performance_unit import PerformanceUnit


class ConvolutionalBlockFeedforwardUnit(PerformanceUnit):
    short_name = 'convff'
    description = 'feedforward on the convolutional block'
    i_vals = range(8, 65, 8)
    o_vals = range(8, 65, 8)
    b_vals = (9, 13, 19)
    mb_vals = range(8, 17, 4)

    def interactive_input(self):
        descriptions = ['Number of input filters', 'Number of output filters', 'Board size', 'Minibatch size']
        ranges = super(ConvolutionalBlockFeedforwardUnit, ConvolutionalBlockFeedforwardUnit)\
            .interactive_input_helper(descriptions)
        self.i_vals = ranges[0]
        self.o_vals = ranges[1]
        self.b_vals = ranges[2]
        self.mb_vals = ranges[3]

    def run(self):
        torch_times = np.zeros((len(self.i_vals), len(self.o_vals), len(self.b_vals), len(self.mb_vals)))
        haida_times = np.zeros((len(self.i_vals), len(self.o_vals), len(self.b_vals), len(self.mb_vals)))
        progress_bar = ProgressBar(len(self.i_vals) * len(self.o_vals) * len(self.b_vals) * len(self.mb_vals))
        progress_bar.start()
        for i, o, b, mb in itertools.product(range(len(self.i_vals)),
                                             range(len(self.o_vals)),
                                             range(len(self.b_vals)),
                                             range(len(self.mb_vals))):
            torch_conv = torch.nn.Sequential(
                torch.nn.Conv2d(self.i_vals[i], self.o_vals[o], 3, padding=1, dtype=torch.float64),
                torch.nn.ReLU()
            )

            haida_conv = ConvolutionalBlock(self.i_vals[i], self.o_vals[o])
            haida_conv.kernels = np.copy(torch.transpose(torch_conv[0].weight, 0, 1).detach().numpy())
            haida_conv.biases = np.copy(torch_conv[0].bias.detach().numpy())
            haida_conv.to = MseStub()

            # time feedforward for both neural nets
            np_in = np.random.randn(self.mb_vals[mb], self.i_vals[i], self.b_vals[b], self.b_vals[b])
            torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
            t = time.time()
            torch_conv(torch_in)
            torch_times[i, o, b, mb] = time.time() - t
            t = time.time()
            haida_conv.feedforward(np_in)
            haida_times[i, o, b, mb] = time.time() - t

            progress_bar.increment()
        progress_bar.end()

        axis_labels = ['input filters', 'output filters', 'board size', 'minibatch size']
        axis_vals = [self.i_vals, self.o_vals, self.b_vals, self.mb_vals]
        times = {'torch': torch_times, 'haida': haida_times}
        data = pickler.PerformanceData(self.short_name, axis_labels, axis_vals, times)
        filename = pickler.save(data)
        print('saved to {}'.format(filename))

    @staticmethod
    def plot(data: pickler.PerformanceData):
        fig = plt.figure(figsize=plt.figaspect(0.5))
        fig.suptitle(data.unit)

        axis_indices = [(0, 1), (2, 3)]
        for i in range(len(axis_indices)):
            x_axis, y_axis = axis_indices[i]
            ax = fig.add_subplot(1, len(axis_indices), i + 1, projection='3d')
            ax.set_xlabel(data.axis_labels[x_axis])
            ax.set_ylabel(data.axis_labels[y_axis])
            ax.set_zlabel('time', rotation='vertical')

            colors = itertools.cycle(['r', 'c', 'g'])
            x, y = np.meshgrid(data.axis_vals[x_axis], data.axis_vals[y_axis])
            axis_maxes = [len(r) - 1 for r in data.axis_vals]
            idx = tuple([slice(None) if i in [x_axis, y_axis] else axis_maxes[i] for i in range(len(data.axis_vals))])
            for name, times in data.times.items():
                z = np.transpose(times[idx].squeeze())
                ax.plot_wireframe(x, y, z, label=name, color=colors.__next__())
            ax.legend()
        plt.show()

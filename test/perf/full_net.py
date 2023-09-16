import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch
import time
from src.progress_bar import ProgressBar
from test.perf import pickler
from test.perf.performance_unit import PerformanceUnit
from test.torch_net import TorchNet
from src.bot.nn.haida_net import HaidaNet


class FullNetUnit(PerformanceUnit):
    short_name = 'full'
    description = 'one training step with full neural net'
    f_vals = range(8, 65, 8)
    mb_vals = range(8, 17, 4)

    def interactive_input(self):
        descriptions = ['Number of filters', 'Minibatch size']
        ranges = super(FullNetUnit, FullNetUnit).interactive_input_helper(descriptions)
        self.f_vals = ranges[0]
        self.mb_vals = ranges[1]

    def run(self):
        torch_times = np.zeros((len(self.f_vals), len(self.mb_vals)))
        haida_times = np.zeros((len(self.f_vals), len(self.mb_vals)))
        input_channels = 17
        residual_blocks = 19
        board_size = 19
        progress_bar = ProgressBar(len(self.f_vals) * len(self.mb_vals))
        progress_bar.start()
        for f, mb in itertools.product(range(len(self.f_vals)), range(len(self.mb_vals))):
            torch_net = TorchNet(residual_blocks, input_channels, self.f_vals[f], board_size)
            haida_net = HaidaNet(board_size, residual_blocks, input_channels, self.f_vals[f])

            # prepare training data
            np_in = np.random.randn(self.mb_vals[mb], input_channels, board_size, board_size)
            torch_in = torch.tensor(np_in, dtype=torch.float64)
            pi_raw = torch.randn(self.mb_vals[mb], (board_size ** 2) + 1, dtype=torch.float64)
            pi = torch.nn.functional.softmax(pi_raw, dim=1, dtype=torch.float64)
            z = torch.randn(self.mb_vals[mb], 1, dtype=torch.float64)
            minibatch = [np_in, pi.detach().numpy(), np.ndarray.flatten(z.detach().numpy())]
            loss1 = torch.nn.CrossEntropyLoss()
            loss2 = torch.nn.MSELoss(reduction='mean')
            optimizer = torch.optim.SGD(torch_net.parameters(), lr=0.01)
            optimizer.zero_grad()

            # time torch net
            t = time.time()
            torch_results = torch_net(torch_in)
            total_loss = loss1(torch_results[0], pi) + loss2(torch_results[1], z)
            total_loss.backward()
            optimizer.step()
            torch_times[f, mb] = time.time() - t

            # time haida net
            t = time.time()
            haida_net.train(minibatch)
            haida_times[f, mb] = time.time() - t

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

from nn.conv_block import ConvolutionalBlock
from mse_stub import MseStub
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch
import time
from perf.progress_bar import ProgressBar
from perf import perf_data


def feedforward(i_vals=range(8, 65, 8), o_vals=range(8, 65, 8), b_vals=(9, 13, 19), mb_vals=range(8, 17, 4), run_name=None):
    torch_times = np.zeros((len(i_vals), len(o_vals), len(b_vals), len(mb_vals)))
    haida_times = np.zeros((len(i_vals), len(o_vals), len(b_vals), len(mb_vals)))
    progress_bar = ProgressBar(len(i_vals) * len(o_vals) * len(b_vals) * len(mb_vals))
    progress_bar.start()
    for i, o, b, mb in itertools.product(range(len(i_vals)),
                                         range(len(o_vals)),
                                         range(len(b_vals)),
                                         range(len(mb_vals))):
        torch_conv = torch.nn.Sequential(
            torch.nn.Conv2d(i_vals[i], o_vals[o], 3, padding=1, dtype=torch.float64),
            torch.nn.ReLU()
        )

        haida_conv = ConvolutionalBlock(i_vals[i], o_vals[o])
        haida_conv.kernels = np.copy(torch.transpose(torch_conv[0].weight, 0, 1).detach().numpy())
        haida_conv.biases = np.copy(torch_conv[0].bias.detach().numpy())
        haida_conv.to = MseStub()

        # time feedforward for both neural nets
        np_in = np.random.randn(mb_vals[mb], i_vals[i], b_vals[b], b_vals[b])
        torch_in = torch.tensor(np_in, dtype=torch.float64, requires_grad=True)
        t = time.time()
        torch_conv(torch_in)
        torch_times[i, o, b, mb] = time.time() - t
        t = time.time()
        haida_conv.feedforward(np_in)
        haida_times[i, o, b, mb] = time.time() - t

        progress_bar.increment()
    progress_bar.end()

    data = {
        'type': 'convff',
        'axes': ['input filters', 'output filters', 'board size', 'minibatch size'],
        'ranges': [i_vals, o_vals, b_vals, mb_vals],
        'torch': torch_times,
        'haida': haida_times
    }
    filename = perf_data.save('convff', data, run_name=run_name)
    print('saved to {}'.format(filename))


def plot(run_name, show_torch=True, show_haida=True):
    data = perf_data.load(run_name)
    torch_times = np.array(data['torch'])
    haida_times = np.array(data['haida'])
    fig = plt.figure(figsize=plt.figaspect(0.5))

    if data['type'] == 'convff':
        axis_indices = [(0, 1), (2, 3)]
    else:
        print('plotting not implemented for type {}'.format(data['type']))
        return
    for i in range(len(axis_indices)):
        x_axis, y_axis = axis_indices[i]
        ax = fig.add_subplot(1, len(axis_indices), i + 1, projection='3d')
        ax.set_xlabel(data['axes'][x_axis])
        ax.set_ylabel(data['axes'][y_axis])

        x, y = np.meshgrid(data['ranges'][x_axis], data['ranges'][y_axis])
        axis_maxes = [len(r) - 1 for r in data['ranges']]
        idx = tuple([slice(None) if i in [x_axis, y_axis] else axis_maxes[i] for i in range(len(data['ranges']))])
        torch_z = np.transpose(torch_times[idx].squeeze())
        haida_z = np.transpose(haida_times[idx].squeeze())
        if show_torch:
            ax.plot_wireframe(x, y, torch_z, color='r', label='torch')
        if show_haida:
            ax.plot_wireframe(x, y, haida_z, color='c', label='haida')
    plt.show()

import scipy
from perf.progress_bar import ProgressBar
import numpy as np
from nn import operations as op
import time
from perf import perf_data
import matplotlib.pyplot as plt


def convolve(sizes=None, run_name=None):
    if sizes is None:
        sizes = [100 * i for i in range(1, 10)]
    haida_times = []
    ndimage_times = []
    fft_times = []
    progress_bar = ProgressBar(len(sizes))
    progress_bar.start()
    for size in sizes:
        m1 = np.random.randn(size, size)
        m2 = np.random.randn(3, 3)

        t = time.time()
        haida_result = op.convolve(m1, m2)
        t = time.time() - t
        haida_times.append(t)

        t = time.time()
        # only equivalent to haida convolve for 3x3 kernels
        ndimage_result = scipy.ndimage.convolve(m1, np.flip(m2), mode='constant', cval=0.0)
        t = time.time() - t
        ndimage_times.append(t)

        t = time.time()
        # only equivalent to haida convolve for 3x3 kernels
        fft_result = scipy.signal.fftconvolve(m1, np.flip(m2), mode='same')
        t = time.time() - t
        fft_times.append(t)

        if not np.allclose(haida_result, ndimage_result):
            raise Exception
        if not np.allclose(haida_result, fft_result):
            raise Exception
        progress_bar.increment()
    progress_bar.end()

    data = {
        'type': 'conv',
        'axis': 'size of first matrix',
        'sizes': sizes,
        'haida': haida_times,
        'ndimage': ndimage_times,
        'fft': fft_times
    }
    filename = perf_data.save('conv', data, run_name=run_name)
    print('saved to {}'.format(filename))


def plot(run_name):
    data = perf_data.load(run_name)
    if data['type'] != 'conv':
        print('plotting not implemented for type {}'.format(data['type']))
        return
    sizes = np.array(data['sizes'])
    fig, ax = plt.subplots()
    ax.plot(sizes, np.array(data['haida']), label='haida')
    ax.plot(sizes, np.array(data['ndimage']), label='ndimage')
    ax.plot(sizes, np.array(data['fft']), label='fft')
    ax.legend()
    plt.show()

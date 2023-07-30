import scipy
from test.perf.progress_bar import ProgressBar
import numpy as np
import nnops_ext
import time
from test.perf import pickler
import matplotlib.pyplot as plt


def correlate(sizes=range(100, 5001, 100), run_name=None):
    haida_times = []
    ndimage_times = []
    fft_times = []
    progress_bar = ProgressBar(len(sizes))
    progress_bar.start()
    for size in sizes:
        m1 = np.random.randn(size, size)
        m2 = np.random.randn(3, 3)

        t = time.time()
        haida_result = nnops_ext.correlate(m1, m2, 1)
        t = time.time() - t
        haida_times.append(t)

        t = time.time()
        # only equivalent to haida correlation for 3x3 kernels
        ndimage_result = scipy.ndimage.correlate(m1, m2, mode='constant', cval=0.0)
        t = time.time() - t
        ndimage_times.append(t)

        t = time.time()
        # only equivalent to haida correlation for 3x3 kernels
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
        'type': 'corr',
        'axis': 'size of first matrix',
        'sizes': sizes,
        'haida': haida_times,
        'ndimage': ndimage_times,
        'fft': fft_times
    }
    filename = pickler.save('corr', data, run_name=run_name)
    print('saved to {}'.format(filename))


def plot(data):
    sizes = np.array(data['sizes'])
    fig, ax = plt.subplots()
    fig.suptitle(data['type'])
    ax.set_xlabel(data['axis'])
    ax.set_ylabel('time')
    ax.plot(sizes, np.array(data['haida']), label='haida')
    ax.plot(sizes, np.array(data['ndimage']), label='ndimage')
    ax.plot(sizes, np.array(data['fft']), label='fft')
    ax.legend()
    plt.show()

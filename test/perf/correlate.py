import scipy
from src.progress_bar import ProgressBar
import numpy as np
import nnops_ext
import time
from test.perf import pickler
import matplotlib.pyplot as plt
from test.perf.performance_unit import PerformanceUnit


class CrossCorrelationUnit(PerformanceUnit):
    short_name = 'corr'
    description = 'cross-correlation between two matrices'
    input_sizes = range(100, 5001, 100)

    def interactive_input(self):
        descriptions = ['Size of the first matrix']
        ranges = super(CrossCorrelationUnit, CrossCorrelationUnit).interactive_input_helper(descriptions)
        self.input_sizes = ranges[0]

    def run(self):
        haida_times = []
        ndimage_times = []
        fft_times = []
        progress_bar = ProgressBar(len(self.input_sizes))
        progress_bar.start()
        for size in self.input_sizes:
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

        axis_labels = ['size of first matrix']
        axis_vals = [self.input_sizes]
        times = {'haida': haida_times, 'ndimage': ndimage_times, 'fft': fft_times}
        data = pickler.PerformanceData(self.short_name, axis_labels, axis_vals, times)
        filename = pickler.save(data)
        print('saved to {}'.format(filename))

    @staticmethod
    def plot(data: pickler.PerformanceData):
        sizes = np.array(data.axis_vals[0])
        fig, ax = plt.subplots()
        fig.suptitle(data.unit)
        ax.set_xlabel(data.axis_labels[0])
        ax.set_ylabel('time')
        for name, times in data.times.items():
            ax.plot(sizes, times, label=name)
        ax.legend()
        plt.show()

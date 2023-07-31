from abc import ABC, abstractmethod
from test.perf import pickler


class PerformanceUnit(ABC):
    short_name = NotImplemented
    description = NotImplemented

    @staticmethod
    def interactive_input_helper(descriptions):
        print('Provide a minimum, maximum, and step size for each parameter. '
              'Use positive integers delimited by spaces.')
        ranges = []
        for description in descriptions:
            while True:
                try:
                    inputs = [int(i) for i in input('  {}: '.format(description)).split(' ')]
                except ValueError:
                    pass
                else:
                    if len(inputs) == 3 and all(i > 0 for i in inputs):
                        break
                print('Try again')
            ranges.append(range(inputs[0], inputs[1], inputs[2]))
        return ranges

    @abstractmethod
    def interactive_input(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    @abstractmethod
    def plot(data: pickler.PerformanceData):
        pass

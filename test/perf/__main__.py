import sys
from test.perf import correlate, conv_block, pickler
from dataclasses import dataclass
from typing import Callable


@dataclass
class CodeUnitInfo:
    run_func: Callable
    plot_func: Callable
    description: str


code_units = {'corr': CodeUnitInfo(correlate.correlate, correlate.plot, 'cross-correlation between two matrices'),
              'convff': CodeUnitInfo(conv_block.feedforward, conv_block.plot, 'feedforward on the convolutional block')}


def describe_code_units():
    print('Code units:')
    for unit, info in code_units.items():
        print('  {}: {}'.format(unit, info.description))


def get_custom_parameters(unit):
    # TODO: implement this
    return ''


def plot(name):
    data = pickler.load(name)
    if data['type'] not in code_units:
        print('plotting not implemented for type {}'.format(data['type']))
        return
    code_units[data['type']].plot_func(data)


if len(sys.argv) <= 1:
    sys.exit(2)
elif sys.argv[1] == 'run':
    if len(sys.argv) <= 2:
        describe_code_units()
    elif sys.argv[2] not in code_units:
        sys.exit(2)
    else:
        if len(sys.argv) >= 4 and sys.argv[3] == '-i':
            custom_parameters = get_custom_parameters(sys.argv[2])
        code_units[sys.argv[2]].run_func()
elif sys.argv[1] == 'plot':
    if len(sys.argv) <= 2:
        sys.exit(2)
    else:
        plot(sys.argv[2])
else:
    sys.exit(2)

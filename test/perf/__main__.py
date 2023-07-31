import sys
from test.perf.correlate import CrossCorrelationUnit
from test.perf.conv_block import ConvolutionalBlockFeedforwardUnit
from test.perf import pickler


performance_units = [CrossCorrelationUnit, ConvolutionalBlockFeedforwardUnit]


def describe_perf_units():
    print('Performance units:')
    for unit in performance_units:
        print('  {}: {}'.format(unit.short_name, unit.description))


def find_perf_unit(short_name):
    units = [u for u in performance_units if u.short_name == short_name]
    if len(units) == 0:
        raise ValueError('Performance unit "{}" not found'.format(short_name))
    if len(units) > 1:
        raise ValueError('More than one performance unit with name "{}"'.format(short_name))
    return units[0]


if len(sys.argv) <= 1:
    sys.exit(2)
elif sys.argv[1] == 'run':
    if len(sys.argv) <= 2:
        describe_perf_units()
    else:
        perf_unit = find_perf_unit(sys.argv[2])()
        if len(sys.argv) >= 4 and sys.argv[3] == '-i':
            perf_unit.interactive_input()
        perf_unit.run()
elif sys.argv[1] == 'plot':
    if len(sys.argv) <= 2:
        sys.exit(2)
    else:
        data = pickler.load(sys.argv[2])
        find_perf_unit(data.unit).plot(data)
else:
    sys.exit(2)

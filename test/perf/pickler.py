import os
import datetime
import glob
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
# TODO: does importing 'typing' have performance implications?
# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class

DATA_PATH = os.path.join('test', 'perf', 'data')


@dataclass
class PerformanceData:
    unit: str                   # name of the code unit that was tested
    axis_labels: List[str]      # descriptions of each parameter
    axis_vals: List[List[int]]  # for each parameter, the values of that parameter for which data was collected
    times: Dict[str, Any]       # times[library providing implementation] = run times for each point in parameter space


def save(data: PerformanceData, run_name=None):
    Path(DATA_PATH).mkdir(exist_ok=True)
    if run_name is None:
        filename = generate_filename(data.unit)
    else:
        filename = '{}.pickle'.format(run_name)
    path = os.path.join(DATA_PATH, filename)
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return path


def load(name) -> PerformanceData:
    with open(os.path.join(DATA_PATH, '{}.pickle'.format(name)), 'rb') as f:
        data = pickle.load(f)
    return data


def generate_filename(unit):
    date = datetime.datetime.today().strftime('%m%d%y')
    existing_files = glob.glob(os.path.join(DATA_PATH, '{}-{}-*'.format(unit, date)))
    file_id = max([int(f.split('-')[-1].split('.')[0]) for f in existing_files], default=0) + 1
    return '{}-{}-{}.pickle'.format(unit, date, file_id)

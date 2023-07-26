import os
import datetime
import glob
import pickle


DATA_PATH = os.path.join('test', 'perf', 'data')


def save(run_type, data, run_name=None):
    if run_name is None:
        filename = generate_filename(run_type)
    else:
        filename = os.path.join(DATA_PATH, '{}.pickle'.format(run_name))
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return filename


def load(name):
    with open(os.path.join(DATA_PATH, '{}.pickle'.format(name)), 'rb') as f:
        data = pickle.load(f)
    return data


def generate_filename(prefix):
    date = datetime.datetime.today().strftime('%m%d%y')
    existing_files = glob.glob(os.path.join(DATA_PATH, '{}-{}-*'.format(prefix, date)))
    file_id = max([int(f.split('-')[-1].split('.')[0]) for f in existing_files], default=0) + 1
    return os.path.join(DATA_PATH, '{}-{}-{}.pickle'.format(prefix, date, file_id))

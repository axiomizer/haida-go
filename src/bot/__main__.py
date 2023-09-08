from src.bot import supervised, pipeline, evaluation
import os
import pickle
import sys
from pathlib import Path


TRAINED_NETS_PATH = os.path.join('src', 'bot', 'trained_nets')


def train(name, sup):
    Path(TRAINED_NETS_PATH).mkdir(exist_ok=True)
    filename = os.path.join(TRAINED_NETS_PATH, '{}.pickle'.format(name))
    if os.path.isfile(filename):
        raise ValueError('file already exists: {}'.format(filename))

    if sup:
        nn = supervised.train()
    else:
        nn = pipeline.train()

    # save neural net
    with open(filename, 'wb') as f:
        pickle.dump(nn, f, pickle.HIGHEST_PROTOCOL)
    print('Trained net saved to: {}'.format(filename))


if len(sys.argv) <= 1:
    sys.exit(2)
if sys.argv[1] == 'train':
    if len(sys.argv) <= 2:
        sys.exit(2)
    sup_flag = len(sys.argv) > 3 and sys.argv[3] == '--supervised'
    train(sys.argv[2], sup_flag)
elif sys.argv[1] == 'rank':
    evaluation.rank_bots(TRAINED_NETS_PATH)

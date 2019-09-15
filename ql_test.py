from ql import ValueLearner
from qllevel import QLLevel
from interface import Application2, Game, Direction
import numpy as np
import qlparams
import random

if __name__ == '__main__':
    qll = QLLevel()
    # print 'numhist', 10
    for i in range(100):
        print 2
        hist = qll.play_games(20)
        params = qlparams.params
        vecs, vals = qll.train_on_played_games(hist, params['sums_back'])
        if len(vecs[0]) > 50:
            qll.models[0].fit(np.vstack(vecs[0]), np.vstack(vals[0]), epochs=10, batch_size=128, verbose=0)
        qll.models[1].fit(np.vstack(vecs[1]), np.vstack(vals[1]), epochs=10, batch_size=128, verbose=0)

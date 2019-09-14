from ql import ValueLearner
from interface import Application2, Game, Direction
import numpy as np
import qlparams
import random

if __name__ == '__main__':
    vl = ValueLearner()
    vl.load_latest()
    for i in range(100):
        hist, _ = vl.play_games(20)
        vecs, vals = vl.train_on_played_games_lowest_2(hist)
        vl.model.fit(np.vstack(vecs), np.vstack(vals), epochs=10, batch_size=32, verbose=0)

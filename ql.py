from interface import Game, Application2, Direction, random_game_board, play_random_games
import copy
import tensorflow as tf
import numpy as np
import random
import multiprocessing as mp
from datetime import datetime
from keras.models import load_model
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



class ValueLearner:

    def __init__(self, lrate=.05):
        self.model = None
        self.vecs = {}
        self.vals = {}
        self.now = datetime.now()
        self.logger = None



    def train_n_save(self, states, values, modelfile, start_lr=0.05, num_epoch=20):
        num_input = 16

        model = Sequential()
        model.add(Dense(50, activation='relu', input_dim=num_input))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        sgd = optimizers.SGD(lr=0.1, momentum=0.05, decay=0.0001, nesterov=False)
        model.compile(optimizer=sgd, loss='mean_squared_error')

        #print len(states), len(values)

        model.fit(states, values, epochs=num_epoch, batch_size=512, verbose=0)

        #print model.predict( np.array([[0,0,0,0,0,0,0,0,0,0,1,2,3,4,5,6],[0,0,6,0,0,3,0,0,0,0,1,2,0,4,5,0]]))

        model.save(modelfile)


    """def load(self):
        self.model = load_model('qmodel.h5')
        #print self.model.predict( np.array([[0,0,0,0,0,0,0,0,0,0,1,2,3,4,5,6],[0,0,6,0,0,3,0,0,0,0,1,2,0,4,5,0]]))"""



    def load_latest(self):
        self.exp_num = latest_experiment_num()
        self.vecfile, self.valfile, self.modelfile = filenames(self.exp_num)

        #self.model = load_model(self.modelfile)
        modelpath = './logs/201909011156-num_epochs/num_epochs:20-model'
        self.model = load_model(modelpath)
        """vecs, vals = read_randomEVs_from_file(self.vecfile, self.valfile)
        sum = -1
        for i in range(len(vals)):
            vec, val = vecs[i], vals[i]
            current_sum = pow2_sum(vec)
            if current_sum not in self.vecs.keys():
                sum = current_sum
                self.vecs[sum] = []
                self.vals[sum] = []
            self.vecs[sum].append(vec)
            self.vals[sum].append(val)"""
        sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        self.model.compile(optimizer=sgd, loss='mean_squared_error')


    def move(self, current_game, mode="avg", eps=None):
        if self.model is None:
            return
        num_each_dir = 10
        values = {}
        if mode == "minmax":
            calcs = []
            for dir in list(Direction):
                if not current_game.direction_legal[dir]:
                    continue
                values[dir] = 2.0
                for _ in range(num_each_dir):
                    game = Game(current_game)
                    game.move(dir)
                    array = game.to_logarray()
                    calcs.append(array)
            poss_arrays = np.vstack(calcs)
            vals = self.model.predict(poss_arrays)
            index = 0
            for dir in list(Direction):
                if dir not in values.keys():
                    continue
                for _ in range(num_each_dir):
                    values[dir] = min(values[dir], vals[index][0])
                    index += 1
        elif mode == "avg":
            calcs = []
            """
            for dir in list(Direction):
                if not current_game.direction_legal[dir]:
                    continue
                values[dir] = 0.0
                for _ in range(num_each_dir):
                    game = Game(current_game)
                    game.move(dir)
                    array = game.to_logarray()
                    calcs.append(array)
            """
            states = current_game.get_states_each_dir(num_each_dir)
            for dir in list(Direction):
                if not current_game.direction_legal[dir]:
                    continue
                values[dir] = 0.0
                calcs.extend(states[dir])
            poss_arrays = np.vstack(calcs)
            vals = self.model.predict(poss_arrays)
            index = 0
            for dir in list(Direction):
                if dir not in values.keys():
                    continue
                for _ in range(num_each_dir):
                    values[dir] += vals[index][0]
                    index += 1
        sorted_dirs = sorted(values.keys(), key=lambda x: values[x])
        if eps is None or len(sorted_dirs) == 1:
            return sorted_dirs[-1]
            """
            max_val = -1
            actual_dir = None
            for dir in values.keys():
                if max_val < values[dir]:
                    actual_dir = dir
                    max_val = values[dir]
            return actual_dir
            """
        else:
            # second choice
            rand = random.random()
            if rand < eps:
                return sorted_dirs[-2]
            return sorted_dirs[-1]


    def log(self, arg, *argv):
        line = str(arg)
        for ar in argv:
            line += " "+str(ar)
        print line
        if self.logger is None:
            return
        self.logger.log(line)

    def play_games(self, total_num):
        min_game = 999999999
        avg = 0
        histories = []
        for i in range(total_num):
            history = []
            game = Game()
            while not game.end:
                history.append(game.to_logarray())
                game.move(self.move(game))
            history.append(game.to_logarray())
            min_game = min(min_game, np.sum(game.board))
            avg += len(history)
            histories.append(history)
        self.log(min_game, float(avg)/total_num)
        return (histories, min_game)


    def train_on_played_games_lowest(self, histories, min_game, params):
        sums_back = params["sums_back"]
        sums_back_from = params["sums_back_from"]
        ###
        min_game -= sums_back_from
        act_arrays = {}
        if min_game < 2*sums_back:
            sums_back = min_game / 2
        for i in range(sums_back/2+1):
            act_arrays[min_game-i*2] = []
        for hist in histories:
            index = 0
            while 1:
                vec = hist[index]
                sum = pow2_sum(vec)
                index += 1
                if sum < min_game-sums_back:
                    continue
                if sum > min_game:
                    break
                act_arrays[sum].append((vec, pow2_sum(hist[-1])-sum))
                if sum == min_game:
                    break
        # normalize vals
        numrange = []
        for sum in act_arrays.keys():
            if len(act_arrays[sum]) == 0:
                continue
            numrange.append(sum)
            self.vecs[sum] = []
            self.vals[sum] = []
            max_val = max(act_arrays[sum], key=lambda x: x[1])[1]
            for pair in act_arrays[sum]:
                game = Game(board=list(pair[0]))
                game_val = float(pair[1])/max_val
                for sym in game.symmetries():
                    self.vecs[sum].append(sym.to_logarray())
                    self.vals[sum].append(game_val)

        # add symetries???
        self.train_on_data_lowest(params, numrange)


    def train_on_data_lowest(self, params, numrange):
        learn_all = params["learn_all"]

        vecs = []
        vals = []
        for key in self.vecs.keys():
            if not learn_all:
                if key not in numrange:
                    continue
            vecs.extend(self.vecs[key])
            vals.extend(self.vals[key])
        matrix = np.vstack(vecs)
        labels = np.vstack(vals)
        self.model.fit(matrix, labels, epochs=params["num_epochs"], batch_size=params["batch_size"], verbose=0)


    def train(self, exp_name, params, logname):
        self.logger = Logger(params, logname)
        sgd = optimizers.SGD(lr=params["learning_rate"], momentum=0.0, decay=0.0, nesterov=False)
        self.model.compile(optimizer=sgd, loss='mean_squared_error')
        if exp_name == "lowest":
            for i in range(params["repeats"]):
                pair = self.play_games(params["num_batch"])
                self.train_on_played_games_lowest(pair[0], pair[1], params)
        self.model.save(self.logger.filename+"-model")
        del self.logger


class Logger:
    def __init__(self, params, logname):
        self.filename = logname
        self.file = open(self.filename+"-log", "w")

    def __del__(self):
        self.file.close()

    def log(self, msg):
        self.file.write(msg+"\n")
        self.file.flush()


def pow2_sum(vec):
    current_sum = 0
    for x in vec:
        if x == 0:
            continue
        current_sum += 1 << x
    return current_sum


def game_sort(board):
    board.sort()
    b = copy.deepcopy(board)
    for i in range(4):
        b[8+i] = board[11-i]
        b[i] = board[3-i]
    return b


def random_EV(sum, num_games, shuffles=5, num_random_games=10):
    logarrays = []
    expvals = []
    for i1 in range(num_games):
        board = game_sort(random_game_board(sum))
        for i2 in range(shuffles):
            b = copy.deepcopy(board)
            random.shuffle(b)
            game = Game(board=b)
            syms = game.symmetries()
            ev = 0.0
            for g in syms:
                ev += play_random_games(g, num_random_games)
                logarrays.append(g.to_logarray())
            expvals.append(ev/8)
        game = Game(board=board)
        syms = game.symmetries()
        ev = 0.0
        for g in syms:
            ev += play_random_games(g, num_random_games)
            logarrays.append(g.to_logarray())
        expvals.append(ev/8)
    minval = min(expvals)
    maxval = max(expvals)
    dif = maxval-minval
    if dif == 0.0:
        return [], []
    evs = []
    for ev in expvals:
        for i3 in range(8):
            evs.append((ev-minval)/dif)
    return logarrays, evs


def do_randomEVs_thread(rangemin, rangemax, num_games=2, shuffles=5, num_random_games=10, thread_num=6):
    num_per_thread = (rangemax-rangemin)/thread_num
    j = rangemin
    logarrays, evs = [], []
    threads = []

    output = mp.Queue()
    for i in range(thread_num):
        evthread = None
        if i == thread_num-1:
            evthread = mp.Process(target=do_randomEVs, args=(i, output, j, rangemax, num_games, shuffles, num_random_games))
        else:
            evthread = mp.Process(target=do_randomEVs, args=(i, output, j, j+num_per_thread, num_games, shuffles, num_random_games))
        threads.append(evthread)
        j += num_per_thread
    for p in threads:
        p.start()
    for p in threads:
        pair = output.get()
        logarrays.extend(pair[0])
        evs.extend(pair[1])
    for p in threads:
        p.join()
    return logarrays, evs


def do_randomEVs(pos, output, rangemin, rangemax, num_games=2, shuffles=5, num_random_games=10):
    logarrays = []
    evs = []
    process = max((rangemax-rangemin)/100, 1)
    for i in range(rangemin, rangemax):
        la, ev = random_EV(i, num_games, shuffles, num_random_games)
        logarrays.extend(la)
        evs.extend(ev)
        if i % process == 0:
            print pos, (0.0+i-rangemin)/(rangemax-rangemin)
    output.put((logarrays, evs))


def write_randomEVs_to_file(num, vecfile, valfile, num_games=10,):
    logarrays, evs = do_randomEVs_thread(2, num, num_games=10)
    arrays = np.vstack(logarrays)
    evals = np.vstack(evs)
    arrays.tofile(vecfile)
    evals.tofile(valfile)


def read_randomEVs_from_file(vecfile, valfile):
    vals = np.fromfile(valfile)
    vals.reshape((len(vals), 1))
    vecs = np.fromfile(vecfile, dtype=int).reshape((len(vals), 16))
    return vecs, vals


def latest_experiment_num():
    data_dir = "./dat/"
    maxnum = -1
    for file in os.listdir(data_dir):
        num = int(file[file.rfind('-')+1:])
        maxnum = max(maxnum, num)
    return maxnum


def filenames(exp_num=None):
    if exp_num is None:
        exp_num = latest_experiment_num()
    data_dir = "./dat/"
    exp_num = str(exp_num)
    return data_dir+"vec-"+exp_num, data_dir+"val-"+exp_num, data_dir+"qvals-"+exp_num


def do_whole_stuff():
    num_games = 20
    range_max = 500
    num_epochs = 1000
    # params
    # shit
    exp_num = latest_experiment_num()+1
    vecfile, valfile, modelfile = filenames(exp_num)
    write_randomEVs_to_file(range_max, vecfile, valfile, num_games)
    states, values = read_randomEVs_from_file(vecfile, valfile)
    vl = ValueLearner()
    vl.train_n_save(states, values, modelfile, num_epoch=num_epochs)
    play_a_game()


def play_a_game(total_num=100):
    vl = ValueLearner()
    vl.load_latest()
    avg = 0
    states = []
    for i in range(total_num):
        game = Game()
        num = 0
        while not game.end:
            states.append(game.to_array())
            game.move(vl.move(game))
            num += 1
        print num
        avg += num
    Application2(states).mainloop()
    print float(avg)/total_num


def do_experience(param_name, exp_name, dir_name, value):
    import qlparams
    params = dict(qlparams.params)
    params[param_name] = value
    vl = ValueLearner()
    vl.load_latest()
    vl.train(exp_name, params, dir_name+param_name+":"+str(value))



if __name__ == '__main__':
    import sys
    do_experience(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    #print sys.argv












# print "asd"
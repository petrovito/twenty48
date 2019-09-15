
from interface import Game
from tools import Direction, Logger, pow2_sum
from qlnode import QLNode, make_eval

import numpy as np
from keras.models import load_model


class QLLevel:

    def __init__(self):
        self.models = []
        self.ranges = []
        self.logger = None
        self.load_models()

    def load_models(self):
        modelpath = './logs/201909011156-num_epochs/num_epochs:20-model'
        self.models.append(load_model(modelpath))
        self.models.append(load_model(modelpath))
        self.ranges.append(500)
        self.ranges.append(5000)

    def log(self, arg, *argv):
        line = str(arg)
        for ar in argv:
            line += " "+str(ar)
        print line
        if self.logger is None:
            return
        self.logger.log(line)

    def get_move_list_exhaustive(
            self, current_game, model_index, num_each_dir=3):
        main_node = QLNode(current_game)
        children = main_node.make_children(num_each_dir)
        for child in children:
            child.make_children(num_each_dir)
        make_eval(main_node, self.models[model_index])
        sorted_dirs =\
            sorted(main_node.legal_dirs, key=lambda x: main_node.dir_values[x])
        return sorted_dirs

    def play_games(self, total_num):
        avg = 0
        histories = []
        for i in range(total_num):
            history = []
            game = Game()
            current_sum = np.sum(game.board)
            model_index = 0
            while not game.end:
                history.append(game.to_logarray())
                dir = self.get_move_list_exhaustive(game, model_index)[-1]
                game.move(dir)
                current_sum = np.sum(game.board)
                if current_sum > self.ranges[model_index]:
                    model_index += 1
            history.append(game.to_logarray())
            avg += len(history)
            histories.append(history)
        self.log(float(avg)/total_num)
        return histories

    def train_on_played_games(self, histories, sums_back=100, num_hist=5):
        histories = sorted(histories, key=lambda x: pow2_sum(x[-1]))
        max_game = pow2_sum(histories[num_hist][-1])
        min_game = pow2_sum(histories[0][-1])-sums_back
        act_arrays = {}
        for i in range(min_game, max_game, 2):
            act_arrays[i] = []
        for hist in histories:
            max_sum = pow2_sum(hist[-1])
            index = 0
            while 1:
                if index >= len(hist):
                    break
                vec = hist[index]
                sum = pow2_sum(vec)
                index += 1
                if sum < min_game:
                    continue
                if sum >= max_game:
                    break
                act_arrays[sum].append((vec, max_sum-sum))
        vecs, vals = [[]], [[]]
        current_index = 0
        for sum in sorted(act_arrays.keys()):
            while self.ranges[current_index] < sum:
                vecs.append([])
                vals.append([])
                current_index += 1
            if len(act_arrays[sum]) == 0:
                continue
            max_val = max(act_arrays[sum], key=lambda x: x[1])[1]
            if max_val == 0:
                continue
            for pair in act_arrays[sum]:
                game = Game(board=list(pair[0]))
                game_val = float(pair[1])/max_val
                for sym in game.symmetries():
                    vecs[-1].append(sym.to_logarray())
                    vals[-1].append(game_val)
        return vecs, vals

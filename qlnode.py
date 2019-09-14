
from direction import Direction
from interface import Game
import numpy as np


class QLNode:

    def __init__(self, game, end_node=True):
        self.game = game
        self.legal_dirs = [dir for dir in Direction if game.direction_legal[dir]]
        self.children = {dir: [] for dir in self.legal_dirs}
        self.dir_values = {dir: None for dir in self.legal_dirs}
        self.end_node = end_node
        self.value = None

    def evaluate(self):
        if self.end_node:
            return
        max_val = -1.0
        for dir in self.legal_dirs:
            dir_val = 0.0
            for child_node in self.children[dir]:
                child_node.evaluate()
                dir_val += child_node.value
            self.dir_values[dir] = dir_val / len(self.children[dir])
            max_val = max(max_val, self.dir_values[dir])
        self.value = max_val

    def get_game_vectors(self):
        if self.end_node:
            if self.value is None:
                return [self.game.to_logarray()], [self]
            else:
                return [], []
        vectors = []
        node_ids = []
        for dir in self.legal_dirs:
            for child_node in self.children[dir]:
                vecs, ids = child_node.get_game_vectors()
                vectors.extend(vecs)
                node_ids.extend(ids)
        return vectors, node_ids

    def get_all_children(self):
        children = []
        for dir in self.legal_dirs:
            for child in self.children[dir]:
                children.append(child)
        return children

    def make_children(self, num_children=5):
        if self.game.end:
            self.value = 0.0
            return []
        self.end_node = False
        children = []
        games = self.game.get_games_each_dir(num_children)
        for dir in self.legal_dirs:
            games_dir = games[dir]
            for new_game in games_dir:
                child = QLNode(new_game)
                self.children[dir].append(child)
                if new_game.end:
                    child.value = 0.0
                else:
                    children.append(child)
        return children


def make_eval(qlnode, vl):
    vecs, ids = qlnode.get_game_vectors()
    if len(vecs) != 0:
        matrix = np.vstack(vecs)
        values = vl.model.predict(matrix)
        #print len(vecs), time.clock()-tim, (time.clock()-tim)/len(vecs)
        for i in range(len(values)):
            ids[i].value = values[i][0]
    #print "-----"
    qlnode.evaluate()
    #print len(vecs), time.clock()-tim, (time.clock()-tim)/len(vecs)


if __name__ == '__main__':
    QLNode(Game())

















# asd

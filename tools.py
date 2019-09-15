from enum import Enum


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


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

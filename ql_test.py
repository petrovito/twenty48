from  ql import ValueLearner
from interface import Application2

if __name__ == '__main__':
    vl = ValueLearner()
    vl.load_latest()
    hist, ming = vl.play_games(100)
    histl = max(hist, key=lambda x: len(x))
    Application2(histl).mainloop()


import random
from enum import Enum
from Tkinter import *
import numpy as np
import copy

import time

def bits(num):
    if num == 0:
        return 0,0
    binary = bin(num)
    setBits = [ones for ones in binary[2:] if ones=='1']
    return len(setBits), len(binary)-2


class Game:


    def __init__(self, game=None, two_r=.75, board=None, empty=False):
        self.two_rate=two_r
        self.end = False
        self.direction_legal = dict()
        if empty:
            return
        elif game is not None:
            self.board = copy.deepcopy(game.board)
            self.end = game.end
            self.direction_legal = dict(game.direction_legal)
            return
        elif board is not None:
            self.board = [[board[4*x+y] for x in range(4)] for y in range(4)]
        else:
            self.board = [[0 for x in range(4)] for y in range(4)]
            self.init()
        self.can_move()


    def init(self):
        count = 0
        while count < 2:
            x=random.randrange(4)
            y=random.randrange(4)
            if self.board[x][y]!=0: continue
            self.board[x][y]=2
            count+=1

    def print_board(self):
        for i in range(4):
            print self.board[i]
        print "----------------"


    def get_states_each_dir(self, num_each_dir=20):
        states = {}
        for direction in list(Direction):
            if not self.direction_legal[direction]:
                continue
            states[direction] = []
            for _ in range(num_each_dir):
                new_game = Game(self)
                new_game.make_move(direction)
                states[direction].append(new_game.to_logarray())
        return states


    def make_move(self, direction):
        if direction == Direction.UP:
            for i in range(4):
                list_nums = []
                for j in range(4):
                    current = self.board[j][i]
                    if current != 0:
                        list_nums.append(current)
                if len(list_nums)==0: continue
                new_list = [list_nums[0]]
                count = 1
                while count < len(list_nums):
                    num = list_nums[count]
                    if num == new_list[-1]:
                        new_list[-1]=2*num
                        count += 1
                        if count == len(list_nums): break
                        num = list_nums[count]
                        new_list.append(num)
                    else: new_list.append(num)
                    count += 1
                j = 0
                while j < len(new_list):
                    self.board[j][i]=new_list[j]
                    j+=1
                while j < 4:
                    self.board[j][i]=0
                    j+=1
        elif direction == Direction.DOWN:
            for i in range(4):
                list_nums = []
                for j in range(4):
                    current = self.board[3-j][i]
                    if current != 0:
                        list_nums.append(current)
                if len(list_nums)==0: continue
                new_list = [list_nums[0]]
                count = 1
                while count < len(list_nums):
                    num = list_nums[count]
                    if num == new_list[-1]:
                        new_list[-1]=2*num
                        count += 1
                        if count == len(list_nums): break
                        num = list_nums[count]
                        new_list.append(num)
                    else: new_list.append(num)
                    count += 1
                j = 0
                while j < len(new_list):
                    self.board[3-j][i]=new_list[j]
                    j+=1
                while j < 4:
                    self.board[3-j][i]=0
                    j+=1
        elif direction == Direction.LEFT:
            for i in range(4):
                list_nums = []
                for j in range(4):
                    current = self.board[i][j]
                    if current != 0:
                        list_nums.append(current)
                if len(list_nums)==0:
                    continue
                new_list = [list_nums[0]]
                count = 1
                while count < len(list_nums):
                    num = list_nums[count]
                    if num == new_list[-1]:
                        new_list[-1]=2*num
                        count += 1
                        if count == len(list_nums):
                            break
                        num = list_nums[count]
                        new_list.append(num)
                    else:
                        new_list.append(num)
                    count += 1
                j = 0
                while j < len(new_list):
                    self.board[i][j]=new_list[j]
                    j += 1
                while j < 4:
                    self.board[i][j] = 0
                    j += 1
        elif direction == Direction.RIGHT:
            for i in range(4):
                list_nums = []
                for j in range(4):
                    current = self.board[i][3-j]
                    if current != 0:
                        list_nums.append(current)
                if len(list_nums)==0:
                    continue
                new_list = [list_nums[0]]
                count = 1
                while count < len(list_nums):
                    num = list_nums[count]
                    if num == new_list[-1]:
                        new_list[-1]=2*num
                        count += 1
                        if count == len(list_nums): break
                        num = list_nums[count]
                        new_list.append(num)
                    else: new_list.append(num)
                    count += 1
                j = 0
                while j < len(new_list):
                    self.board[i][3-j]=new_list[j]
                    j+=1
                while j < 4:
                    self.board[i][3-j]=0
                    j+=1
        num_zeros = 0
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    num_zeros += 1
        if num_zeros > 0:
            #add random
            new_num = 2
            if random.random() > self.two_rate:
                new_num = 4
            while 1:
                x = random.randrange(4)
                y = random.randrange(4)
                if self.board[x][y] != 0:
                    continue
                self.board[x][y] = new_num
                break




    def move(self, direction):
        if not self.direction_legal[direction]:
            return
        self.make_move(direction)
        self.can_move()

    def can_move(self):
        self.direction_legal[Direction.RIGHT] = self.can_move_right()
        self.direction_legal[Direction.LEFT] = self.can_move_left()
        self.direction_legal[Direction.DOWN] = self.can_move_down()
        self.direction_legal[Direction.UP] = self.can_move_up()
        self.end = True
        for dir in Direction:
            if self.direction_legal[dir]: self.end = False


    def can_move_left(self):
        for i in range(4):
            num = self.board[i][0]
            for j in range(1,4):
                if self.board[i][j]==0:
                    num = 0
                    continue
                if self.board[i][j]==num or num==0: return True
                num = self.board[i][j]
        return False

    def can_move_right(self):
        for i in range(4):
            num = self.board[i][3]
            for j in range(1,4):
                if self.board[i][3-j]==0:
                    num = 0
                    continue
                if self.board[i][3-j]==num or num==0: return True
                num = self.board[i][3-j]
        return False

    def can_move_up(self):
        for i in range(4):
            num = self.board[0][i]
            for j in range(1,4):
                if self.board[j][i]==0:
                    num = 0
                    continue
                if self.board[j][i]==num or num==0: return True
                num = self.board[j][i]
        return False

    def can_move_down(self):
        for i in range(4):
            num = self.board[3][i]
            for j in range(1,4):
                if self.board[3-j][i]==0:
                    num = 0
                    continue
                if self.board[3-j][i]==num or num==0: return True
                num = self.board[3-j][i]
        return False


    def to_array(self):
        array = np.zeros((16,), dtype=int)
        for i in range(4):
            for j in range(4):
                array[4*i+j]=self.board[i][j]
        return array

    def to_logarray(self):
        array = np.zeros((16,), dtype=int)
        for i in range(4):
            for j in range(4):
                if self.board[i][j]>0:
                    array[4*i+j]=np.log2(self.board[i][j])
        return array


    def random_uniform_move(self):
        while 1:
            dir = random.choice(list(Direction))
            if self.direction_legal[dir]: return dir

    def random_distr_move(self, distr):
        distr2 = [distr[0]]
        for i in range(1,4):
            distr2.append(distr2[-1]+distr[i])
        return self.random_distr_move2(distr2)

    def random_distr_move2(self, distr):
        while 1:
            rand = random.random()
            for i, dir in enumerate(list(Direction)):
                if distr[i]>rand:
                    if self.direction_legal[dir]: return dir
                    else: break

    def symmetries(self):
        g = Game(self)
        t = self.mirror()
        syms = [g,t]
        for i in range(3):
            syms.append(syms[-2].rotate90())
            syms.append(syms[-2].rotate90())
        return syms


    def rotate90(self):
        game = Game(empty=True)
        game.board = [[self.board[3-i][j] for i in range(4)] for j in range(4)]
        game.can_move()
        return game

    def mirror(self):
        game = Game(empty=True)
        game.board = [[self.board[j][3-i] for i in range(4)] for j in range(4)]
        game.can_move()
        return game


def play_random_games(game, num_games):
    ev = 0.0
    for i in range(num_games):
        new_game = Game(game)
        while not new_game.end:
            dir = new_game.random_uniform_move()
            new_game.move(dir)
            ev += 1
    return ev/num_games


    #2*n total sum
def random_game_deprecated(sum):
    board = []
    num_bits, max_bit = bits(sum)
    for i in range(16):
        while 1:
            rand = random.randint(0,max_bit)
            num = 0
            if rand > 0:
                num = 1 << (rand-1)
            #print i, rand, num, sum, sum-num, bits(sum-num)[0], bin(sum-num)
            if sum<num: return
            if bits(sum-num)[0] < 16-i:
                sum = sum - num
                board.append(num * 2)
                num_bits, max_bit = bits(sum)
                break
    return Game(board=board)

    #2*n total sum
def random_game_board(sum):
    board = []
    for i in range(16):
        if sum == 0:
            board.append(0)
            continue
        num = get_random_2pow(sum, i)
        #print sum, num
        sum = sum - num
        board.append(num * 2)
    return board

def get_random_2pow(sum,i):
    j = 1
    num = 1
    #num = 2^(j-1)
    prob_sum = 0.0
    possible_nums = []
    while num <= sum:
        if bits(sum-num)[0] < 16-i:
            possible_nums.append((j,num))
            prob_sum += 1.0 / j
        num = num << 1
        j += 1
    possible_nums.append((j,0))
    prob_sum += 1.0 / j
    rand = random.random()*prob_sum

    sum  = 0
    for pair in possible_nums:
        sum += 1.0/pair[0]
        if sum > rand: return pair[1]


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3




class Application(Tk):


    def initialize(self):
        self.labels = []
        self.textv = []
        self.game = None
        self.grid()

    def __init__(self, game, parent=None):
        Tk.__init__(self,parent)
        self.parent = parent
        self.initialize()
        self.game=game
        self.title('clicker')
        self.geometry('500x600+300+150')

        for i in range(4):
            self.labels.append([])
            self.textv.append([])
            for j in range(4):
                self.textv[-1].append(StringVar())
                lbl = Label(self, textvariable=self.textv[i][j], font=("Arial Bold", 30))
                lbl.grid(column=j, row=i)
                self.labels[-1].append(lbl)
        self.refresh()
        self.bind("<Key>",self.key)

    def key(self,event):
        if event.keysym == "Right":
            self.game.move(Direction.RIGHT)
        elif event.keysym == "Left":
            self.game.move(Direction.LEFT)
        elif event.keysym == "Up":
            self.game.move(Direction.UP)
        elif event.keysym == "Down":
            self.game.move(Direction.DOWN)
        else: return
        self.refresh()

    def refresh(self):
        for i in range(4):
            for j in range(4):
                self.textv[i][j].set(str(self.game.board[i][j]))
                self.labels[i][j].config(height=3, width=5)




class Application2(Tk):

    def initialize(self):
        self.labels = []
        self.textv = []
        self.game = None
        self.grid()

    def __init__(self, gamez):
        Tk.__init__(self,None)
        self.initialize()
        self.gamez=gamez
        self.current_num = 0
        self.title('clicker')
        self.geometry('500x600+300+150')

        for i in range(4):
            self.labels.append([])
            self.textv.append([])
            for j in range(4):
                self.textv[-1].append(StringVar())
                lbl = Label(self, textvariable=self.textv[i][j], font=("Arial Bold", 30))
                lbl.grid(column=j, row=i)
                self.labels[-1].append(lbl)
                self.labels[i][j].config(height=3, width=5)
        self.refresh()
        self.bind("<Key>",self.key)

    def key(self,event):
        if event.keysym == "Right":
            if self.current_num < len(self.gamez)-1:
                self.current_num += 1
        elif event.keysym == "Left":
            if self.current_num > 0:
                self.current_num -= 1
        self.refresh()



    def refresh(self):
        for i in range(4):
            for j in range(4):
                self.textv[i][j].set(str(self.gamez[self.current_num][4*i+j]))
        self.title('clicker-'+str(self.current_num))

if __name__ == '__main__':
    print random_game_board(200)











#asd

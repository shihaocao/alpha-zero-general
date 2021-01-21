from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
# from .OthelloLogic import Board
import numpy as np
import gym

class GoGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        raise NotImplementedError
        # return OthelloGame.square_content[piece]

    def __init__(self, n):
        self.n = n
        self.env = gym.make('gym_go:go-v0', size=7, komi=0, reward_method='real')
        self.env.seed(123)
        
        self.state = None
        self.reward = None
        self.player = None
        self.terminal = None
        self.info = None
        
        self.flat_move_size = int(n*n)+1
        self.state = self.env.reset()
        
    def getInitBoard(self):
        # return initial board (numpy board)
        
        return self.state

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        # if action == self.n*self.n:
        #     return (board, -player)
        # b = Board(self.n)
        # b.pieces = np.copy(board)
        # move = (int(action/self.n), action%self.n)
        # b.execute_move(move, player)
        
        self.state, self.reward, self.terminal, self.info = self.env.step(action)
    
        return self.state, self.state[2][0][0]

    def getValidMoves(self, board, player):
        s = self.state
        
        b_pieces = np.array(s[0], dtype=int)
        w_pieces = np.array(s[1], dtype=int)
        ko_illegals = np.array(s[3], dtype=int)

        invalid_moves = np.bitwise_or(b_pieces,w_pieces)
        invalid_moves = np.bitwise_or(invalid_moves,ko_illegals)
        invalid_moves = np.clip(invalid_moves, 0, 1) # .reshape((self.flat_move_size,1))

        invalid_moves = np.bitwise_not(invalid_moves)
        invalid_moves = np.append(invalid_moves.flatten(),1) #pass is always valid
        
        return invalid_moves

    def getGameEnded(self, board, player):
        # # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # # player = 1
        # b = Board(self.n)
        # b.pieces = np.copy(board)
        # if b.has_legal_moves(player):
        #     return 0
        # if b.has_legal_moves(-player):
        #     return 0
        # if b.countDiff(player) > 0:
        #     return 1
        # return -1
        
        if self.terminal:
            if player:
                return self.reward
            else:
                return -self.reward
        else:
            return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return self.state

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i,axes=(1,2))
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return "".join([str(x) for x in board.astype(dtype=int).flatten()])
        # CAN COLLAPSE INDICATOR LAYERS

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        raise RuntimeError("DONT RUN THIS HAHA")
        if player:
            self.reward
        else:
            -self.reward

    @staticmethod
    def display(board):
        self.env.render()

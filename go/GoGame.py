from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
# from .OthelloLogic import Board
import numpy as np
from gym_go import gogame
from gym_go import govars

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
        self.size = n
        self.komi = 3.5
        
    def getInitBoard(self):
        # return initial board (numpy board)
        
        return gogame.init_state(self.size)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.size, self.size)

    def getActionSize(self):
        # return number of actions
        return self.size*self.size + 1

    def getNextState(self, game_state, player, action):
    # def getCanonicalState(self, game_state, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        # if action == self.n*self.n:
        #     return (board, -player)
        # b = Board(self.n)
        # b.pieces = np.copy(board)
        # move = (int(action/self.n), action%self.n)
        # b.execute_move(move, player)
        if isinstance(action, tuple) or isinstance(action, list) or isinstance(action, np.ndarray):
            assert 0 <= action[0] < self.size
            assert 0 <= action[1] < self.size
            action = self.size * action[0] + action[1]
        elif action is None:
            action = self.size ** 2

        next_state = gogame.next_state(game_state, action, canonical=False)
        
        if next_state[govars.TURN_CHNL][0][0]:
            p = 1
        else:
            p = -1
        return next_state, p
        # self.done = gogame.game_ended(self.state_)
        # return np.copy(self.state_), self.reward(), self.done, self.info()

    def getValidMoves(self, game_state, player):        
        b_pieces = np.array(game_state[govars.WHITE], dtype=int)
        w_pieces = np.array(game_state[govars.BLACK], dtype=int)
        ko_illegals = np.array(game_state[govars.INVD_CHNL], dtype=int)

        invalid_moves = np.bitwise_or(b_pieces,w_pieces)
        invalid_moves = np.bitwise_or(invalid_moves,ko_illegals)
        invalid_moves = np.clip(invalid_moves, 0, 1) # .reshape((self.flat_move_size,1))
        
        invalid_moves = 1-invalid_moves
        invalid_moves = np.append(invalid_moves.flatten(),1) #pass is always valid
        
        return invalid_moves

    def winning(self, game_state):
        """
        :return: Who's currently winning in BLACK's perspective, regardless if the game is over
        """
        return gogame.winning(game_state, self.komi)

    def winner(self, game_state, player):
        """
        Get's the winner in BLACK's perspective
        :return:
        """

        if self.game_ended(game_state):
            return self.winning(game_state)
        else:
            return 0

    def game_ended(self, game_state):
        if gogame.game_ended(game_state):
            return True
        return False

    def getGameEnded(self, game_state, player):
        '''
        black pov
        '''
        return self.winner(game_state, player)

    def getCanonicalForm(self, game_state, player):
        # return state if player==1, else return -state if player==-1
        return gogame.canonical_form(game_state)

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.size**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.size, self.size))
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
        board_s = "".join(str(x) for x in np.array(board, dtype=int).flatten())
        return board_s
    
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

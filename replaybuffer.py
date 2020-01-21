from collections import deque
import numpy as np
import random

class replayBuffer:

    def __init__(self, memory_size, state_size, batch_size):
        print(state_size)
        self.state = np.ndarray(shape=(*state_size, memory_size))
        self.next_state = np.ndarray(shape=(*state_size, memory_size))
        self.action = np.ndarray(shape=(memory_size)) #consider this shape
        self.reward = np.ndarray(shape=(memory_size))
        self.terminal = np.ndarray(shape=(memory_size))

        self.memory_size = memory_size
        self.len = 0
        self.currentElement = 0
        self.full = False
        self.batch_size = batch_size

    def addElement(self, state, action, reward, next_state, terminal):
        if self.currentElement == self.memory_size:
            self.currentElement = 0
            self.full = True
        print(state)
        self.state[self.len] = state
        self.action[self.len] = action
        self.reward[self.len] = reward
        self.next_state[self.len] = next_state
        self.terminal[self.len] = terminal

        self.currentElement = self.currentElement + 1
        self.len = self.len + 1 if not self.full else self.len

    def addCollection(self, collection):
        for state, action, reward, next_state, is_terminal \
          in zip(collection[0] , collection[1], collection[2], collection[3], collection[4]):
            print(state)
            self.addElement(state, action, reward, next_state, is_terminal)
        #consider optimizing

    def getBatch(self):
        indexes = random.sample(range(0, self.len), self.batch_size)

        return (self.state[indexes], self.action[indexes], 
            self.reward[indexes], self.next_state[indexes], 
            self.terminal[indexes])
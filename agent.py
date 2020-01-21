import tensorflow as tf
from tensorflow.keras import layers, models

from replaybuffer import replayBuffer
import numpy as np
# (state, action, reward, next_state, is_terminal) 

class agent:

    def __init__(self, input_shape, output_size,
                learning_rate=0.01, discount=0.9, curiosity_start=0.99,
                curiosity_end=0.01, curiosity_decriment=1/10_000,
                memory_size=1_000, batch_size=32):

        self.input_shape = input_shape
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.batch_size = batch_size

        self.curiosity = curiosity_start
        self.curiosity_start = curiosity_start
        self.curiosity_end = curiosity_end
        self.curiosity_decriment = curiosity_decriment

        self.model = self.createNetwork(input_shape, output_size)
        self.replay_memory = replayBuffer(memory_size, input_shape, batch_size)

    #taken from: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
    def createNetwork(self, image_size, actions):
        return models.Sequential([
            #consider 1x1 conv here
            layers.Conv2D(1, (1, 1), activation='relu',
                input_shape=image_size),
            layers.Conv2D(1, (3, 3), activation='relu'),
            layers.Conv2D(1, (3, 3), strides=2, activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            #dropout + recurrent layer may be wise here
            layers.Dense(actions)
        ])

    def getQs(self, batch):
        Qs = self.model.predict(batch[3])
        qMax = np.argmax(Qs, axis=1)
        for i in range(self.batch_size):
            Qs[i][batch[1]] = batch[2] + self.discount * qMax[i] 
            #update action taken (batch[1]) with q formula
        return Qs

    def trainAction(self, state, action, reward ,next_state, terminal):
        self.replay_memory.addElement(state, action, reward, next_state, terminal)

        if self.replay_memory.len > self.batch_size * 10:
          self.trainOnMemory()

    def trainOnMemory(self):
        batch = self.replay_memory.getBatch()                  
        return self.model.fit(
            batch[0], self.getQs(batch), batch_size=self.batch_size, 
            verbose=0, shuffle=False
        )
    
    def set_curiosity(self, start, stop, iterations):
        self.curiosity_start = start
        self.curiosity_end = stop
        self.curiosity_decriment = (start-stop)/iterations
        self.reset_curiosity()

    def decriment_curiosity(self):
        self.curiosity = self.curiosity - self.curiosity_decriment \
            if self.curiosity > self.curiosity_end \
            else self.curiosity_end

    def reset_curiosity(self):
        self.curiosity = self.curiosity_start

 
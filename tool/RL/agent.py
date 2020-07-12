import numpy as np
import tensorflow as tf
import os
from importlib import *
from DQNalign.tool.RL.alignment import Pairwise
from DQNalign.tool.RL.Learning import *
import time

class Agent():
    def __init__(self, FLAGS, istrain, game_env, model, seq1 = [], seq2 = []):
        """ Get parameters from files """
        self.FLAGS = FLAGS
        self.param = import_module('DQNalign.param.'+self.FLAGS.network_set)
        self.istrain = istrain

        """ Define sequence alignment environment """
        if self.istrain:
            self.env = Pairwise(game_env,0,Z=self.param.Z)
        else:
            if len(seq1)+len(seq2) > 0:
                self.env = Pairwise(game_env,1,seq1,seq2,Z=self.param.Z)
            else:
                self.env = Pairwise(game_env,0,Z=self.param.Z)

        if (self.FLAGS.model_name == "DQN") or (self.FLAGS.model_name == "SSD"):
            self.mainQN = model.mainQN
            self.targetQN = model.targetQN
            self.trainables = model.trainables
            self.targetOps = model.targetOps

        """ Initialize the variables """
        self.total_steps = 0
        self.start = time.time()
        self.myBuffer = experience_buffer()

        """ Exploration strategy """
        if self.istrain:
            self.l_seq = game_env.l_seq
            self.e = self.param.startE
            self.stepDrop = (self.param.startE - self.param.endE) / self.param.annealing_steps

    def reset(self):
        """ Define sequence alignment environment """
        self.istrain = True
        self.env.sizeS1 = self.l_seq[0]
        self.env.sizeS2 = self.l_seq[1]

    def set(self, seq1 = [], seq2 = []):
        """ Define sequence alignment environment """
        self.istrain = False
        self.env.test(seq1,seq2)

    def train(self, sess):
        trainBatch = self.myBuffer.sample(self.param.batch_size)  # Select the batch from the experience buffer
        #print(np.shape(np.vstack(trainBatch[:, 3])))
        
        if (self.FLAGS.model_name == "DQN") or (self.FLAGS.model_name == "SSD"):
            # The estimated Q value from main network is Q1, from target network is Q2
            Q1 = sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
            Q2 = sess.run(self.targetQN.Qout, feed_dict={self.targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
            
            # trainBatch[:,4] means that the action was the last step of the episode
            # If the action is the last step, the reward is used for update Q value
            # IF not, the Q value is updated as follows:
            # Qmain(s,a) = r(s,a) + yQtarget(s1,argmaxQmain(s1,a))
            end_multiplier = -(trainBatch[:, 4] - 1)
            doubleQ = Q2[range(self.param.batch_size), Q1]
            targetQ = trainBatch[:, 2] + (self.param.y * doubleQ * end_multiplier)
            _, loss = sess.run([self.mainQN.updateModel, self.mainQN.loss], feed_dict={self.mainQN.scalarInput: np.vstack(trainBatch[:, 0]), self.mainQN.targetQ: targetQ, self.mainQN.actions: trainBatch[:, 1]})
            updateTarget(self.targetOps, sess)  # Update target network with 'tau' ratio


    def skip(self):
        a = []

        seq1end = min(self.env.x+self.env.win_size-1,self.env.sizeS1-1)
        seq2end = min(self.env.y+self.env.win_size-1,self.env.sizeS2-1)
        minend = min(seq1end-self.env.x,seq2end-self.env.y)+1
        diff = np.where(self.env.seq1[self.env.x:self.env.x + minend] != self.env.seq2[self.env.y:self.env.y + minend])
        if np.size(diff) > 0:
            a = [0] * np.min(diff)
        else:
            a = [0] * minend

        return a

    def play(self, sess, record=0):
        # Newly define experience buffer for new episode
        past = time.time()
        if self.FLAGS.show_align:
            dot_plot = 255*np.ones((self.env.sizeS1,self.env.sizeS2))
        if self.FLAGS.print_align:
            Nucleotide = ["N","A","C","G","T"]
        if self.istrain:
            episodeBuffer = experience_buffer()        
            # Environment reset for each episode
            s1 = self.env.reset() # Rendered image of the alignment environment
            s1 = processState(s1) # Resize to 1-dimensional vector
        else:
            s = processState(self.env.renderEnv())

        d = False # The state of the game (End or Not)
        rT1 = 0 # Total reward
        rT2 = 0 # Total match
        j = 0
 
        while j < self.env.sizeS1+self.env.sizeS2:  # Training step is proceeded until the maximum episode length
            if self.FLAGS.display_process:
                if j % 1000 == 0:
                    now = time.time()

            # Exploration step
            if self.istrain and (np.random.rand(1) < self.e or self.total_steps < self.param.pre_train_steps):
                a = [np.random.randint(0, self.param.n_action)]
            elif self.env.seq1[self.env.x]==self.env.seq2[self.env.y]:
                a = self.skip()
            else:
                #test = time.time()
                s1 = processState(self.env.renderEnv())
                #print("Rendering stage :",time.time()-test)
                #test = time.time()
                a = sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s1]})
                #print("Prediction stage :",time.time()-test)
                #test = time.time()


            # Update the DQN network
            if self.istrain:
                # Calculate the change of the state, reward and d(one)
                s = s1
                s1, r, d = self.env.step(a[0])
                j += 1
                s1 = processState(s1)
                self.total_steps += 1
                rT1 += r
                rT2 += (r>0)
                episodeBuffer.add(np.reshape(np.array([s, a[0], r, s1, d]), [1, 5]))  # Save the result into episode buffer

                if self.total_steps > self.param.pre_train_steps:
                    # Refresh exploration probability (epsilon-greedy)
                    if self.e > self.param.endE:
                        self.e -= self.stepDrop

                    # For every update_freq, update the main network
                    if self.total_steps % (self.param.update_freq) == 0:
                        self.train(sess)
                        #print("Training stage :",time.time()-test)
            else:
                for _ in range(np.size(a)):
                    if self.FLAGS.show_align:
                        dot_plot[self.env.x][self.env.y] = 0
                    if self.FLAGS.print_align:
                        record.record(self.env.x, self.env.y, a[_], Nucleotide[self.env.seq1[self.env.x]+1], Nucleotide[self.env.seq2[self.env.y]+1])

                    r, d = self.env.teststep(a[_])
                    j += 1
                    rT1 += r
                    rT2 += (r>0)
                    if d == True:
                        break
                #print("Do step stage :",time.time()-test)

            if d == True:
                break

            if self.FLAGS.display_process:
                if j % 1000 == 1000-1:
                    print("Align step is processed :",j+1,"with",time.time()-now)
    
        # Add the results of the episode into the total results
        if self.istrain:
            self.myBuffer.add(episodeBuffer.buffer)

        now = time.time()
        if self.FLAGS.show_align and self.FLAGS.print_align:
            return rT1, rT2, now - past, j, dot_plot
        elif self.FLAGS.show_align:
            return rT1, rT2, now - past, j, dot_plot
        elif self.FLAGS.print_align:
            return rT1, rT2, now - past, j
        return rT1, rT2, now - past, j

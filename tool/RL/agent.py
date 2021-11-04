import numpy as np
import tensorflow as tf
import os
from importlib import *
from DQNalign.tool.RL.alignment import Pairwise
from DQNalign.tool.RL.Learning import *
import DQNalign.tool.util.function as function
import time

class Agent():
    def __init__(self, FLAGS, istrain, game_env, model, seq1 = [], seq2 = [], ismeta = False):
        """ Get parameters from files """
        self.FLAGS = FLAGS
        self.istrain = istrain

        if ismeta:
            self.param = import_module('DQNalign.param.MAML')
        else:
            self.param = import_module('DQNalign.param.'+self.FLAGS.network_set)

            """ Exploration strategy """
            if self.istrain:
                self.l_seq = game_env.l_seq
                self.e = self.param.startE
                self.stepDrop = (self.param.startE - self.param.endE) / self.param.annealing_steps

        """ Define sequence alignment environment """
        if self.istrain:
            self.env = Pairwise(game_env,0,Z=self.param.Z)
        else:
            if len(seq1)+len(seq2) > 0:
                self.env = Pairwise(game_env,1,seq1,seq2,Z=self.param.Z)
            else:
                self.env = Pairwise(game_env,0,Z=self.param.Z)

        if ismeta:
            self.mainQN = model.mainQN
            self.tempQN = model.targetQN
            self.trainables = model.trainables
            self.copyOps = model.copyOps
        elif (self.FLAGS.model_name == "DQN") or (self.FLAGS.model_name == "SSD") or (self.FLAGS.model_name == "DiffSSD") or (self.FLAGS.model_name == "FFTDQN"):
            self.mainQN = model.mainQN
            self.targetQN = model.targetQN
            self.trainables = model.trainables
            self.targetOps = model.targetOps

        """ Initialize the variables """
        self.total_steps = 0
        self.start = time.time()
        self.myBuffer = experience_buffer()

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
        
        if (self.FLAGS.model_name == "DQN") or (self.FLAGS.model_name == "SSD") or (self.FLAGS.model_name == "DiffSSD") or (self.FLAGS.model_name == "FFTDQN"):
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
        minend = min(seq1end-self.env.x,seq2end-self.env.y)
        diff = np.where(self.env.seq1[self.env.x:self.env.x + minend + 1] != self.env.seq2[self.env.y:self.env.y + minend + 1])
        if np.size(diff) > 0:
            a = [0] * np.min(diff)
        else:
            a = [0] * minend

        return a

    def reverseskip(self):
        a = []

        seq1end = max(self.env.x-self.env.win_size+1,0)
        seq2end = max(self.env.y-self.env.win_size+1,0)
        minend = min(self.env.x-seq1end,self.env.y-seq2end)
        diff = np.where(self.env.seq1[self.env.x-minend:self.env.x + 1][::-1] != self.env.seq2[self.env.y-minend:self.env.y + 1][::-1])
        if np.size(diff) > 0:
            a = [0] * np.max(diff)
        else:
            a = [0] * minend

        return a

    def skipRC(self):
        a = []

        seq1end = min(self.env.x+self.env.win_size-1,self.env.sizeS1-1)
        seq2end = max(self.env.y-self.env.win_size+1,0)
        minend = min(seq1end-self.env.x,self.env.y-seq2end)
        diff = np.where(self.env.seq1[self.env.x:self.env.x + minend + 1] != self.env.rev2[self.env.y-minend:self.env.y + 1][::-1])
        if np.size(diff) > 0:
            a = [0] * np.min(diff)
        else:
            a = [0] * minend

        return a

    def reverseskipRC(self):
        a = []

        seq1end = max(self.env.x-self.env.win_size+1,0)
        seq2end = min(self.env.y+self.env.win_size-1,self.env.sizeS2-1)
        minend = min(self.env.x-seq1end,seq2end-self.env.y)
        diff = np.where(self.env.seq1[self.env.x-minend:self.env.x + 1][::-1] != self.env.rev2[self.env.y:self.env.y + minend + 1])
        if np.size(diff) > 0:
            a = [0] * np.max(diff)
        else:
            a = [0] * minend

        return a

    def metatrain(self, sess, mainBuffer=False, X=0):
        episodeBuffer = experience_buffer()
        if self.istrain:
            # Environment reset for each episode
            s1 = self.env.reset() # Rendered image of the alignment environment
            s1 = processState(s1)
        else:
            s1 = processState(self.env.renderEnv())

        d = False # The state of the game (End or Not)
        j = 0
        rT1 = 0 # Total reward
        rT2 = 0 # Total reward
        best = 0
        flag = False
 
        while j < self.env.sizeS1+self.env.sizeS2:  # Training step is proceeded until the maximum episode length
            if self.env.seq1[self.env.x]==self.env.seq2[self.env.y]:
                a = self.skip()
            else:
                s1 = processState(self.env.renderEnv())
                a = sess.run(self.tempQN.predict, feed_dict={self.tempQN.scalarInput: [s1]})

            #print(self.env.x,self.env.y,a,self.env.seq1[self.env.x],self.env.seq2[self.env.y],j,rT1)
            # Update the DQN network
            # Calculate the change of the state, reward and d(one)
            for _ in range(np.size(a)):
                s = s1
                s1, r, d = self.env.step(a[_])
                s1 = processState(s1)
                episodeBuffer.add(np.reshape(np.array([s, a[_], r, s1, d]), [1, 5]))  # Save the result into episode buffer
                rT1 += r
                rT2 += (r>0)
                j += 1
                if rT1 >= best:
                    best = rT1

                # if score drops more than X, extension will be ended
                if (rT1 < best - X) and (X>0):
                    flag = True
                    break

                if d == True:
                    break

            if (j % self.param.update_freq == 0) and (j >= self.param.batch_size) and self.istrain:
                #print(j, self.env.x, self.env.y)
                # update the temp network
                trainBatch = episodeBuffer.sample(self.param.batch_size)  # Select the batch from the experience buffer
        
                # The estimated Q value from main network is Q1, from target network is Q2
                Q1 = sess.run(self.tempQN.predict, feed_dict={self.tempQN.scalarInput: np.vstack(trainBatch[:, 3])})
                Q2 = sess.run(self.tempQN.Qout, feed_dict={self.tempQN.scalarInput: np.vstack(trainBatch[:, 3])})
            
                # trainBatch[:,4] means that the action was the last step of the episode
                # If the action is the last step, the reward is used for update Q value
                # IF not, the Q value is updated as follows:
                # Qmain(s,a) = r(s,a) + yQtarget(s1,argmaxQmain(s1,a))
                end_multiplier = -(trainBatch[:, 4] - 1)
                doubleQ = Q2[range(self.param.batch_size), Q1]
                targetQ = trainBatch[:, 2] + (self.param.y * doubleQ * end_multiplier)
                _ = sess.run(self.tempQN.updateModel, feed_dict={self.tempQN.scalarInput: np.vstack(trainBatch[:, 0]), self.tempQN.targetQ: targetQ, self.tempQN.actions: trainBatch[:, 1]})

            if d == True:
                break

            if flag == True:
                break

        if self.istrain:
            # Environment reset for each episode
            s1 = self.env.reset() # Rendered image of the alignment environment
            s1 = processState(s1)

            d = False # The state of the game (End or Not)
            j = 0
            rT1 = 0 # Total reward
            rT2 = 0 # Total reward
 
            while j < self.env.sizeS1+self.env.sizeS2:  # Training step is proceeded until the maximum episode length
                if self.env.seq1[self.env.x]==self.env.seq2[self.env.y]:
                    a = self.skip()
                else:
                    s1 = processState(self.env.renderEnv())
                    a = sess.run(self.tempQN.predict, feed_dict={self.tempQN.scalarInput: [s1]})

                # Update the DQN network
                # Calculate the change of the state, reward and d(one)
                for _ in range(np.size(a)):
                    s = s1
                    s1, r, d = self.env.step(a[_])
                    s1 = processState(s1)
                    mainBuffer.add(np.reshape(np.array([s, a[_], r, s1, d]), [1, 5]))  # Save the result into episode buffer
                    rT1 += r
                    rT2 += (r>0)
                    j += 1

                    if d == True:
                        break

                if d == True:
                    break

        return rT1, rT2, j, mainBuffer

    def metatrain2(self, sess, mainBuffer=False, X=0):
        episodeBuffer = experience_buffer()
        if self.istrain:
            # Environment reset for each episode
            s1 = self.env.reset(2) # Rendered image of the alignment environment
            s1 = processState(s1)
        else:
            s1 = processState(self.env.renderDiff())

        d = False # The state of the game (End or Not)
        j = 0
        rT1 = 0 # Total reward
        rT2 = 0 # Total reward
        best = 0
        flag = False
 
        while j < self.env.sizeS1+self.env.sizeS2:  # Training step is proceeded until the maximum episode length
            if self.env.seq1[self.env.x]==self.env.seq2[self.env.y]:
                a = self.skip()
            else:
                s1 = processState(self.env.renderDiff())
                a = sess.run(self.tempQN.predict, feed_dict={self.tempQN.scalarInput: [s1]})

            #print(self.env.x,self.env.y,a,self.env.seq1[self.env.x],self.env.seq2[self.env.y],j,rT1)
            # Update the DQN network
            # Calculate the change of the state, reward and d(one)
            for _ in range(np.size(a)):
                s = s1
                s1, r, d = self.env.stepDiff(a[_])
                s1 = processState(s1)
                episodeBuffer.add(np.reshape(np.array([s, a[_], r, s1, d]), [1, 5]))  # Save the result into episode buffer
                rT1 += r
                rT2 += (r>0)
                j += 1
                if rT1 >= best:
                    best = rT1

                # if score drops more than X, extension will be ended
                if (rT1 < best - X) and (X>0):
                    flag = True
                    break

                if d == True:
                    break

            if (j % self.param.update_freq == 0) and (j >= self.param.batch_size) and self.istrain:
                #print(j, self.env.x, self.env.y)
                # update the temp network
                trainBatch = episodeBuffer.sample(self.param.batch_size)  # Select the batch from the experience buffer
        
                # The estimated Q value from main network is Q1, from target network is Q2
                Q1 = sess.run(self.tempQN.predict, feed_dict={self.tempQN.scalarInput: np.vstack(trainBatch[:, 3])})
                Q2 = sess.run(self.tempQN.Qout, feed_dict={self.tempQN.scalarInput: np.vstack(trainBatch[:, 3])})
            
                # trainBatch[:,4] means that the action was the last step of the episode
                # If the action is the last step, the reward is used for update Q value
                # IF not, the Q value is updated as follows:
                # Qmain(s,a) = r(s,a) + yQtarget(s1,argmaxQmain(s1,a))
                end_multiplier = -(trainBatch[:, 4] - 1)
                doubleQ = Q2[range(self.param.batch_size), Q1]
                targetQ = trainBatch[:, 2] + (self.param.y * doubleQ * end_multiplier)
                _ = sess.run(self.tempQN.updateModel, feed_dict={self.tempQN.scalarInput: np.vstack(trainBatch[:, 0]), self.tempQN.targetQ: targetQ, self.tempQN.actions: trainBatch[:, 1]})

            if d == True:
                break

            if flag == True:
                break

        if self.istrain:
            # Environment reset for each episode
            s1 = self.env.reset(2) # Rendered image of the alignment environment
            s1 = processState(s1)

            d = False # The state of the game (End or Not)
            j = 0
            rT1 = 0 # Total reward
            rT2 = 0 # Total reward
 
            while j < self.env.sizeS1+self.env.sizeS2:  # Training step is proceeded until the maximum episode length
                if self.env.seq1[self.env.x]==self.env.seq2[self.env.y]:
                    a = self.skip()
                else:
                    s1 = processState(self.env.renderDiff())
                    a = sess.run(self.tempQN.predict, feed_dict={self.tempQN.scalarInput: [s1]})

                # Update the DQN network
                # Calculate the change of the state, reward and d(one)
                for _ in range(np.size(a)):
                    s = s1
                    s1, r, d = self.env.stepDiff(a[_])
                    s1 = processState(s1)
                    mainBuffer.add(np.reshape(np.array([s, a[_], r, s1, d]), [1, 5]))  # Save the result into episode buffer
                    rT1 += r
                    rT2 += (r>0)
                    j += 1

                    if d == True:
                        break

                if d == True:
                    break

        return rT1, rT2, j, mainBuffer

    def Global(self, sess, record=0):
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
            #print(self.env.x, self.env.y)
            if self.FLAGS.display_process:
                if j % 1000 == 0:
                    now = time.time()

            # Exploration step
            if self.env.seq1[self.env.x]==self.env.seq2[self.env.y]:
                a = self.skip()
            elif self.istrain:
                if self.FLAGS.exploration == "e-greedy":
                    if (np.random.rand(1) < self.e or self.total_steps < self.param.pre_train_steps):
                        a = [np.random.randint(0, self.param.n_action)]
                    else:
                        s1 = processState(self.env.renderEnv())
                        a = sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s1]})
                elif self.FLAGS.exploration == "boltzmann":
                    temp = self.e
                    s1 = processState(self.env.renderEnv())
                    Qprobs = sess.run(self.mainQN.Qdist, feed_dict={self.mainQN.scalarInput: [s1], self.mainQN.Temp: [temp]})
                    action_value = np.random.choice(Qprobs[0],p=Qprobs[0])
                    a = [np.argmax(Qprobs[0] == action_value)]
                elif self.FLAGS.exploration == "bayesian":
                    keep = 1-self.e
                    temp = self.e
                    s1 = processState(self.env.renderEnv())
                    Qprobs = sess.run(self.mainQN.Qdist,feed_dict={self.mainQN.scalarInput: [s1], self.mainQN.Temp: [temp], self.mainQN.keep_per: [keep]})
                    action_value = np.random.choice(Qprobs[0],p=Qprobs[0])
                    a = [np.argmax(Qprobs[0] == action_value)]
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
                # Calculate the change of the state, reward and done
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

    def DiffGlobal(self, sess, record=0):
        # Newly define experience buffer for new episode
        past = time.time()
        if self.FLAGS.show_align:
            dot_plot = 255*np.ones((self.env.sizeS1,self.env.sizeS2))
        if self.FLAGS.print_align:
            Nucleotide = ["N","A","C","G","T"]
        if self.istrain:
            episodeBuffer = experience_buffer()        
            # Environment reset for each episode
            s1 = self.env.reset(2) # Rendered image of the alignment environment
            s1 = processState(s1) # Resize to 1-dimensional vector
        else:
            s = processState(self.env.renderDiff())

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
                s1 = processState(self.env.renderDiff())
                #print("Rendering stage :",time.time()-test)
                #test = time.time()
                a = sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s1]})
                #print("Prediction stage :",time.time()-test)
                #test = time.time()


            # Update the DQN network
            if self.istrain:
                # Calculate the change of the state, reward and d(one)
                s = s1
                s1, r, d = self.env.stepDiff(a[0])
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

    def FFTGlobal(self, sess, record=0):
        # Newly define experience buffer for new episode
        past = time.time()
        if self.FLAGS.show_align:
            dot_plot = 255*np.ones((self.env.sizeS1,self.env.sizeS2))
        if self.FLAGS.print_align:
            Nucleotide = ["N","A","C","G","T"]
        if self.istrain:
            episodeBuffer = experience_buffer()        
            # Environment reset for each episode
            s1 = self.env.reset(3) # Rendered image of the alignment environment
            s1 = processState(s1) # Resize to 1-dimensional vector
        else:
            s = processState(self.env.renderFFT())

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
                s1 = processState(self.env.renderFFT())
                #print("Rendering stage :",time.time()-test)
                #test = time.time()
                a = sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s1]})
                #print("Prediction stage :",time.time()-test)
                #test = time.time()


            # Update the DQN network
            if self.istrain:
                # Calculate the change of the state, reward and d(one)
                s = s1
                s1, r, d = self.env.stepFFT(a[0])
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


    def Local(self, sess, uX1, uX2, uY1, uY2, X):
        # Reverse Complement 기능 추가해야함
        # Newly define experience buffer for new episode
        if uY1 < uY2:
            RCmode = 0
        else:
            RCmode = 1

        past = time.time()

        rT1o = 0
        rT2o = 0

        pathx = []
        pathy = []

        d = False # The state of the game (End or Not)
        rT1 = 1 # Total reward
        rT2 = 1 # Total match
        j = 0

        if RCmode == 0:
            best = 1
            best2 = 1
            flag = 0
            pathx1 = []
            pathy1 = []

            #Forward Extension
            if (uX2+1 <= self.env.sizeS1) and (uY2+1 <= self.env.sizeS2):
                self.env.x = uX2 + 1
                self.env.y = uY2 + 1
                pathx1.append(self.env.x)
                pathy1.append(self.env.y)
                bestxy = [self.env.x,self.env.y]

                while j < self.env.sizeS1+self.env.sizeS2-uX2-uY2:
                    # Skip process
                    if self.env.seq1[self.env.x]==self.env.seq2[self.env.y]:
                        a = self.skip()
                    else:
                        #test = time.time()
                        s1 = processState(self.env.renderEnv())
                        #print("Rendering stage :",time.time()-test)
                        #test = time.time()
                        a = sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s1]})
                        #print("Prediction stage :",time.time()-test)
                        #test = time.time()
            
                    for _ in range(np.size(a)):
                        r, d = self.env.teststep(a[_])
                        pathx1.append(self.env.x)
                        pathy1.append(self.env.y)
                        j += 1
                        rT1 += r
                        rT2 += (r>0)
                        if rT1 >= best:
                            best = rT1
                            best2 = rT2
                            bestxy = [self.env.x, self.env.y]

                        # if score drops more than X, extension will be ended
                        if rT1 < best - X:
                            flag = 1
                            break

                        if d == True:
                            flag = 1
                            break
                        #print("Do step stage :",time.time()-test)

                    if flag:
                        break

                bestp = function.check_where(pathx1, pathy1, bestxy)
                pathx1 = pathx1[:bestp+1]
                pathy1 = pathy1[:bestp+1]

            rT1o += best
            rT2o += best2

            d = False # The state of the game (End or Not)
            rT1 = 1 # Total reward
            rT2 = 1 # Total match
            j = 0

            best = 1
            best2 = 1
            flag = 0
            pathx2 = []
            pathy2 = []

            #Reverse Extension
            if (uX1-1 >= 0) and (uY1-1 >= 0):
                self.env.x = uX1 - 1
                self.env.y = uY1 - 1
                pathx2.append(self.env.x)
                pathy2.append(self.env.y)
                bestxy = [self.env.x,self.env.y]

                while j < uX1+uY1:
                    # Skip process
                    if self.env.seq1[self.env.x]==self.env.seq2[self.env.y]:
                        a = self.reverseskip()
                    else:
                        #test = time.time()
                        s1 = processState(self.env.renderRev())
                        #print("Rendering stage :",time.time()-test)
                        #test = time.time()
                        a = sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s1]})
                        #print("Prediction stage :",time.time()-test)
                        #test = time.time()
            
                    for _ in range(np.size(a)):
                        r, d = self.env.teststep(10+a[_])
                        pathx2.append(self.env.x)
                        pathy2.append(self.env.y)
                        j += 1
                        rT1 += r
                        rT2 += (r>0)
                        if rT1 >= best:
                            best = rT1
                            best2 = rT2
                            bestxy = [self.env.x, self.env.y]

                        # if score drops more than X, extension will be ended
                        if rT1 < best - X:
                            flag = 1
                            break

                        if d == True:
                            flag = 1
                            break
                        #print("Do step stage :",time.time()-test)

                    if flag:
                        break

                bestp = function.check_where(pathx2, pathy2, bestxy)
                pathx2 = pathx2[:bestp+1]
                pathy2 = pathy2[:bestp+1]

            pathx = pathx2[::-1]+list(range(uX1,uX2+1))+pathx1
            pathy = pathy2[::-1]+list(range(uY1,uY2+1))+pathy1

            rT1o += best
            rT2o += best2

            same = np.sum(np.array(self.env.seq1[list(range(uX1,uX2+1))]) == np.array(self.env.seq2[list(range(uY1,uY2+1))]))
            length = uX2-uX1+1

            rT1o += self.env.reward[0]*same+self.env.reward[1]*(length-same)
            rT2o += same

            path = [pathx, pathy]

        else:
            best = 1
            best2 = 1
            flag = 0
            pathx1 = []
            pathy1 = []

            #Forward Extension
            if (uX2+1 <= self.env.sizeS1) and (uY2-1 >= 0):
                self.env.x = uX2 + 1
                self.env.y = uY1 - 1
                pathx1.append(self.env.x)
                pathy1.append(self.env.y)
                bestxy = [self.env.x,self.env.y]

                while j < self.env.sizeS1-uX2+uY2:
                    # Skip process
                    if self.env.seq1[self.env.x]==self.env.rev2[self.env.y]:
                        a = self.skipRC()
                    else:
                        #test = time.time()
                        s1 = processState(self.env.renderRC())
                        #print("Rendering stage :",time.time()-test)
                        #test = time.time()
                        a = sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s1]})
                        #print("Prediction stage :",time.time()-test)
                        #test = time.time()
            
                    for _ in range(np.size(a)):
                        r, d = self.env.stepRC(a[_])
                        pathx1.append(self.env.x)
                        pathy1.append(self.env.y)
                        j += 1
                        rT1 += r
                        rT2 += (r>0)
                        if rT1 >= best:
                            best = rT1
                            best2 = rT2
                            bestxy = [self.env.x, self.env.y]

                        # if score drops more than X, extension will be ended
                        if rT1 < best - X:
                            flag = 1
                            break

                        if d == True:
                            flag = 1
                            break
                        #print("Do step stage :",time.time()-test)

                    if flag:
                        break

                bestp = function.check_where(pathx1, pathy1, bestxy)
                pathx1 = pathx1[:bestp+1]
                pathy1 = pathy1[:bestp+1]

            rT1o += best
            rT2o += best2

            d = False # The state of the game (End or Not)
            rT1 = 1 # Total reward
            rT2 = 1 # Total match
            j = 0

            best = 1
            best2 = 1
            flag = 0
            pathx2 = []
            pathy2 = []

            #Reverse Extension
            if (uX1-1 >= 0) and (uY1+1 <= self.env.sizeS2):
                self.env.x = uX1 - 1
                self.env.y = uY1 + 1
                pathx2.append(self.env.x)
                pathy2.append(self.env.y)
                bestxy = [self.env.x,self.env.y]

                while j < uX1+self.env.sizeS2-uY1:
                    # Skip process
                    if self.env.seq1[self.env.x]==self.env.rev2[self.env.y]:
                        a = self.reverseskipRC()
                    else:
                        #test = time.time()
                        s1 = processState(self.env.renderRCRev())
                        #print("Rendering stage :",time.time()-test)
                        #test = time.time()
                        a = sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s1]})
                        #print("Prediction stage :",time.time()-test)
                        #test = time.time()
            
                    for _ in range(np.size(a)):
                        r, d = self.env.stepRC(10+a[_])
                        pathx2.append(self.env.x)
                        pathy2.append(self.env.y)
                        j += 1
                        rT1 += r
                        rT2 += (r>0)
                        if rT1 >= best:
                            best = rT1
                            best2 = rT2
                            bestxy = [self.env.x, self.env.y]

                        # if score drops more than X, extension will be ended
                        if rT1 < best - X:
                            flag = 1
                            break

                        if d == True:
                            flag = 1
                            break
                        #print("Do step stage :",time.time()-test)

                    if flag:
                        break

                bestp = function.check_where(pathx2, pathy2, bestxy)
                pathx2 = pathx2[:bestp+1]
                pathy2 = pathy2[:bestp+1]

            pathx = pathx2[::-1]+list(range(uX1,uX2+1))+pathx1
            pathy = pathy2[::-1]+list(range(uY1,uY2-1,-1))+pathy1

            rT1o += best
            rT2o += best2

            same = np.sum(np.array(self.env.seq1[list(range(uX1,uX2+1))]) == np.array(self.env.rev2[list(range(uY1,uY2-1,-1))]))
            length = uX2-uX1+1

            rT1o += self.env.reward[0]*same+self.env.reward[1]*(length-same)
            rT2o += same

            path = [pathx, pathy]

        now = time.time()

        return rT1o, rT2o, now - past, j, path

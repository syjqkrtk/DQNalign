import numpy as np
import random
import itertools
import cv2
import scipy.misc
import DQNalign.lcs as lcs
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

Real1 = open("lib/Hepatitis_1.txt", 'r')
Real2 = open("lib/Hepatitis_2.txt", 'r')
Real3 = open("lib/Hepatitis_3.txt", 'r')

RealTemp1 = list(Real1.readline().replace("\n",""))
RealSeq1 = np.zeros(np.size(RealTemp1))
for _ in range(np.size(RealTemp1)):
    RealSeq1[_] = (RealTemp1[_] == 'A') + 2 * (RealTemp1[_] == 'C') + 3 * (RealTemp1[_] == 'G') + 4 * (RealTemp1[_] == 'T') - 1

RealTemp2 = list(Real2.readline().replace("\n",""))
RealSeq2 = np.zeros(np.size(RealTemp2))
for _ in range(np.size(RealTemp2)):
    RealSeq2[_] = (RealTemp2[_] == 'A') + 2 * (RealTemp2[_] == 'C') + 3 * (RealTemp2[_] == 'G') + 4 * (RealTemp2[_] == 'T') - 1

RealTemp3 = list(Real3.readline().replace("\n",""))
RealSeq3 = np.zeros(np.size(RealTemp3))
for _ in range(np.size(RealTemp3)):
    RealSeq3[_] = (RealTemp3[_] == 'A') + 2 * (RealTemp3[_] == 'C') + 3 * (RealTemp3[_] == 'G') + 4 * (RealTemp3[_] == 'T') - 1

Real = open('lib/HEV.txt','r')
HEVname = []
HEVseq = []
for __ in range(47):
    HEVname.append(Real.readline().replace('\n','').replace('>',''))
    RealTemp = Real.readline().replace('\n','')
    RealSeq = np.zeros(len(RealTemp),dtype=int)
    for _ in range(len(RealTemp)):
        RealSeq[_] = (RealTemp[_] == 'A') + 2 * (RealTemp[_] == 'C') + 3 * (RealTemp[_] == 'G') + 4 * (RealTemp[_] == 'T') - 1
    HEVseq.append(RealSeq)
#print(HEVname)
#print(HEVseq)

def readseq(filename):
    file = open(filename,'r')
    seq = file.read().replace('\n', '')
    return seq

def readseqs(filename):
    Real = open('lib/HEV.txt','r')
    HEVname = []
    HEVseq = []
    for __ in range(47):
        HEVname.append(Real.readline().replace('\n','').replace('>',''))
        RealTemp = Real.readline().replace('\n','')
        HEVseq.append(RealTemp)
    return HEVseq

def preprocess(seq1, seq2):
    return lcs.longestSubstring(seq1,seq2)

def zipfian(s,N):
    temp0 = np.array(range(1,N+1))
    temp0 = np.sum(1/temp0**s)
    temp = random.random() * temp0

    for i in range(N):
        temp2 = 1 / ((i + 1) ** s)

        if temp < temp2:
            return i+1
        else:
            temp = temp - temp2

    return 0

class gameEnv():
    def __init__(self, reward, seq_size, win_size, p, maxI, str, seq1=[], seq2=[]):
        # reward must be defined as the score of [match, mismatch, indel]
        self.reward = reward
        self.sizeS1 = seq_size[0]
        self.sizeS2 = seq_size[1]
        self.win_size = win_size
        self.p = p
        self.maxI = maxI
        self.actions = 3
        if str==0:
            a = self.reset()
        elif str > 0:
            a = self.test(str)
        else:
            a = self.final(seq1,seq2)
        plt.imshow(a, interpolation="nearest", cmap="rainbow")
        #plt.show()

    def reset(self):
        self.x = 0
        self.y = 0
        self.seq1 = np.random.randint(4, size=self.sizeS1)
        self.seq2 = np.mod(self.seq1 + (np.random.rand(self.sizeS1) < self.p[0]) * np.random.randint(4, size=self.sizeS1), 4)
        count1 = 0
        count2 = 0
        for kk in range(self.sizeS1):
            if np.random.rand() < self.p[1]:
                indel = zipfian(1.6, self.maxI)
                ranval = np.random.rand()
                if ranval < 1 / 2:
                    temp1 = self.seq1[0:kk + count1]
                    temp4 = self.seq1[kk + count1:]
                    self.seq1 = np.append(np.append(temp1, np.random.randint(4, size=indel)), temp4)
                    count1 = count1 + indel
                else:
                    temp2 = self.seq2[0:kk + count2]
                    temp5 = self.seq2[kk + count2:]
                    self.seq2 = np.append(np.append(temp2, np.random.randint(4, size=indel)), temp5)
                    count2 = count2 + indel

        if np.size(self.seq1) >= self.sizeS1:
            self.seq1 = self.seq1[0:self.sizeS1]
        else:
            tempseq = np.random.randint(4, size=self.sizeS1-np.size(self.seq1))
            self.seq1 = self.seq1.append(tempseq)

        if np.size(self.seq2) >= self.sizeS2:
            self.seq2 = self.seq2[0:self.sizeS2]
        else:
            tempseq = np.random.randint(4, size=self.sizeS2-np.size(self.seq2))
            self.seq2 = self.seq2.append(tempseq)

        self.state = self.renderEnv()

        return self.state

    def final(self, seq1, seq2):
        if np.size(seq1)>0 and np.size(seq2)>0:
            if (seq1[0] == 0) or (seq1[0] == 1) or (seq1[0] == 2) or (seq1[0] == 3) :
                seqTemp1 = seq1
                seqTemp2 = seq2
                self.seq1 = seqTemp1.astype(int)
                self.seq2 = seqTemp2.astype(int)
            else:
                seqTemp1 = np.zeros(len(seq1), dtype=int)
                for _ in range(len(seqTemp1)):
                    seqTemp1[_] = (seq1[_] == 'A') + 2 * (seq1[_] == 'C') + 3 * (seq1[_] == 'G') + 4 * (
                            seq1[_] == 'T') - 1
                seqTemp2 = np.zeros(len(seq2), dtype=int)
                for _ in range(len(seqTemp2)):
                    seqTemp2[_] = (seq2[_] == 'A') + 2 * (seq2[_] == 'C') + 3 * (seq2[_] == 'G') + 4 * (
                            seq2[_] == 'T') - 1
                self.seq1 = seqTemp1.astype(int)
                self.seq2 = seqTemp2.astype(int)
        self.x = 0
        self.y = 0
        self.sizeS1 = np.size(self.seq1)
        self.sizeS2 = np.size(self.seq2)

        self.state = self.renderEnv()
        print("alignlen", self.sizeS1, self.sizeS2)

        return self.state

    def test(self, str):
        self.x = 0
        self.y = 0
        if str == 1:
            self.seq1 = RealSeq1.astype(int)
            self.seq2 = RealSeq2.astype(int)
            self.sizeS1 = np.size(self.seq1)
            self.sizeS2 = np.size(self.seq2)
        elif str == 2:
            self.seq1 = RealSeq1.astype(int)
            self.seq2 = RealSeq3.astype(int)
            self.sizeS1 = np.size(self.seq1)
            self.sizeS2 = np.size(self.seq2)
        elif str < 20000:
            #print(int(np.floor((str-10000)/100)), np.mod((str-10000),100))
            self.seq1 = HEVseq[int(np.floor((str-10000)/100))]
            self.seq2 = HEVseq[np.mod((str-10000),100)]
            self.sizeS1 = np.size(self.seq1)
            self.sizeS2 = np.size(self.seq2)
            print(np.size(self.seq1), np.size(self.seq2))
        else:
            Ecoli1 = open("lib\\Ecoli_1.txt", 'r')
            Ecoli2 = open("lib\\Ecoli_2.txt", 'r')
            
            EcoliTemp1 = list(Ecoli1.readline().replace("\n",""))
            EcoliSeq1 = np.zeros(np.size(EcoliTemp1))
            for _ in range(np.size(EcoliTemp1)):
                EcoliSeq1[_] = (EcoliTemp1[_] == 'A') + 2 * (EcoliTemp1[_] == 'C') + 3 * (EcoliTemp1[_] == 'G') + 4 * (EcoliTemp1[_] == 'T') - 1

            EcoliTemp2 = list(Ecoli2.readline().replace("\n",""))
            EcoliSeq2 = np.zeros(np.size(EcoliTemp2))
            for _ in range(np.size(EcoliTemp2)):
                EcoliSeq2[_] = (EcoliTemp2[_] == 'A') + 2 * (EcoliTemp2[_] == 'C') + 3 * (EcoliTemp2[_] == 'G') + 4 * (EcoliTemp2[_] == 'T') - 1
            self.seq1 = EcoliSeq1.astype(int)
            self.seq2 = EcoliSeq2.astype(int)
            self.sizeS1 = np.size(self.seq1)
            self.sizeS2 = np.size(self.seq2)
            print(self.sizeS1)
            print(self.sizeS2)

        self.state = self.renderEnv()

        return self.state

    def moveChar(self, action):
        # 0 - Match, 1 - Seq1 Insertion, 2 - Seq2 Insertion)
        if action == 0:
            if self.seq1[self.x] == self.seq2[self.y]:
                reward = self.reward[0]
            else:
                reward = self.reward[1]
            self.x += 1
            self.y += 1

        if action == 1:
            reward = self.reward[2]
            self.x += 1

        if action == 2:
            reward = self.reward[2]
            self.y += 1

        if self.x >= self.sizeS1:
            done = True
        elif self.y >= self.sizeS2:
            done = True
        else:
            done = False

        #print(self.x)
        #print(self.y)
        #print(done)
        return reward, done

    def renderEnv(self):
        a = np.zeros([self.win_size + 2, 4, 4]).astype(int)

        if self.x+self.win_size > self.sizeS1:
            #print(self.win_size)
            #print(self.seq1[self.x:self.sizeS1])
            i = np.zeros([self.sizeS1-self.x,4])
            i[np.arange(self.sizeS1-self.x),self.seq1[self.x:self.sizeS1]]=1
            a[1:1+self.sizeS1-self.x, 1, :] = i
            a[self.sizeS1-self.x:,1,:]=1
        else:
            #print(self.win_size)
            #print(self.seq1[self.x:self.x+self.win_size])
            i = np.zeros([self.win_size,4])
            i[np.arange(self.win_size),self.seq1[self.x:self.x+self.win_size]]=1
            a[1:-1, 1, :] = i

        if self.y+self.win_size > self.sizeS2:
            #print(self.win_size)
            #print(self.seq1[self.y:self.sizeS2])
            i = np.zeros([self.sizeS2-self.y,4])
            i[np.arange(self.sizeS2-self.y),self.seq2[self.y:self.sizeS2]]=1
            a[1:1+self.sizeS2-self.y, 2,:] = i
            a[self.sizeS2-self.y:,2,:]=1
        else:
            #print(self.win_size)
            #print(self.seq1[self.y:self.y+self.win_size])
            i = np.zeros([self.win_size,4])
            i[np.arange(self.win_size),self.seq2[self.y:self.y+self.win_size]]=1
            a[1:-1, 2, :] = i

        r = (1-a[:,:,0])*(1-a[:,:,3])
        g = (1-a[:,:,1])*(1-a[:,:,3])
        b = (1-a[:,:,2])*(1-a[:,:,3])
        r = scipy.misc.imresize(r, [3*self.win_size+6, 12, 1], interp='nearest')
        g = scipy.misc.imresize(g, [3*self.win_size+6, 12, 1], interp='nearest')
        b = scipy.misc.imresize(b, [3*self.win_size+6, 12, 1], interp='nearest')

        a = np.stack([r,g,b], axis=2)

        return a

    def step(self, action):
        reward, done = self.moveChar(action)
        state = self.renderEnv()
        return state, reward, done

    def teststep(self, action):
        if action > 2:
            self.x += action-2
            self.y += action-2
            reward = self.reward[0]*(action-2)

            if self.x >= self.sizeS1:
                done = True
            elif self.y >= self.sizeS2:
                done = True
            else:
                done = False
        else:
            reward, done = self.moveChar(action)

        return reward, done

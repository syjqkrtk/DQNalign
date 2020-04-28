import matplotlib.pyplot as plt
import numpy as np
import time
import random

n_length = 2       # The number of sequences
l_seqs = 1000
l_seqs2 = 1000
reward = [1,-1,-1]    # reward for each (match, gap, mismatch)

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

def rewardfun(s1, s2, a):
    result = 0
    if s1 == s2 and a == 0:
        result = result + reward[0]
    elif s1 != s2 and a == 0:
        result = result + reward[2]
    else:
        result = result + reward[1]

    return result

def rewardarray(s1, s2, a):
    if a == 0:
        result = [reward[1]]*len(s2)+(reward[0]-reward[1])*np.equal(s2,s1)
    else:
        result = [reward[2]]*len(s2)

    return result
    
def seqgen(l_seqs,n_length):
    seq1 = np.random.randint(4, size=l_seqs)
    seq2 = np.mod(seq1 + (np.random.rand(l_seqs)  < 0.1)*np.random.randint(4, size=l_seqs),4)
    count1 = 0
    count2 = 0
    for kk in range(l_seqs):
        if np.random.rand() < 0.05:
            indel = zipfian(1.6,3)
            ranval = np.random.rand()
            if ranval < 1/2:
                temp1 = seq1[0:kk+count1]
                temp4 = seq1[kk+count1:]
                seq1 = np.append(np.append(temp1,np.random.randint(4, size=indel)),temp4)
                count1 = count1 + indel
            else:
                temp2 = seq2[0:kk+count2]
                temp5 = seq2[kk+count2:]
                seq2 = np.append(np.append(temp2,np.random.randint(4, size=indel)),temp5)
                count2 = count2 + indel

    extendseq = np.zeros(n_length - 1, dtype=int)-1
    seq1 = np.append(seq1, extendseq)
    seq2 = np.append(seq2, extendseq)

    return seq1, seq2

def align(s1, s2):
    x_length = len(s1)
    y_length = len(s2)
    score = np.zeros([x_length+1,y_length+1], dtype=int)
    pathx = np.zeros([x_length,y_length], dtype=int)
    pathy = np.zeros([x_length,y_length], dtype=int)
    score[0,:] = range(0,-y_length-1,-1)
    score[:,0] = range(0,-x_length-1,-1)

    for i in range(1,x_length+1):
        for j in range(1,y_length+1):
            temp = [score[i-1,j-1]+rewardfun(s1[i-1],s2[j-1],0),score[i-1,j]+rewardfun(s1[i-1],s2[j-1],1),score[i,j-1]+rewardfun(s1[i-1],s2[j-1],2)]
            temp2 = max(temp)
            score[i, j] = temp2
            temp3 = temp.index(temp2)
            if temp3 == 0:
                pathx[i-1,j-1] = i-2
                pathy[i-1,j-1] = j-2
            elif temp3 == 1:
                pathx[i-1,j-1] = i-2
                pathy[i-1,j-1] = j-1
            else:
                pathx[i-1,j-1] = i-1
                pathy[i-1,j-1] = j-2

    match = 0
    x, y = np.where(score == np.max(score))
    x = x[0]-1
    y = y[0]-1
    while x >= 0 and y >= 0:
        xp = pathx[x,y]
        yp = pathy[x,y]
        if not (x == xp or y == yp):
            if (s1[x] == s2[y]):
                match += 1
        x = xp
        y = yp

    return max(np.max(score[-1,:]),np.max(score[:,-1])), match

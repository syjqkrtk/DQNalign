import numpy as np
import random

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

def check_exist(path, uX1, uX2, uY1, uY2):
    #이거 새로 함수 짜긴 해야할듯
    for temppath in path:
        if (uX1 in temppath[0]) and (uX2 in temppath[0]) and (uY1 in temppath[1]) and (uY2 in temppath[1]):
            return 1

    return 0


def check_where(pathx, pathy, bestxy):
    candix = np.where(np.array(pathx) == bestxy[0])
    candiy = np.where(np.array(pathy) == bestxy[1])

    return np.intersect1d(candix, candiy)[0]

def get_reverse(seq):
    seq = seq.replace("A","O")
    seq = seq.replace("T","A")
    seq = seq.replace("O","T")

    seq = seq.replace("a","o")
    seq = seq.replace("t","a")
    seq = seq.replace("o","t")

    seq = seq.replace("C","O")
    seq = seq.replace("G","C")
    seq = seq.replace("O","G")

    seq = seq.replace("c","o")
    seq = seq.replace("g","c")
    seq = seq.replace("o","g")

    return seq


def sortalign(uX1, uX2, uY1, uY2):
    tX1 = np.array(uX1)
    tX2 = np.array(uX2)
    tY1 = np.array(uY1)
    tY2 = np.array(uY2)
    diff = tX2-tX1
    index = np.argsort(diff)[::-1]
    
    return list(tX1[index]), list(tX2[index]), list(tY1[index]), list(tY2[index])

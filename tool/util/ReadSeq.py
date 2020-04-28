import numpy as np

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

def seqtoint(seq):
    data = []
    for i in range(len(seq)):
        if seq[i] == 'A':
            data.append(0)
        if seq[i] == 'C':
            data.append(1)
        if seq[i] == 'G':
            data.append(2)
        if seq[i] == 'T':
            data.append(3)

    return data

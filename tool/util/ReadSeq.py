import numpy as np

def readseq(filename):
    file = open(filename,'r')
    seq = file.read().replace('\n', '')
    file.close()

    return seq

def readseqs(filename):
    Real = open('lib/HEV.txt','r')
    HEVname = []
    HEVseq = []
    for __ in range(47):
        HEVname.append(Real.readline().replace('\n','').replace('>',''))
        RealTemp = Real.readline().replace('\n','')
        HEVseq.append(RealTemp)

    Real.close()

    return HEVseq

def splitseqs(filename):
    Real = open('lib/HEV.txt','r')
    HEVname = []
    HEVseq = []
    for __ in range(47):
        HEVname.append(Real.readline().replace('\n','').replace('>',''))
        RealTemp = Real.readline().replace('\n','')
        HEVseq.append(RealTemp)

    Real.close()

    for _ in range(47):
        file = open("lib/HEV/HEV_"+str(_)+".fasta",'w')
        file.write(">"+HEVname[_]+"\n")
        file.write(HEVseq[_])
        file.close()

    return HEVname

<<<<<<< HEAD
def inttoseq(data):
    seq = []
    for i in range(len(data)):
        if data[i] == 0:
            seq.append('A')
        if data[i] == 1:
            seq.append('C')
        if data[i] == 2:
            seq.append('G')
        if data[i] == 3:
            seq.append('T')

    return data

=======
>>>>>>> aa3cc47a779f10b4c9f586ff4d9f620328b6dda2
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

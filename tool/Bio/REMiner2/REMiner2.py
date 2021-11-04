import ctypes
import numpy as np
import os
import gc
import time

dirname = os.path.dirname(__file__)
lib = ctypes.cdll.LoadLibrary(os.path.join(dirname,"REMinerII.so"))

def Seq2File(seq1, seq2):
    if type(seq1[0]) is int:
        seq1 = inttoseq(seq1)
    if type(seq2[0]) is int:
        seq2 = inttoseq(seq2)

    file = open(os.path.join(dirname,"seq1.txt"),"w")
    file.write(">seq1\n");
    file.write(seq1)
    file.close()

    file = open(os.path.join(dirname,"seq2.txt"),"w")
    file.write(">seq2\n");
    file.write(seq2)
    file.close()

    return seq1, seq2


def SetParameter(mode, fname1 = os.path.join(dirname,"seq1.txt"), fname2 = os.path.join(dirname,"seq2.txt")):
    file = open("param.txt","w")

    file.write("QUERY_FILE	"+fname1+"	// input file name (name size limit: 256)\n")
    file.write("DATABASE_FILE	"+fname2+"	// input file name (name size limit: 256)\n")
    file.write("WORD_SIZE	14		// WORD size (W, supported to 16 )\n")
    file.write("ALLOW_SIZE	1		// allowable size per word (m, suport 0 or 1)\n")
    file.write("SPACE_SIZE	2		// SPACE size (SP)\n")
    file.write("MIN_SEED_LEN	100		// minimum length to be a seed (L)\n")
    file.write("SCORE_MAT	1		// match score (integer)\n")
    file.write("SCORE_MIS	-2		// mismatch score (integer)\n")
    file.write("SCORE_THR	-10		// score threshold (in ungapped extension)\n")
    file.write("GREEDY_X	30		// value of X in greedy algorithm (in gapped extension)\n")
    file.write("GREEDY_MIN_L	-10240		// maximum value of insertion\n")
    file.write("GREEDY_MAX_U	10240		// maximum value of deletion\n")
    file.write("WD_SIZE		20		// window size for filtering\n")
    file.write("T_THR		0.6		// threshold for filtering\n")
    file.write("ALIGN_MODE	"+str(mode)+"		// alignment mode (0: entire alignment process with REMINER2, 1: do until seeding process, extension process will be done at outer program)\n")

    file.close()


def REMiner2(mode, seq1, seq2):
    if (".fasta" in seq1) or (".txt" in seq1):
        SetParameter(mode, seq1, seq2)
    else:
        Seq2File(seq1, seq2)
        SetParameter(mode)

    gc.collect()

    SeedNum = lib.main()
    gc.collect()

    return SeedNum

def GetRE(SeedNum, printRE = True):
    ArrayType = ctypes.c_int * SeedNum
    lib.GetResFile.restype = ctypes.POINTER(ArrayType)
    gc.collect()

    uX1 = [i for i in lib.GetResFile(ctypes.c_int(SeedNum), ctypes.c_int(0), ctypes.c_bool(printRE)).contents]
    uX2 = [i for i in lib.GetResFile(ctypes.c_int(SeedNum), ctypes.c_int(1), ctypes.c_bool(False)).contents]
    uY1 = [i for i in lib.GetResFile(ctypes.c_int(SeedNum), ctypes.c_int(2), ctypes.c_bool(False)).contents]
    uY2 = [i for i in lib.GetResFile(ctypes.c_int(SeedNum), ctypes.c_int(3), ctypes.c_bool(False)).contents]
    gc.collect()

    #print(uX1)
    #print(uX2)
    #print(uY1)
    #print(uY2)

    return uX1, uX2, uY1, uY2

def GetSEED(SeedNum, printRE = True):
    ArrayType = ctypes.c_int * SeedNum
    lib.GetSeedFile.restype = ctypes.POINTER(ArrayType)
    gc.collect()

    uX1 = [i for i in lib.GetSeedFile(ctypes.c_int(SeedNum), ctypes.c_int(0), ctypes.c_bool(printRE)).contents]
    uX2 = [i for i in lib.GetSeedFile(ctypes.c_int(SeedNum), ctypes.c_int(1), ctypes.c_bool(False)).contents]
    uY1 = [i for i in lib.GetSeedFile(ctypes.c_int(SeedNum), ctypes.c_int(2), ctypes.c_bool(False)).contents]
    uY2 = [i for i in lib.GetSeedFile(ctypes.c_int(SeedNum), ctypes.c_int(3), ctypes.c_bool(False)).contents]
    gc.collect()

    i = 0

    for i in range(SeedNum):
        if (uX1[i] == 0) and (uX2[i] == 0) and (uY1[i] == 0) and (uY2[i] == 0):
            break

    uX1 = uX1[:i]
    uX2 = uX2[:i]
    uY1 = uY1[:i]
    uY2 = uY2[:i]

    #print(uX1)
    #print(uX2)
    #print(uY1)
    #print(uY2)

    return uX1, uX2, uY1, uY2

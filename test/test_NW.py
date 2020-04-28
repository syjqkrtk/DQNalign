import numpy as np
import DQNalign.tool.Bio.NW as NW
import DQNalign.tool.util.ReadSeq as readseq
import time

start = time.time()
startdate = time.localtime()

rawseq = readseq.readseqs('lib/HEV.txt')
seq = []
for _ in range(47):
    seq.append(readseq.seqtoint(rawseq[_]))

for _ in range(47):
    #print(_)
    for __ in range(_+1,47):
        seq1 = seq[_]
        seq2 = seq[__]
        past = time.time()
        score, match = NW.align(seq1,seq2)
        now = time.time()
        
        print("test",_,__)
        print("result", score, match, "time", str(now-past)+"s", str(now-start)+"s")
    
        filename = "result/NW/result%04d%02d%02d%02d%02d%02d.txt" % (
            startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
            startdate.tm_sec)

        file = open(filename,"a")
        file.write(str(match)+" "+str(now-past)+" "+str(now-start)+"\n")
        file.close()

import numpy as np
import DQNalign.tool.Bio.NW as NW
import DQNalign.tool.util.ReadSeq as readseq
import time

start = time.time()
startdate = time.localtime()

'''
rawseq = readseq.readseqs('lib/HEV.txt')
seq = []
for _ in range(47):
    seq.append(readseq.seqtoint(rawseq[_]))

for _ in range(47):
    #print(_)
    for __ in range(_+1,47):
        seq1 = seq[_]
        seq2 = seq[__]
        
        print("test",_,__)
        print("result", len(seq1), len(seq2))
    
        filename = "result/NW/result%04d%02d%02d%02d%02d%02d.txt" % (
            startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
            startdate.tm_sec)

        file = open(filename,"a")
        file.write(str(_)+"_"+str(__)+" "+str(len(seq1))+" "+str(len(seq2))+"\n")
        file.close()
'''

for _ in range(34685):
    len1 = len(readseq.readseq2('lib/Rat/'+str(_)+'.txt'))
    len2 = len(readseq.readseq2('lib/Mouse/'+str(_)+'.txt'))
    print('Rat_'+str(_)+' : '+str(len1))
    print('Mouse_'+str(_)+' : '+str(len2))
    
    filename = "result/NW/result%04d%02d%02d%02d%02d%02d.txt" % (
        startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
        startdate.tm_sec)

    file = open(filename,"a")
    file.write(str(_)+" "+str(len1)+" "+str(len2)+"\n")
    file.close()

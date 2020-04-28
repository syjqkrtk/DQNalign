import numpy as np
import DQNalign.tool.Bio.conventional as conventional
import DQNalign.tool.util.ReadSeq as readseq
import time
import tensorflow as tf
import DQNalign.flags as flags
FLAGS = tf.app.flags.FLAGS

start = time.time()
startdate = time.localtime()

seq1 = readseq.readseq('lib/Ecoli_1.txt')
seq2 = readseq.readseq('lib/Ecoli_2.txt')

past = time.time()
c = conventional.Clustal(True,seq1,seq2)
match = c.pair_align()
now = time.time()
        
print("Ecoli test")
print("result", match, "time", str(now-past)+"s", str(now-start)+"s")
    
filename = "result/Clustal/result%04d%02d%02d%02d%02d%02d_Ecoli.txt" % (
    startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
    startdate.tm_sec)

file = open(filename,"a")
file.write(str(match)+" "+str(now-past)+" "+str(now-start)+"\n")
file.close()
    
filename = "img/Clustal/result%04d%02d%02d%02d%02d%02d_Ecoli" % (
    startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
    startdate.tm_sec)

if FLAGS.print_align:
    filename = "align/Clustal/result%04d%02d%02d%02d%02d%02d" % (
        startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
        startdate.tm_sec)
    c.print(filename+"_Ecoli.txt")

if FLAGS.show_align:
    c.display(filename+".jpg")

'''
seq = readseq.readseqs('lib/HEV.txt')

for _ in range(47):
    #print(_)
    for __ in range(_+1,47):
        seq1 = seq[_]
        seq2 = seq[__]
        past = time.time()
        c = conventional.Clustal(True,seq1,seq2)
        match = c.pair_align()
        now = time.time()
        
        print("test",_,__)
        print("result", match, "time", str(now-past)+"s", str(now-start)+"s")
    
        filename = "result/Clustal/result%04d%02d%02d%02d%02d%02d.txt" % (
            startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
            startdate.tm_sec)

        file = open(filename,"a")
        file.write(str(match)+" "+str(now-past)+" "+str(now-start)+"\n")
        file.close()

        if FLAGS.print_align:
            filename = "align/Clustal/result%04d%02d%02d%02d%02d%02d" % (
                startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                startdate.tm_sec)
            c.print(filename+"_"+str(_)+"_"+str(__)+".txt")
        
        if FLAGS.show_align:
            filename = "img/Clustal/result%04d%02d%02d%02d%02d%02d" % (
                startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                startdate.tm_sec)
            c.display(filename+"_"+str(_)+"_"+str(__)+".jpg")
'''

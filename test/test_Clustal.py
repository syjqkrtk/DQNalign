import numpy as np
import DQNalign.tool.Bio.conventional as conventional
import DQNalign.tool.util.ReadSeq as readseq
import time
import tensorflow as tf
import DQNalign.flags as flags
FLAGS = tf.app.flags.FLAGS

start = time.time()
startdate = time.localtime()

'''
seq1 = readseq.readseq('lib/Prochlorococcus_1.fna')
seq2 = readseq.readseq('lib/Prochlorococcus_2.fna')

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

for pairs in [["Homo_sapiens_BRCA","Mus_musculus_BRCA"],["Mus_musculus_BRCA","Rattus_norvegicus_BRCA"],["Rattus_norvegicus_BRCA","Homo_sapiens_BRCA"],["Homo_sapiens_ELK1","Mus_musculus_ELK1"],["Mus_musculus_ELK1","Rattus_norvegicus_ELK1"],["Rattus_norvegicus_ELK1","Homo_sapiens_ELK1"],["Homo_sapiens_CCDC91","Mus_musculus_CCDC91"],["Mus_musculus_CCDC91","Rattus_norvegicus_CCDC91"],["Rattus_norvegicus_CCDC91","Homo_sapiens_CCDC91"],["Homo_sapiens_FOXP2","Mus_musculus_FOXP2"],["Mus_musculus_FOXP2","Rattus_norvegicus_FOXP2"],["Rattus_norvegicus_FOXP2","Homo_sapiens_FOXP2"]]:
    seq1 = readseq.readseq('lib/Mammal/'+pairs[0]+'.txt')
    seq2 = readseq.readseq('lib/Mammal/'+pairs[1]+'.txt')
    #print(_)
    past = time.time()

    c = conventional.Clustal(True,seq1,seq2)
    match = c.pair_align()

    now = time.time()
    #NWresult = np.max(NW.match(alignment.HEVseq[_],alignment.HEVseq[__]))
    print(pairs, len(seq1), len(seq2), match, "time", str(now-past)+"s", str(now-start)+"s")
    
    filename = "result/Clustal/result%04d%02d%02d%02d%02d%02d.txt" % (
        startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
        startdate.tm_sec)

    file = open(filename,"a")
    file.write(str(match)+" "+str(now-past)+" "+str(now-start)+"\n")
    file.close()

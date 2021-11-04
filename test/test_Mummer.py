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
past = time.time()
m = conventional.MUMmer(True,"lib/Ecoli_31.fasta","lib/Ecoli_42.fasta",["Ecoli_31","Ecoli_42"],"result/MUMmer/result%04d%02d%02d%02d%02d%02d_Ecoli" % (
        startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min, startdate.tm_sec))
m.align()
coords1, coords2, aligns1, aligns2, score = m.export_info()
now = time.time()
        
print("Ecoli test")
print("result", score, "time", str(now-past)+"s", str(now-start)+"s")
    
filename = "result/MUMmer/result%04d%02d%02d%02d%02d%02d_Ecoli.txt" % (
    startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
    startdate.tm_sec)

file = open(filename,"a")
file.write(str(match)+" "+str(now-past)+" "+str(now-start)+"\n")
file.close()
    
filename = "img/MUMmer/result%04d%02d%02d%02d%02d%02d_Ecoli" % (
    startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
    startdate.tm_sec)

if FLAGS.print_align:
    filename = "align/MUMmer/result%04d%02d%02d%02d%02d%02d" % (
        startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
        startdate.tm_sec)
    m.print(filename+"_Ecoli.txt")

if FLAGS.show_align:
    m.display(filename+".jpg")

seq = readseq.splitseqs('lib/HEV.txt')

for _ in range(47):
    #print(_)
    for __ in range(_+1,47):
        seq1 = "lib/HEV/HEV_"+str(_)+".fasta"
        seq2 = "lib/HEV/HEV_"+str(__)+".fasta"
        past = time.time()
        m = conventional.MUMmer(True,seq1,seq2,[seq[_],seq[__]],"result/MUMmer/result%04d%02d%02d%02d%02d%02d" % (
        startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min, startdate.tm_sec))
        m.align()
        coords1, coords2, aligns1, aligns2, score = m.export_info()
        now = time.time()
        
        print("test",_,__)
        print("result", score, "time", str(now-past)+"s", str(now-start)+"s")
    
        filename = "result/MUMmer/result%04d%02d%02d%02d%02d%02d.txt" % (
            startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
            startdate.tm_sec)

        file = open(filename,"a")
        file.write(str(score)+" "+str(now-past)+" "+str(now-start)+"\n")
        file.close()

        if FLAGS.print_align:
            filename = "align/MUMmer/result%04d%02d%02d%02d%02d%02d" % (
                startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                startdate.tm_sec)
            m.print(filename+"_"+str(_)+"_"+str(__))
        
        if FLAGS.show_align:
            filename = "img/MUMmer/result%04d%02d%02d%02d%02d%02d" % (
                startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                startdate.tm_sec)
            m.display(filename+"_"+str(_)+"_"+str(__)+".jpg")

'''

for pairs in [["Homo_sapiens_BRCA","Mus_musculus_BRCA"],["Mus_musculus_BRCA","Rattus_norvegicus_BRCA"],["Rattus_norvegicus_BRCA","Homo_sapiens_BRCA"],["Homo_sapiens_ELK1","Mus_musculus_ELK1"],["Mus_musculus_ELK1","Rattus_norvegicus_ELK1"],["Rattus_norvegicus_ELK1","Homo_sapiens_ELK1"],["Homo_sapiens_CCDC91","Mus_musculus_CCDC91"],["Mus_musculus_CCDC91","Rattus_norvegicus_CCDC91"],["Rattus_norvegicus_CCDC91","Homo_sapiens_CCDC91"],["Homo_sapiens_FOXP2","Mus_musculus_FOXP2"],["Mus_musculus_FOXP2","Rattus_norvegicus_FOXP2"],["Rattus_norvegicus_FOXP2","Homo_sapiens_FOXP2"]]:
    #print(_)
    past = time.time()

    m = conventional.MUMmer(True,'lib/Mammal2/'+pairs[0]+'.fasta','lib/Mammal2/'+pairs[1]+'.fasta',pairs,"result/MUMmer/result%04d%02d%02d%02d%02d%02d_Mammal" % (
        startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min, startdate.tm_sec))
    m.align()
    coords1, coords2, aligns1, aligns2, score = m.export_info()

    now = time.time()
    #NWresult = np.max(NW.match(alignment.HEVseq[_],alignment.HEVseq[__]))
    print(pairs, score, "time", str(now-past)+"s", str(now-start)+"s")
    
    filename = "result/MUMmer/result%04d%02d%02d%02d%02d%02d.txt" % (
        startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
        startdate.tm_sec)

    file = open(filename,"a")
    file.write(str(score)+" "+str(now-past)+" "+str(now-start)+"\n")
    file.close()

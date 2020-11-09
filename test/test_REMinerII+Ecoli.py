from DQNalign.tool.RL.agent import Agent
from DQNalign.tool.RL.alignment import Pairwise
from DQNalign.tool.RL.Learning import *
import DQNalign.tool.Bio.lcs as lcs
import DQNalign.tool.Bio.REMiner2.REMiner2 as REMiner2
import DQNalign.tool.util.ReadSeq as readseq
import DQNalign.tool.util.RecordAlign as recordalign
import DQNalign.tool.util.function as function
from importlib import *
import copy
import time
import os
import cv2
import tensorflow as tf
import DQNalign.flags as flags
FLAGS = tf.app.flags.FLAGS
param = import_module('DQNalign.param.'+FLAGS.network_set)

class game_env():
    def __init__(self):
        self.l_seq = [8000, 8000]
        self.win_size = 100
        self.maxI = 10 # maximum indel length
        self.p = [0.1,0.02] # The probability of SNP, indel
        self.reward = [1,-1,-2] # Alignment score of the match, mismatch, indel

        self.path = "./network/"+FLAGS.model_name+"/"+str(self.win_size)+'_'+str(param.n_step)

train_env = game_env()

class model():
    def __init__(self):
        self.param = param
        self.env = Pairwise(train_env,-1,Z=self.param.Z)
        self.LEARNING_RATE = 0.0000001

        tf.reset_default_graph()
        
        """ Define Deep reinforcement learning network """
        if FLAGS.model_name == "DQN":
            self.mainQN = Qnetwork(self.param.h_size,self.env,self.LEARNING_RATE,self.param.n_step)
            self.targetQN = Qnetwork(self.param.h_size,self.env,self.LEARNING_RATE,self.param.n_step)
            self.trainables = tf.trainable_variables()
            self.targetOps = updateTargetGraph(self.trainables, self.param.tau)
        elif FLAGS.model_name == "SSD":
            self.mainQN = SSDnetwork(self.param.h_size,self.env,"main",self.LEARNING_RATE,self.param.n_step)
            self.targetQN = SSDnetwork(self.param.h_size,self.env,"target",self.LEARNING_RATE,self.param.n_step)
            self.trainables = tf.trainable_variables()
            self.targetOps = updateTargetGraph(self.trainables, self.param.tau)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

train_model = model()
init = train_model.init
saver = train_model.saver

seq1 = readseq.readseq('lib/Ecoli_31.txt')
seq2 = readseq.readseq('lib/Ecoli_42.txt')

if not os.path.exists(train_env.path):
    os.makedirs(train_env.path)

if np.size(os.listdir(train_env.path)) > 0:
    resume = FLAGS.resume
else:
    resume = False

""" Main test step """
with tf.Session() as sess:
    if FLAGS.use_GPU:
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    else:
        sess = tf.Session(config=tf.ConfigProto(device_count={'CPU': 0}))

    sess.run(init)
    agent = Agent(FLAGS, False, train_env, train_model)

    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(train_env.path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    
    start = time.time()
    startdate = time.localtime()

    X = 100	# Greedy-X algorithm parameter

    agent.set(seq1,seq2)
    path = []
    score = []
    ptime = []

    if FLAGS.show_align:
        dot_plot = 255*np.ones((len(seq1),len(seq2)))
        for i in range(lcslen):
            dot_plot[start1+i,start2+i]=0
    if FLAGS.print_align:
        record = recordalign.record_path()

    print("Ecoli test")
    print("raw seq len",len(seq1),len(seq2))

    past = time.time()
    SeedNum = REMiner2.REMiner2(1, seq1, seq2)
    uX1, uX2, uY1, uY2 = REMiner2.GetSEED(SeedNum,True)

    uX1, uX2, uY1, uY2 = function.sortalign(uX1, uX2, uY1, uY2)

    print("Pre-processing stage is completed : "+str(np.size(uX1))+" seeds are found")
    print("Time spent : "+str(time.time()-past))


    #순서를 다시 잡도록 노력하자, RC 부분이랑 나눠서 순서 정하기


    for i in range(np.size(uX1)):
        now = time.time()
        if function.check_exist(path, uX1[i], uX2[i], uY1[i], uY2[i]):
            continue
        else:
            rT1, rT2, processingtime, j, temppath = agent.Local(sess, uX1[i], uX2[i], uY1[i], uY2[i], X)
            score.append([rT1,rT2])
            ptime.append(processingtime)
            path.append(temppath)

        print("---------------------------")
        print("Seed # "+str(i)+" / "+str(np.size(uX1))+" is processed")
        print("Time spent : "+str(time.time()-now))
        print("Full-time spent : "+str(time.time()-past))
        print("Seed position : ("+str(uX1[i])+"-"+str(uX2[i])+" , "+str(uY1[i])+"-"+str(uY2[i])+")")
        print("Score before post-processing : "+str(rT1)+" & exact match : "+str(rT2))

    now = time.time()

    filename = "result/REMiner2/result%04d%02d%02d%02d%02d%02d_%d_%d.txt" % (
        startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
        startdate.tm_sec, train_env.win_size, X)

    file = open(filename,"a")

    print("---------------------------")
    for i in range(len(path)):
        print(i,"th alignment")
        print("align position : ",path[i][0][0],path[i][0][-1],path[i][1][0],path[i][1][-1])
        print("align length : ",len(path[i][0]))
        print("score : ",score[i][0])
        print("exact match : ",score[i][1])
        print("identity : ",score[i][1]/len(path[i][0]))
        print("processing time : ",ptime[i])
        print("total time : ",now-start)
        print("---------------------------")

        file.write(str(i)+"th alignment"+"\n")
        file.write("align position : "+str(path[i][0][0])+" "+str(path[i][0][-1])+" "+str(path[i][1][0])+" "+str(path[i][1][-1])+"\n")
        file.write("align length : "+str(len(path[i][0]))+"\n")
        file.write("score : "+str(score[i][0])+"\n")
        file.write("exact match : "+str(score[i][1])+"\n")
        file.write("identity : "+str(score[i][1]/len(path[i][0]))+"\n")
        file.write("processing time : "+str(ptime[i])+"\n")
        file.write("total time : "+str(now-start)+"\n")
        file.write("---------------------------"+"\n")

    file.close()

    """
    if FLAGS.show_align:
        path로 이미지 그리는 함수
        filename = "img/"+FLAGS.model_name+"/result%04d%02d%02d%02d%02d%02d_%d_%d" % (
            startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
            startdate.tm_sec, train_env.win_size, train_env.maxI)
        cv2.imwrite(filename+"_"+str(_)+"_"+str(__)+".jpg",dot_plot)
    """

    if FLAGS.print_align:
        record.set(seq1,seq2,path)

        filename = "align/REMiner2/%d_%d/result%04d%02d%02d%02d%02d%02d_%d_%d.txt" % (train_env.win_size, train_env.maxI,
            startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
            startdate.tm_sec, train_env.win_size, X)
        file = open(filename,"w")

        file.write("DQNalign Project v2.0\n")
        file.write("Local alignment algorithm (REMINER II) with deep reinforcement learning based heuristic alignment agent (DQNalign)\n")
        file.write("Sequence 1 : Ecoli_31, length : "+str(len(seq1))+"\n")
        file.write("Sequence 2 : Ecoli_42, length : "+str(len(seq2))+"\n")
        file.write("\n")

        record.print(file)

        file.close()

    past = now

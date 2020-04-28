from DQNalign.tool.RL.agent import Agent
from DQNalign.tool.RL.alignment import Pairwise
from DQNalign.tool.RL.Learning import *
import DQNalign.tool.Bio.lcs as lcs
import DQNalign.tool.Bio.conventional as conventional
import DQNalign.tool.util.ReadSeq as readseq
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
        self.reward = [1,-1,-1] # Alignment score of the match, mismatch, indel

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

seq = readseq.readseqs('lib/HEV.txt')

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

    for _ in range(47):
        #print(_)
        for __ in range(_+1,47):
            seq1 = seq[_]
            seq2 = seq[__]
            rT2o = 0
            jo = 0
            processingtimeo = 0

            print("test",_,__)
            print("raw seq len",len(seq1),len(seq2))

            past = time.time()

            if FLAGS.show_align:
                dot_plot = 255*np.ones((len(seq1),len(seq2)))
            if FLAGS.print_align:
                record = recordalign.record_align()

            c = conventional.Clustal(True,seq1,seq2)
            anchors, anchore = c.preprocess()

            if (anchors[0][0] > 0) and (anchors[0][1] > 0):
                agent.set(seq1[anchors[0][0]-1::-1]+"A",seq2[anchors[0][1]-1::-1]+"A")
                if FLAGS.show_align and FLAGS.print_align:
                    rT1, rT2, processingtime, j, dot_plot1 = agent.play(sess,record)
                    dot_plot[:anchors[0][0],:anchors[0][1]] = dot_plot1[::-1,::-1]
                    record.reverse(anchors[0][0]-1,anchors[0][1]-1)
                elif FLAGS.show_align:
                    rT1, rT2, processingtime, j, dot_plot1 = agent.play(sess)
                    dot_plot[:anchors[0][0],:anchors[0][1]] = dot_plot1[::-1,::-1]
                elif FLAGS.print_align:
                    rT1, rT2, processingtime, j = agent.play(sess, record)
                    record.reverse(anchors[0][0]-1,anchors[0][1]-1)
                else:
                    rT1, rT2, processingtime, j = agent.play(sess)

                rT2o += rT2
                processingtimeo += processingtime
                jo += j

            for i in range(len(anchors)-1):
                stemp = 0
                if FLAGS.show_align:
                    for k in range(anchore[i][0]-anchors[i][0]):
                        dot_plot[anchors[i][0]+k,anchors[i][1]+k]=0
                if FLAGS.print_align:
                    stemp = record.record([anchors[i][0],anchore[i][0]],[anchors[i][1],anchore[i][1]],-1,seq1[anchors[i][0]:anchore[i][0]],seq2[anchors[i][1]:anchore[i][1]])
                else:
                    for k in range(anchore[i][0]-anchors[i][0]):
                        if seq1[anchors[i][0]+k] == seq2[anchors[i][1]+k]:
                            stemp += 1
                        else:
                            stemp += 0

                rT2o += stemp
                jo += anchore[i][0]-anchors[i][0]

                agent.set(seq1[anchore[i][0]:anchors[i+1][0]+1],seq2[anchore[i][1]:anchors[i+1][1]+1])
                if FLAGS.show_align and FLAGS.print_align:
                    index = np.size(record.xtemp)
                    rT1, rT2, processingtime, j, dot_plot1 = agent.play(sess, record)
                    dot_plot[anchors[i][0]:anchore[i][0],anchors[i][1]:anchore[i][1]] = dot_plot1
                    record.shift(index,anchore[i][0],anchore[i][1])
                elif FLAGS.show_align:
                    rT1, rT2, processingtime, j, dot_plot1 = agent.play(sess)
                    dot_plot[anchors[i][0]:anchore[i][0],anchors[i][1]:anchore[i][1]] = dot_plot1
                elif FLAGS.print_align:
                    index = np.size(record.xtemp)
                    rT1, rT2, processingtime, j = agent.play(sess, record)
                    record.shift(index,anchore[i][0],anchore[i][1])
                else:
                    rT1, rT2, processingtime, j = agent.play(sess)

                rT2o += rT2
                processingtimeo += processingtime
                jo += j

            stemp = 0
            if FLAGS.show_align:
                for k in range(anchore[-1][0]-anchors[-1][0]):
                    dot_plot[anchors[-1][0]+k,anchors[-1][1]+k]=0
            if FLAGS.print_align:
                stemp = record.record([anchors[-1][0],anchore[-1][0]],[anchors[-1][1],anchore[-1][1]],-1,seq1[anchors[-1][0]:anchore[-1][0]],seq2[anchors[-1][1]:anchore[-1][1]])
            else:
                for k in range(anchore[-1][0]-anchors[-1][0]):
                    if seq1[anchors[-1][0]+k] == seq2[anchors[-1][1]+k]:
                        stemp += 1
                    else:
                        stemp += 0

            rT2o += stemp

            if (anchore[-1][0] < len(seq1)) and (anchore[-1][1] < len(seq2)):
                agent.set(seq1[anchore[-1][0]:]+"A",seq2[anchore[-1][1]:]+"A")
                if FLAGS.show_align and FLAGS.print_align:
                    index = np.size(record.xtemp)
                    rT1, rT2, processingtime, j, dot_plot1 = agent.play(sess, record)
                    dot_plot[anchore[-1][0]:,anchore[-1][1]:] = dot_plot1
                    record.shift(index,anchore[-1][0],anchore[-1][1])
                elif FLAGS.show_align:
                    rT1, rT2, processingtime, j, dot_plot1 = agent.play(sess)
                    dot_plot[anchore[-1][0]:,anchore[-1][1]:] = dot_plot1
                elif FLAGS.print_align:
                    index = np.size(record.xtemp)
                    rT1, rT2, processingtime, j = agent.play(sess, record)
                    record.shift(index,anchore[-1][0],anchore[-1][1])
                else:
                    rT1, rT2, processingtime, j = agent.play(sess)

                rT2o += rT2
                processingtimeo += processingtime
                jo += j

            now = time.time()
            #NWresult = np.max(NW.match(alignment.HEVseq[_],alignment.HEVseq[__]))
            print("result", str(rT2o), "time", str(processingtimeo)+"s", str(now-past)+"s", str(now-start)+"s")
    
            filename = "result/"+FLAGS.model_name+"+Clustal/result%04d%02d%02d%02d%02d%02d_%d_%d.txt" % (
                startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                startdate.tm_sec, train_env.win_size, train_env.maxI)

            file = open(filename,"a")
            file.write(str(rT2o)+" "+str(processingtimeo)+" "+str(now-past)+" "+str(now-start)+"\n")
            file.close()

            if FLAGS.show_align:
                filename = "img/"+FLAGS.model_name+"+Clustal/result%04d%02d%02d%02d%02d%02d_%d_%d" % (
                    startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                    startdate.tm_sec, train_env.win_size, train_env.maxI)
                cv2.imwrite(filename+"_"+str(_)+"_"+str(__)+".jpg",dot_plot)

            if FLAGS.print_align:
                filename = "align/"+FLAGS.model_name+"+Clustal/%d_%d/result%04d%02d%02d%02d%02d%02d_%d_%d.txt" % (train_env.win_size, train_env.maxI,
                    startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                    startdate.tm_sec, _, __)
                file = open(filename,"w")
                
                file.write("DQNalign Project v1.0\n")
                file.write("Pairwise alignment algorithm with deep reinforcement learning based heuristic alignment agent and preprocessing procedure of Clustal Omega\n")
                file.write("Sequence 1 : HEV_"+str(_)+", length : "+str(len(seq1))+"\n")
                file.write("Sequence 2 : HEV_"+str(__)+", length : "+str(len(seq2))+"\n")
                file.write("\n")

                record.print(file)

                file.close()

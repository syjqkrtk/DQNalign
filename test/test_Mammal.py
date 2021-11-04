from DQNalign.tool.RL.agent import Agent
from DQNalign.tool.RL.alignment import Pairwise
from DQNalign.tool.RL.Learning import *
import DQNalign.tool.Bio.lcs as lcs
import DQNalign.tool.util.ReadSeq as readseq
import DQNalign.tool.util.RecordAlign as recordalign
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

    for pairs in [["Homo_sapiens_BRCA","Mus_musculus_BRCA"],["Mus_musculus_BRCA","Rattus_norvegicus_BRCA"],["Rattus_norvegicus_BRCA","Homo_sapiens_BRCA"],["Homo_sapiens_ELK1","Mus_musculus_ELK1"],["Mus_musculus_ELK1","Rattus_norvegicus_ELK1"],["Rattus_norvegicus_ELK1","Homo_sapiens_ELK1"],["Homo_sapiens_CCDC91","Mus_musculus_CCDC91"],["Mus_musculus_CCDC91","Rattus_norvegicus_CCDC91"],["Rattus_norvegicus_CCDC91","Homo_sapiens_CCDC91"],["Homo_sapiens_FOXP2","Mus_musculus_FOXP2"],["Mus_musculus_FOXP2","Rattus_norvegicus_FOXP2"],["Rattus_norvegicus_FOXP2","Homo_sapiens_FOXP2"]]:
        seq1 = readseq.readseq('lib/Mammal/'+pairs[0]+'.txt')
        seq2 = readseq.readseq('lib/Mammal/'+pairs[1]+'.txt')
        #print(_)
        past = time.time()

        agent.set(seq1+"A", seq2+"A")
        rT1, rT2, processingtime, j = agent.Global(sess)

        now = time.time()
        #NWresult = np.max(NW.match(alignment.HEVseq[_],alignment.HEVseq[__]))
        print(pairs, len(seq1), len(seq2), rT2, "time", str(processingtime)+"s", str(now-start)+"s")
    
        filename = "result/"+FLAGS.model_name+"/result%04d%02d%02d%02d%02d%02d_%d_%d.txt" % (
            startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
            startdate.tm_sec, train_env.win_size, train_env.maxI)

        file = open(filename,"a")
        file.write(str(rT2)+" "+str(processingtime)+" "+str(now-start)+"\n")
        file.close()

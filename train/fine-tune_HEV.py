from DQNalign.tool.RL.agent import Agent
from DQNalign.tool.RL.alignment import Pairwise
from DQNalign.tool.RL.Learning import *
import DQNalign.tool.util.ReadSeq as readseq
import DQNalign.tool.Bio.lcs as lcs
from importlib import *
import copy
import time
import os
import tensorflow as tf
import DQNalign.flags as flags
FLAGS = tf.app.flags.FLAGS
param = import_module('DQNalign.param.MAML')

class game_env():
    def __init__(self, p=[0.1,0.02],maxI=10):
        self.l_seq = [2000, 2000]
        self.win_size = 100
        self.maxI = 10 # maximum indel length
        self.p = p # The probability of SNP, indel
        self.reward = [0.5,-0.5,-1] # Alignment score of the match, mismatch, indel

        self.path = "./network/MAML/"+FLAGS.model_name+"/"+str(self.win_size)+'_'+str(param.n_step)


class model():
    def __init__(self):
        self.env = Pairwise(game_env(),-1,Z=param.Z)

        tf.reset_default_graph()
        
        """ Define Deep reinforcement learning network """
        self.mainQN = SSDnetwork(param.h_size,self.env,"main",param.beta,param.n_step)
        self.targetQN = SSDnetwork(param.h_size,self.env,"target",param.alpha,param.n_step)
        self.trainables = tf.trainable_variables()
        self.copyOps = copyGraphOp(self.trainables)
        self.copyOps2 = copyGraphOp2(self.trainables)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

train_model = model()
init = train_model.init
saver = train_model.saver

seq = readseq.readseqs('lib/HEV.txt')
range_seq = list(range(0,47))

if not os.path.exists(game_env().path):
    os.makedirs(game_env().path)

if np.size(os.listdir(game_env().path)) > 0:
    resume = FLAGS.resume
else:
    resume = False

seq = readseq.readseqs('lib/HEV.txt')

""" Main training step """
with tf.Session() as sess:
    if FLAGS.use_GPU:
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    else:
        sess = tf.Session(config=tf.ConfigProto(device_count={'CPU': 0}))

    sess.run(init)

    if resume:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(game_env().path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        
    startdate = time.localtime()
        
    copyGraph(train_model.copyOps, sess)
    startdate = time.localtime()
    for i in range(param.num_finetune):
        pair = random.sample(range_seq, 2)
        seq1 = seq[pair[0]]
        seq2 = seq[pair[1]]
        start1,start2,lcslen = lcs.longestSubstring(seq1,seq2)
        agent = Agent(FLAGS, False, game_env(), train_model, seq1[start1+lcslen:]+"A", seq2[start2+lcslen:]+"A", ismeta=True)
        rT11, rT21, j1, _ = agent.metatrain(sess)
        agent = Agent(FLAGS, False, game_env(), train_model, seq1[start1 - 1::-1]+"A", seq2[start2 - 1::-1]+"A", ismeta=True)
        rT12, rT22, j2, _ = agent.metatrain(sess)
        print(i,pair[0],pair[1],j1+lcslen+j2,rT21+lcslen+rT22,rT11+lcslen+rT12)

        copyGraph(train_model.copyOps2, sess)
        saver.save(sess, game_env().path + '/tuned-' + str(i) + '.ckpt')
        print(str(i)+"th Model Saved")


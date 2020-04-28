from DQNalign.tool.RL.agent import Agent
from DQNalign.tool.RL.alignment import Pairwise
from DQNalign.tool.RL.Learning import *
import DQNalign.tool.Bio.lcs as lcs
import DQNalign.tool.util.ReadSeq as readseq
from importlib import *
import copy
import time
import os
import tensorflow as tf
from tensorflow.python.client import timeline
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
sess = tf.InteractiveSession()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

if FLAGS.use_GPU:
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
else:
    sess = tf.Session(config=tf.ConfigProto(device_count={'CPU': 0}))

sess.run(init)

param = import_module('DQNalign.param.'+FLAGS.network_set)

""" Define sequence alignment environment """
env = Pairwise(train_env,0,Z=train_model.param.Z)

mainQN = train_model.mainQN
targetQN = train_model.targetQN
trainables = train_model.trainables
targetOps = train_model.targetOps

""" Initialize the variables """
total_steps = 0
start = time.time()
myBuffer = experience_buffer()

print('Loading Model...')
ckpt = tf.train.get_checkpoint_state(train_env.path)
saver.restore(sess, ckpt.model_checkpoint_path)

s = env.reset() # Rendered image of the alignment environment
s = processState(s) # Resize to 1-dimensional vector

a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]}, options=run_options, run_metadata=run_metadata)
writer = tf.summary.FileWriter(logdir='tensorboard/graph_'+str(FLAGS.model_name)+'_'+str(train_env.win_size),graph=sess.graph)
print(a)

tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open('tensorboard/timelineOfBug.json', 'w') as f:
    f.write(ctf)

writer.add_run_metadata(run_metadata,"mySess")
writer.close()
sess.close()

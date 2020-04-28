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
import DQNalign.flags as flags
FLAGS = tf.app.flags.FLAGS


class game_env():
    def __init__(self):
        self.l_seq = [8000, 8000]
        self.win_size = 1000
        self.maxI = 10 # maximum indel length
        self.p = [0.1,0.02] # The probability of SNP, indel
        self.reward = [1,-1,-1] # Alignment score of the match, mismatch, indel

        self.path = "./network/"+FLAGS.model_name+"/"+str(self.win_size)+'_'+str(self.maxI)

train_env = game_env()

class model():
    def __init__(self):        
        self.param = import_module('DQNalign.param.'+FLAGS.network_set)
        self.env = Pairwise(train_env.reward,train_env.l_seq,train_env.win_size,train_env.p,train_env.maxI,-1)

        tf.reset_default_graph()
        
        """ Define Deep reinforcement learning network """
        if FLAGS.model_name == "DQN":
            self.mainQN = Qnetwork(self.param.h_size,self.env)
            self.targetQN = Qnetwork(self.param.h_size,self.env)
            self.trainables = tf.trainable_variables()
            self.targetOps = updateTargetGraph(self.trainables, self.param.tau)
        elif FLAGS.model_name == "SSD":
            self.mainQN = SSDnetwork(self.param.h_size,self.env,"main")
            self.targetQN = SSDnetwork(self.param.h_size,self.env,"target")
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
    g = sess.graph
    gdef = g.as_graph_def()
    tf.train.write_graph(gdef,".","graph_"+str(FLAGS.model_name)+"_"+str(train_env.win_size)+".pb",False)
    bm = tf.test.Benchmark()
    with tf.compat.v1.Session() as sess:
        logging.info('Initializing variables...')

        variables = model.weights + optimizer.weights
        for name in ('learning_rate', 'momentum'):
            a = getattr(optimizer, name, None)
            if isinstance(a, tf.Variable):
                variables.append(a)
        sess.run([v.initializer for v in variables])

        logging.info('Starting benchmarking...')
        result = bm.run_op_benchmark(sess,
                                     op,
                                     burn_iters=burn_iters,
                                     min_iters=min_iters)
        logging.info('Wall time (ms): {}'.format(result['wall_time'] *
                                                     1000))
        gpu_mem = result['extras'].get(
            'allocator_maximum_num_bytes_GPU_0_bfc', 0)
        logging.info('Memory (Mb):    {}'.format(gpu_mem / 1024**2))


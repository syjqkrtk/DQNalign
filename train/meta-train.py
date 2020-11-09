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
        self.l_seq = [10000, 10000]
        self.win_size = 1000
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

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


range_SNP = 0.001 * np.array(range(0,300), dtype=int)
range_indel = 0.0001 * np.array(range(0,300), dtype=int)
range_maxI = 5 + np.array(range(0,15), dtype=int)

train_model = model()
init = train_model.init
saver = train_model.saver

seq = readseq.readseqs('lib/HEV.txt')

if not os.path.exists(game_env().path):
    os.makedirs(game_env().path)

if np.size(os.listdir(game_env().path)) > 0:
    resume = FLAGS.resume
else:
    resume = False


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
    for i in range(param.num_episodes):
        mainBuffer = experience_buffer()
        p = [random.choice(range_SNP), random.choice(range_indel)]
        maxI = random.choice(range_maxI)
        print(i, p, maxI)
        agent = Agent(FLAGS, True, game_env(p, maxI), train_model, ismeta=True)

        for k in range(param.K):
            copyGraph(agent.copyOps, sess)
            rT1, rT2, j, mainBuffer = agent.metatrain(sess, mainBuffer)
            print(i,k,j,rT1)

        for t in range(param.meta_train_step):
            # update the main network
            trainBatch = mainBuffer.sample(param.meta_batch_size)  # Select the batch from the experience buffer
        
            # The estimated Q value from main network is Q1, from target network is Q2
            Q1 = sess.run(agent.mainQN.predict, feed_dict={agent.mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
            Q2 = sess.run(agent.mainQN.Qout, feed_dict={agent.mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
            
            # trainBatch[:,4] means that the action was the last step of the episode
            # If the action is the last step, the reward is used for update Q value
            # IF not, the Q value is updated as follows:
            # Qmain(s,a) = r(s,a) + yQtarget(s1,argmaxQmain(s1,a))
            end_multiplier = -(trainBatch[:, 4] - 1)
            doubleQ = Q2[range(param.meta_batch_size), Q1]
            targetQ = trainBatch[:, 2] + (agent.param.y * doubleQ * end_multiplier)
            _ = sess.run(agent.mainQN.updateModel, feed_dict={agent.mainQN.scalarInput: np.vstack(trainBatch[:, 0]), agent.mainQN.targetQ: targetQ, agent.mainQN.actions: trainBatch[:, 1]})

        saver.save(sess, game_env().path + '/model-' + str(i) + '.ckpt')
        print(str(i)+"th Model Saved")


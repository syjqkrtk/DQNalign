from DQNalign.alignment import gameEnv
from DQNalign.Learning import *
import time
import DQNalign.NW as NW
import DQNalign.alignment as alignment

""" Initialize the parameters of the in-silico sequence generator """
l_seq = [8000, 8000]
win_size = 100
maxI = 10 # maximum indel length
p = [0.1,0.02] # The probability of SNP, indel
reward = [1,-1,-1] # Alignment score of the match, mismatch, indel

""" Initialize the parameters of the DQN algorithm """
batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.05 #Final chance of random action
annealing_steps = 50000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 50000 #How many steps of random actions before training begins.
max_epLength = np.sum(l_seq) #The max allowed length of our episode.
load_model = True #Whether to load a saved model.
path = "./network/Pairwise/"+str(win_size)+'_'+str(maxI) #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network
n_action = 3

""" Define sequence alignment environment """
env = gameEnv(reward,l_seq,win_size,p,maxI,0)
testenv1 = gameEnv(reward,l_seq,win_size,p,maxI,1)
testenv2 = gameEnv(reward,l_seq,win_size,p,maxI,2)

""" Define Deep reinforcement learning network """
tf.reset_default_graph()
mainQN = Qnetwork(h_size,env)
targetQN = Qnetwork(h_size,env)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)
myBuffer = experience_buffer()

""" Exploration strategy """
e = startE
stepDrop = (startE - endE) / annealing_steps

""" Initialize the variables """
jList = []
rList = []
total_steps = 0

if not os.path.exists(path):
    os.makedirs(path)

startdate = time.localtime()

start = time.time()
seq1 = alignment.readseq('lib/Ecoli_1.txt')
seq2 = alignment.readseq('lib/Ecoli_2.txt')

""" Main test step """
with tf.Session() as sess:
    #sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    sess.run(init)
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(path) 
    saver.restore(sess, ckpt.model_checkpoint_path)
    past = time.time()
    start1,start2,lcslen = alignment.preprocess(seq1,seq2)

    testenv1 = gameEnv(reward, l_seq, win_size, p, maxI, -1, seq1[start1 - 1::-1], seq2[start2 - 1::-1])
    rT1 = 0
    j = 0
    rT2 = 0
    if testenv1.sizeS1 > 0 and testenv1.sizeS2 > 0 :
        while j < testenv1.sizeS1+testenv1.sizeS2:
            if testenv1.seq1[testenv1.x]==testenv1.seq2[testenv1.y]:
                seq1end = min(testenv1.x+win_size-1,testenv1.sizeS1-1)
                seq2end = min(testenv1.y+win_size-1,testenv1.sizeS2-1)
                minend = min(seq1end-testenv1.x,seq2end-testenv1.y)+1
                diff = np.where(testenv1.seq1[testenv1.x:testenv1.x + minend] != testenv1.seq2[testenv1.y:testenv1.y + minend])
                if np.size(diff) > 0:
                    a = 2 + np.min(diff)
                    rT2 += np.min(diff)
                else:
                    a = 2 + minend
                    rT2 += minend
                j += a - 2
            else:
                j += 1
                sr1 = processState(testenv1.renderEnv())
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [sr1]})[0]
            #xx, yy = testenv1.seq1[testenv1.x], testenv1.seq2[testenv1.y]
            r, d = testenv1.teststep(a)
            rT1 += r
            #print(j, a, xx, testenv1.x, yy,testenv1.y, rT1)
            if d == True:
                break

    rT2o = rT2
    testenv2 = gameEnv(reward,l_seq,win_size,p,maxI,-1,seq1[start1+lcslen:],seq2[start2+lcslen:])
    rT1 = 0
    j = 0
    rT2 = 0
    if testenv2.sizeS1 > 0 and testenv2.sizeS2 > 0 :
        while j < testenv2.sizeS1+testenv2.sizeS2:
            if testenv2.seq1[testenv2.x]==testenv2.seq2[testenv2.y]:
                seq1end = min(testenv2.x+win_size-1,testenv2.sizeS1-1)
                seq2end = min(testenv2.y+win_size-1,testenv2.sizeS2-1)
                minend = min(seq1end-testenv2.x,seq2end-testenv2.y)+1
                diff = np.where(testenv2.seq1[testenv2.x:testenv2.x + minend] != testenv2.seq2[testenv2.y:testenv2.y + minend])
                if np.size(diff) > 0:
                    a = 2 + np.min(diff)
                    rT2 += np.min(diff)
                else:
                    a = 2 + minend
                    rT2 += minend
                j += a - 2
            else:
                j += 1
                sr1 = processState(testenv2.renderEnv())
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [sr1]})[0]
            #xx, yy = testenv1.seq1[testenv1.x], testenv1.seq2[testenv1.y]
            r, d = testenv2.teststep(a)
            rT1 += r
            #print(j, a, xx, testenv1.x, yy,testenv1.y, rT1)
            if d == True:
                break

    now = time.time()
    #NWresult = np.max(NW.match(alignment.HCVseq[_],alignment.HCVseq[__]))
    print("result", lcslen + rT2o + rT2, "rawdata", lcslen, rT2o, rT2, str(np.floor(now-past))+"s", str(np.floor(now-start))+"s")

    filename = "result/result%04d%02d%02d%02d%02d%02d_%d_%d.txt" % (
        startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
        startdate.tm_sec, win_size,maxI)
    
    file = open(filename,"a")
    file.write(str(lcslen + rT2o + rT2)+" "+str(np.floor(now-past))+" "+str(np.floor(now-start))+"\n")
    file.close()
    past = now

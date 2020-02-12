from alignment import gameEnv
from Learning import *
import time

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
endE = 0.01 #Final chance of random action
annealing_steps = 50000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 50000 #How many steps of random actions before training begins.
max_epLength = np.sum(l_seq) #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
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

""" Main test step """
with tf.Session() as sess:
    #sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    sess.run(init)
    past = time.time()
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    now2 = time.time()
    
    sr1 = processState(testenv1.test(1))
    rT1 = 0
    j = 0
    while j < testenv1.sizeS1+testenv1.sizeS2:
        j += 1
        a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [sr1]})[0]
        st1, r, d = testenv1.step(a)
        st1 = processState(st1)
        rT1 += r
        sr1 = st1
        if d == True:
            break

    sr2 = processState(testenv2.test(2))
    rT2 = 0
    j = 0
    while j < testenv2.sizeS1+testenv2.sizeS2:
        j += 1
        a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [sr2]})[0]
        st2, r, d = testenv2.step(a)
        st2 = processState(st2)
        rT2 += r
        sr2 = st2
        if d == True:
            break

    now = time.time()
    print(rT1, rT2, str(np.floor(now-now2))+"s", str(np.floor(now-past))+"s", str(np.floor(now-start))+"s")

    filename = "result\\result%04d%02d%02d%02d%02d%02d.txt" % (
        startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
        startdate.tm_sec)

    file = open(filename,"a")
    file.write(str(rT1)+" "+str(rT2)+" "+str(np.floor(now-now2))+" "+str(np.floor(now-past))+" "+str(np.floor(now-start))+"\n")
    file.close()
    past = now

    saver.save(sess, path + '/model-' + str(i) + '.ckpt')
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")

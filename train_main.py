from alignment import gameEnv
from Learning import *
import time

""" Initialize the parameters of the in-silico sequence generator """
l_seq = [1000, 1000]
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
path = "./"+str(win_size)+'_'+str(maxI) #The path to save our model to.
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

""" Main training step """
with tf.Session() as sess:
    #sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    sess.run(init)
    past = time.time()
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        
    for i in range(num_episodes):
        # Newly define experience buffer for new episode
        episodeBuffer = experience_buffer()
        
        # Environment reset for each episode
        s = env.reset() # Rendered image of the alignment environment
        s = processState(s) # Resize to 1-dimensional vector
        d = False # The state of the game (End or Not)
        rAll = 0 # Total reward
        j = 0
        #print(i)

        while j < max_epLength:  # Training step is proceeded until the maximum episode length
            j += 1
            #print(i, j)
            # Exploration step
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, n_action)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]

            # Calculate the change of the state, reward and d(one)
            s1, r, d = env.step(a)
            s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(
                np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the result into episode buffer

            # Update the DQN network
            if total_steps > pre_train_steps:
                # Refresh exploration probability (epsilon-greedy)
                if e > endE:
                    e -= stepDrop

                # For every update_freq, update the main network
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)  # Select the batch from the experience buffer

                    # The estimated Q value from main network is Q1, from target network is Q2
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})

                    # trainBatch[:,4] means that the action was the last step of the episode
                    # If the action is the last step, the reward is used for update Q value
                    # IF not, the Q value is updated as follows:
                    # Qmain(s,a) = r(s,a) + yQtarget(s1,argmaxQmain(s1,a))
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                    _, loss = sess.run([mainQN.updateModel, mainQN.loss], \
                                 feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,
                                            mainQN.actions: trainBatch[:, 1]})
                    if total_steps % 1000 == 0:
                        print("loss : "+str(loss))
                    updateTarget(targetOps, sess)  # Update target network with 'tau' ratio

            rAll += r
            s = s1

            if d == True:
                break

        print(i)
        # Add the results of the episode into the total results
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        # Periodically save the model.
        if i % 20 == 0:
            saver.save(sess, path + '/model-' + str(i) + '.ckpt')
            print("Saved Model")

        if len(rList) % 1 == 0:
            now = time.time()
            print(i+1, total_steps, rList[-1], str(float("{0:.2f}".format(now-start)))+"s")
            
    saver.save(sess, path + '/model-' + str(i) + '.ckpt')
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")

from alignment import gameEnv
import time
import NW
import alignment
from Learning import *

""" alignment 관련 파라미터 초기화 """
l_seq = [8000, 8000]
win_size = 50
maxI = 20 # indel 최대 길이
p = [0.1,0.02] # SNP, indel 확률
reward = [1,-1,-1] # match, mismatch, indel의 점수

""" 각종 파라미터 초기화 """
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
path = "./50" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network
n_action = 3

""" 설정한 환경 불러오기 """
env = gameEnv(reward,l_seq,win_size,p,maxI,0)
testenv1 = gameEnv(reward,l_seq,win_size,p,maxI,1)
""" 메인 학습 과정 """
with tf.Session() as sess:    
#    for _ in range(47):
#        for __ in range(_+1,47):
            #testenv1.test(10000+100*_+__)
    testenv1.test(20000)
    #print(np.size(testenv1.sizeS1))
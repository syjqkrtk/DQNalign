from alignment import gameEnv
from Learning import *
import time

""" alignment 관련 파라미터 초기화 """
l_seq = [1000, 1000]
win_size = 100
maxI = 10 # indel 최대 길이
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
path = "./"+str(win_size)+'_'+str(maxI) #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network
n_action = 3

""" 설정한 환경 불러오기 """
env = gameEnv(reward,l_seq,win_size,p,maxI,0)
testenv1 = gameEnv(reward,l_seq,win_size,p,maxI,1)
testenv2 = gameEnv(reward,l_seq,win_size,p,maxI,2)

""" 메인 네트워크 정의 """
tf.reset_default_graph()
mainQN = Qnetwork(h_size,env)
targetQN = Qnetwork(h_size,env)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

""" 타겟 네트워크 정의 """
targetOps = updateTargetGraph(trainables, tau)

""" 전체 버퍼 정의 """
myBuffer = experience_buffer()

""" pre_train_step(완전 랜덤)을 지나고 학습 과정 초기에 얼마나 랜덤을 고려할지 """
e = startE
stepDrop = (startE - endE) / annealing_steps

# 전체 리워드나 스텝 같은걸 저장하기 위한 변수 설정
jList = []
rList = []
total_steps = 0

""" 모델 저장을 위한 부분 """
if not os.path.exists(path):
    os.makedirs(path)

startdate = time.localtime()

start = time.time()
""" 메인 학습 과정 """
with tf.Session() as sess:
    #sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    # 정의한 모델의 초기화
    sess.run(init)
    past = time.time()
    
    # 저장해둔 모델을 불러올 경우, weight 값을 불러옴
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        
    # 메인 학습과정의 메인 학습과정
    for i in range(num_episodes):
        # 현재 에피소드에서의 버퍼 정의
        episodeBuffer = experience_buffer()
        
        # 환경을 초기화하여 불러오고, 이에 대한 변수들 설정
        s = env.reset() # 이미지화 한 환경 상태
        s = processState(s) # 1차원 벡터로 변경
        d = False # 목표를 밟아 게임이 끝났는지 안 끝났는지 확인
        rAll = 0 # 게임 끝날 때 까지의 reward 총합
        j = 0
        #print(i)

        # 메인 학습과정의 메인 학습과정의 메인 학습과정
        while j < max_epLength:  # 한 에피소드의 한계점까지 진행
            j += 1
            #print(i, j)
            # pre_train_step이거나 메인에서 아직 랜덤을 진행하고 있을때, 액션을 랜덤하게 할지 원래꺼 중에 고를지에 대한 부분
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, n_action)
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]

            # 현재 액션을 취했을 때, 돌아오는 state, reward, 그리고 끝났는지 안끝났는지의 d(one)
            s1, r, d = env.step(a)
            s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(
                np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # 현재의 결과를 episode_buffer에 저장함

            # 메인 학습과정의 메인 학습과정의 메인 학습과정의 메인 학습과정
            if total_steps > pre_train_steps:
                # pre_train_step 끝나자마자 랜덤 스탭 있을때의 확률 계산
                if e > endE:
                    e -= stepDrop

                # 계산 속도 및 안정성을 위해 network를 한번에 업데이트 하지 않는다고 함, 그래서 update_freq마다 업데이트 함
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)  # experience buffer에서 맨처음에 설정한 batch_size 개수 만큼의 상태를 잡는다

                    # 메인 학습과정의 메인 학습과정의 메인 학습과정의 메인 학습과정의 메인 학습과정 (Double DQN 업데이트 진행)
                    # 앞에서 잡은 trainBatch의 상황에서 메인에서 예측한 action을 Q1, target에서 예측한 Q 값을 Q2라고 한다.
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})

                    # trainBatch[:,4]는 게임이 종료됬는지의 d에 대한 변수들이 저장된 곳
                    # 게임이 끝났던 episode 들이면 reward 값을 그대로 학습하고 (미래의 state의 예측 Q 값이 0이기 때문), 아닐 경우 Double DQN의 메인 식에 따라 Q 값을 업데이트 한다.
                    # Qmain(s,a) = r(s,a) + yQtarget(s1,argmaxQmain(s1,a)) ... Double DQN 업데이트 식
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                    _, loss = sess.run([mainQN.updateModel, mainQN.loss], \
                                 feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,
                                            mainQN.actions: trainBatch[:, 1]})
                    if total_steps % 1000 == 0:
                        print("loss : "+str(loss))
                    updateTarget(targetOps, sess)  # 타겟 네트워크는 처음에 정의한 tau값에 따라 그 비율만큼 동기화하게 한다

            rAll += r
            s = s1

            if d == True:
                break

        print(i)
        # 전체 buffer, 리워드 리스트, 스텝 같은걸 현재 episode에 기반하여 업데이트, 저장한다.
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
#
#            if len(rList) >= 100:
#                now2 = time.time()
#                testenv1.test(1)
#                rT1 = 0
#                j = 0
#                while j < testenv1.sizeS1+testenv1.sizeS2:
#                    if testenv1.seq1[testenv1.x]==testenv1.seq2[testenv1.y]:
#                        seq1end = min(testenv1.x+win_size-1,testenv1.sizeS1-1)
#                        seq2end = min(testenv1.y+win_size-1,testenv1.sizeS2-1)
#                        minend = min(seq1end-testenv1.x,seq2end-testenv1.y)+1
#                        diff = np.where(testenv1.seq1[testenv1.x:testenv1.x + minend] != testenv1.seq2[testenv1.y:testenv1.y + minend])
#                        if np.size(diff) > 0:
#                            a = 2 + np.min(diff)
#                        else:
#                            a = 2 + minend
#                        j += a - 2
#                    else:
#                        j += 1
#                        sr1 = processState(testenv1.renderEnv())
#                        a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [sr1]})[0]
#                    #xx, yy = testenv1.seq1[testenv1.x], testenv1.seq2[testenv1.y]
#                    r, d = testenv1.teststep(a)
#                    rT1 += r
#                    #print(j, a, xx, testenv1.x, yy,testenv1.y, rT1)
#                    if d == True:
#                        break
#
#                now3 = time.time()
#                testenv2.test(2)
#                rT2 = 0
#                j = 0
#                while j < testenv2.sizeS1+testenv2.sizeS2:
#                    if testenv2.seq1[testenv2.x]==testenv2.seq2[testenv2.y]:
#                        seq1end = min(testenv2.x+win_size-1,testenv2.sizeS1-1)
#                        seq2end = min(testenv2.y+win_size-1,testenv2.sizeS2-1)
#                        minend = min(seq1end-testenv2.x,seq2end-testenv2.y)+1
#                        diff = np.where(testenv2.seq1[testenv2.x:testenv2.x+minend]!=testenv2.seq2[testenv2.y:testenv2.y+minend])
#                        if np.size(diff) > 0:
#                            a = 2 + np.min(diff)
#                        else:
#                            a = 2 + minend
#                        j += a - 2
#                    else:
#                        j += 1
#                        sr2 = processState(testenv2.renderEnv())
#                        a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [sr2]})[0]
#                    #xx, yy = testenv2.seq1[testenv2.x], testenv2.seq2[testenv2.y]
#                    r, d = testenv2.teststep(a)
#                    rT2 += r
#                    #print(j, a, xx, testenv2.x, yy,testenv2.y, rT2)
#                    if d == True:
#                        break
#
#                now = time.time()
#                print(i+1, total_steps, np.mean(rList[-10:]), rT1, rT2, str(float("{0:.2f}".format(now3-now2)))+"s", str(float("{0:.2f}".format(now-now3)))+"s", str(float("{0:.2f}".format(now2-past)))+"s", str(float("{0:.2f}".format(now-start)))+"s")
#    #            print(i+1, total_steps, np.mean(rList[-100:]), rT1, str(float("{0:.2f}".format(now-now2)))+"s", str(float("{0:.2f}".format(now2-past)))+"s", str(float("{0:.2f}".format(now-start)))+"s")
#
#                filename = "result\\result%04d%02d%02d%02d%02d%02d.txt" % (
#                    startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
#                    startdate.tm_sec)
#
#                file = open(filename,"a")
#                file.write(str(i+1)+" "+str(total_steps)+" "+str(np.mean(rList[-10:]))+" "+str(rT1)+" "+str(rT2)+" "+str(float("{0:.2f}".format(now3-now2)))+" "+str(float("{0:.2f}".format(now-now3)))+" "+str(float("{0:.2f}".format(now2-past)))+" "+str(float("{0:.2f}".format(now-start)))+"\n")
#    #            file.write(str(i+1)+" "+str(total_steps)+" "+str(np.mean(rList[-100:]))+" "+str(rT1)+" "+str(float("{0:.2f}".format(now-now2)))+" "+str(float("{0:.2f}".format(now2-past)))+" "+str(float("{0:.2f}".format(now-start)))+"\n")
#                file.close()
#                past = now

    saver.save(sess, path + '/model-' + str(i) + '.ckpt')
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")

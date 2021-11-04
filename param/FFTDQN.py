""" Initialize the parameters of the DQN algorithm """
batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
test_freq = 10 #How often to perform a test network.
save_freq = 10 #How often to perform a save network.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.05 #Final chance of random action
annealing_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 50000 #How many steps of random actions before training begins.
load_model = True #Whether to load a saved model.
h_size = 256 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network
n_action = 3
n_step = 1
Z = 1 #Pixels per a nucleotide in the rendered image

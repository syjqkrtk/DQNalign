""" Initialize the parameters of the SSD algorithm """
y = .99 #Discount factor on the target Q-values
load_model = True #Whether to load a saved model.
num_finetune = 500 #How many episodes of game environment to train network with.
num_episodes = 500 #How many episodes of game environment to train network with.
h_size = 256 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
n_action = 3
n_step = 1
Z = 2 #Pixels per a nucleotide in the rendered image
alpha = 1e-7
beta = 1e-9
K = 4
update_freq = 100 #How often to perform a training step.
batch_size = 8 #How many experiences to use for each training step.
meta_batch_size = 8 #How many experiences to use for each training step.
meta_train_step = 32

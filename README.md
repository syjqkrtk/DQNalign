# DQNalign
Dueling Double DQN based sequence alignment tool

Description of the files

alignment : Define the environment, state, reward of the sequence alignment  
Learning : Define the structure of the dueling double Deep-Q Network  
lcs : For using the longest common substrings method  
NW : For using the Needleman-Wunsch algorithm  
train_main : Training the DQN network  
test_Insilico : Test the DQNalign method for various in-silico scenarios  
test_HEV : Test the DQNalign method for 47 HEV sequences  
test_Ecoli : Test the DQNalign method for 2 E.coli sequences  

The code implementation was proceeded based on the explanation provided by the "https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df". 

And the github link of reference code is https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb

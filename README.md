# DQNalign
Dueling Double DQN based sequence alignment tool

We propose a novel pairwise alignment tool using the deep reinforcement learning. By defining the local best path selection model, we can adapt the reinforcement learning into the sequence alignment problem. We verified the DQNalign algorithm in the various cases: 1) In-silico sequences based on the model of evolution 2) HEV sequences 3) E.coli sequences.  

The code was implemented in the python 3.6 and TensorFlow 1.12.0.

<img height = "300" src = "https://user-images.githubusercontent.com/49563250/74353540-68e31a80-4dfd-11ea-90dc-dbb076f02d37.png"></img>
<img height = "300" src = "https://user-images.githubusercontent.com/49563250/74353543-6a144780-4dfd-11ea-9584-e2b21db82ab4.png"></img>
<img height = "300" src = "https://user-images.githubusercontent.com/49563250/74353547-6aacde00-4dfd-11ea-8103-de4ba0f9538d.png"></img>

## Description of the project structure

- **alignment** : Define the environment, state, reward of the sequence alignment  
- **Learning** : Define the structure of the dueling double Deep-Q Network  
- **lcs** : For using the longest common substrings method  
- **NW** : For using the Needleman-Wunsch algorithm  
- **train_main** : Training the DQN network  
- **test_Insilico** : Test the DQNalign method for various in-silico scenarios  
- **test_HEV** : Test the DQNalign method for 47 HEV sequences  
- **test_Ecoli** : Test the DQNalign method for 2 E.coli sequences  

We refer the explanation provided by the "https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df".  
And the github link of reference code is https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb

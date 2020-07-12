# DQNalign
Dueling Double DQN based sequence alignment tool

We propose a novel pairwise alignment tool using the deep reinforcement learning. By defining the local best path selection model, we can adapt the reinforcement learning into the sequence alignment problem. We verified the DQNalign algorithm in the various cases: 1) In-silico sequences based on the model of evolution 2) HEV sequences 3) E.coli sequences.  

The code was implemented in the python 3.6 and TensorFlow 1.12.0.

<img height = "300" src = "https://user-images.githubusercontent.com/49563250/74353540-68e31a80-4dfd-11ea-90dc-dbb076f02d37.png"></img>
<img height = "300" src = "https://user-images.githubusercontent.com/49563250/74353543-6a144780-4dfd-11ea-9584-e2b21db82ab4.png"></img>
<img height = "300" src = "https://user-images.githubusercontent.com/49563250/74353547-6aacde00-4dfd-11ea-8103-de4ba0f9538d.png"></img>

## Description of the project structure

- **tool** : Define the environment, state, reward of the sequence alignment
- **test** : Define the structure of the dueling double Deep-Q Network
- **train** : For using the longest common substrings method
- **param** : For using the Needleman-Wunsch algorithm
- **flags.py** : Training the DQN network

- **network** : Test the DQNalign method for 2 E.coli sequences
- **align** : Test the DQNalign method for various in-silico scenarios
- **img** : Test the DQNalign method for 47 HEV sequences
- **lib** : Test the DQNalign method for 2 E.coli sequences
- **result** : Test the DQNalign method for 2 E.coli sequences
- **tensorboard** : Test the DQNalign method for 2 E.coli sequences

Our code implementation refers the following links:

Explanation - "https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df".

Deep reinforcement learning - https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb

Code architecture - https://github.com/veronicachelu/meta-learning

Clustal - https://github.com/etetoolkit/ext_apps/tree/master/src/clustal-omega-1.2.1

MUMmer 3.23 - https://sourceforge.net/projects/mummer/

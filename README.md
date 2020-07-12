# DQNalign
DQN based sequence alignment tool

We propose a novel pairwise global alignment tool using the deep reinforcement learning. By defining the local best path selection model, we can adapt the reinforcement learning into the sequence alignment problem. We verified the DQNalign algorithm in the various cases: 1) In-silico sequences based on the model of evolution 2) HEV sequences 3) E.coli sequences. Then, we tried to compare the performance of the conventional alignment methods (the MUMmer and Cluster), and also we tried to combine these methods and our DQNalign method. Detailed explanation will be written in our paper.

The code was implemented in the python 3.5.5 and TensorFlow 1.12.0.

<img height = "300" src = "https://user-images.githubusercontent.com/49563250/74353540-68e31a80-4dfd-11ea-90dc-dbb076f02d37.png"></img>
<img height = "300" src = "https://user-images.githubusercontent.com/49563250/74353543-6a144780-4dfd-11ea-9584-e2b21db82ab4.png"></img>
<img height = "300" src = "https://user-images.githubusercontent.com/49563250/74353547-6aacde00-4dfd-11ea-8103-de4ba0f9538d.png"></img>

## Description of the project structure

- **tool** : Base algorithms are implemented in this folder
-- **Bio** : About bioinformatics methods
-- **RL** : About deep reinforcement learning methods
-- **util** : Extra functions
- **test** : Experiments are implemented in this folder
- **train** : Training procedure is implemented in this folder
- **param** : Parameters are written in this folder
- **flags.py** : Selection of parameters

Because of the file sizes, we will replace the networks into the GoogleDrive link. Please download the trained networks in the following link.
**network** : https://drive.google.com/file/d/1UDqTMKoADPFCz2hiFXbm_GEZ0aObT-S2/view?usp=sharing


Our code implementation refers the following links:

Explanation - "https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df".

Deep reinforcement learning - https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb

Code architecture - https://github.com/veronicachelu/meta-learning

Clustal - https://github.com/etetoolkit/ext_apps/tree/master/src/clustal-omega-1.2.1

MUMmer 3.23 - https://sourceforge.net/projects/mummer/

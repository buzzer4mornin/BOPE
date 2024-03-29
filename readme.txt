Note: Python 2.x code used for this project.
***********************************************************************
Posterior inference in topic models with provably guaranteed algorithms
***********************************************************************

(C) Copyright 2015, Khoat Than and Tung Doan. 
This package is for academic uses only. OTHER USAGES MUST ASK FOR PERMISSION.

------------------------------------------------------------------------
This Python package contains six algorithms for learning Latent Dirichlet 
Allocation (LDA) at large scales, including: ML-FW, Online-FW, Streaming-FW, ML-OPE, Online-OPE, Streaming-OPE. They are stochastic algorithms that can work with big text collections and text streams. Their cores are two fast inference approaches to understanding individual texts.

The two inference approaches are Frank-Wolfe (FW) and Online Maximum a Posteriori Estimation (OPE). FW has a fast convergence rate, offers a principled way to trade off sparsity of solutions against quality. OPE theoretically converges to local optimum
or stationary point with a fast rate. These two methods can be easily employed to do posterior inference for various probabilistic models. OPE also can be readily used to solve nonconvex optimization problems.

If you find the package useful, please consider to cite our related work:

Than, Khoat, and Tu Bao Ho. "Inference in topic models: sparsity and trade-off." arXiv preprint arXiv:1512.03300 (2015).
Than, Khoat, and Tung Doan. "Guaranteed inference in topic models." arXiv preprint arXiv:1512.03308 (2015).

------------------------------------------------------------------------
TABLE OF CONTENTS


A. LEARNING 

   1. SETTINGS FILE

   2. DATA FILE FORMAT

B. MEASURE

C. PRINTING TOPICS


------------------------------------------------------------------------
A. LEARNING 

To learn LDA by a method, do the following steps:

- Change the current directory to the folder which contain a learning method (e.g., ML-FW)
- Estimate a model by executing:

     python run_[name of algorithm].py  [train file] [setting file] 
[model folder] [test data folder]

[train file]                      the training data.
[setting file]                    setting file which provides parameters for learning/inference.
[model folder]                    folder for saving the learned model.
[test data folder]             	 folder containing data for computing perplexity (described in details in B).

The model folder will contain some more files. These files contain some statistics of how the model is after a mini-batch is processed. These statistics include topic mixture sparsity, perplexity, top ten words of each topic, and time for finishing the E and M steps. 

Example: python ./run_ML_FW.py ../data/nyt_50k.txt ../settings.txt ../models/ML_FW/nyt ../data

1. Settings file

See settings.txt for a sample.

2. Data format

The implementations only support reading data type in LDA. Please refer to the following site for instructions.

http://www.cs.columbia.edu/~blei/lda-c/

Under LDA, the words of each document are assumed exchangeable.  Thus, each document is succinctly represented as a sparse vector of word counts. The data is a file where each line is of the form:

     [M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]

where [M] is the number of unique terms in the document, and the [count] associated with each term is how many times that term appeared in the document.  Note that [term_1] is an integer which indexes the term; it is not a string.


------------------------------------------------------------------------

B. MEASURE

Perplexity is a popular measure to see predictiveness and generalization of a topic model.

In order to compute perplexity of the model, testing data is needed. Each document in the testing data is randomly divided into two disjoint part w_obs and w_ho with the ratio 80:20
They are stored in [test data folder] with corresponding file name is of the form:

data_test_part_1.txt
data_test_part_2.txt


------------------------------------------------------------------------

D. PRINTING TOPICS

The Python script topics.py lets you print out the top N
words from each topic in a .topic file.  Usage is:

     python topics.py [beta file] [vocab file] [n words] [result file]

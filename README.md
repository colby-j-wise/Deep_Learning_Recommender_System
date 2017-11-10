# Deep Learning Final Project Repo
#### Colby Wise | Mike Alvarino | Richard Dewey @ Columbia.edu

### Project Overview:
In this research paper we apply the methodology outlined in the arXiv working
paper: ”Joint Deep Modeling of Users and Items Using Reviews for
Recommendation” for rating prediction of movies using the Amazon Instant Video
data set and GloVe.6B 50 dimensional word embeddings. Of the data set there
are only 18,000  text reviews. The approach used in this paper models users
and items jointly using review text in two cooperative neural networks.

Before attempting to train the networks as provided in this repository, the
user must preprocess the amazon instant video dataset. The goal of this
process is for one data point to contain all of the users reviews (excluding
the review for the current movie), all of the movie's reviews (excluding that
written by the current user), and the associated rating. We have provided some
notebooks and examples in the `Preprocessing` directory that may be useful.

Because one of the primary goals of our project was to explore the
effectiveness of different sequential data modeling neural network layers, it
was natural to split the code base into three different source files
corresponding with the three different architectures we analyzed.


### Data Utilized:
1. Amazon Instant Video 5-core via Julian McAuley @ UCSD.
   Available as of 11/27/17
   URL: http://jmcauley.ucsd.edu/data/amazon/

1. Global Vectors for Word Representation (GloVe) version: 6B.50d.txt
   via J.Pennington, R. Socher, C.Manning @ Stanford
   Available as of 11/27/17
   URL: https://nlp.stanford.edu/projects/glove/

### Environment:
1. requirements.txt included for reference of packages used.

### Source Code:
1. DeepCoNN-CNN.ipynb - re-implementation of the paper
1. DeepCoNN-GRU.ipynb - joint model with GRU instead of CNN
1. DeepCoNN-LSTM.ipynb - joint model with LSTM instead of CNN
1. Custom Functions.py - utility functions implemented

|Model|Training Time|Test MSE|
|-|-|-|
|CNN|12 min 53 s                   |1.48519089265|
|CNN 100|17 min 45 s               |0.854974748883|
|CNN Dropout|12 min 1 s            |1.13791756289|
|CNN Dropout 100|19 min 12 s       |1.12168053715|
|LSTM|1 hr 54 min 30 s             |1.53920110091|
|LSTM 100|2 hr 3 min 46 s          |1.3328432198|
|LSTM Dropout|2 hr 4 min 7 s       |1.11078817165|
|LSTM Dropout 100|2 hr 0 min 20 s  |1.47418677544|
|GRU|1 hr 33 min 27 s              |1.07871009008|
|GRU 100|1 hr 49 min 52 s          |1.12462539877|
|GRU Dropout|1 hr 28 min 49 s      |1.21747808816|
|GRU Dropout 100|1 hr 43 min 7 s   |1.82918500817|


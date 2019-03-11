# imdb_sentiment
imdb_sentiment algos

I have created two algorithms for predicted sentiment from movie review's using IMDB's Sentiment dataset 
(http://ai.stanford.edu/~amaas/data/sentiment/).  In the dataset train and test data and equally seperated 50/50, 
but it resplit the data to be 80 percent train and 20 test.

In my first algorithm, I used some basic preprocessing tools to remove punctuation, numbers, and spellling errors. 
I then fit a keras Tokenizer on the training data and used it transform both train and test data into integer sequences,
which I padded to a max length of 70.  After than I created a 3-layer combined bidirectional LSTM/GRU model with first layer GloVe embeddings
and an attention layer, which fed into a neural network, with a sigmoid output layer, to given a probability for the binary 
classification of the reviews as either positive or negative.  The GloVe embedding size was 300 dimensions.  I trained this model and it reached 91% train data and 
86% validation data accuracy after 11 epochs.

In my second algorithm, I experimented with ELMo embeddings using the allennlp framework. I used the same preprocessing tools
and created a simple dataset reader to convert the training and test data into instances.
I created a more simple 2 layer bidirectional LSTM with 25% dropout fed into a sigmoid function.  As the documentation and feautures
of allennlp and pytorch are not fully developed yet, I was not able to solve the bugs I had using binary cross entropy loss
and one out dimension, so I had to settle on 2 output dimesions for each class of positive and negative labels and
use categorical cross entropy, rather than having a single probability for positive or not.
I reached 95% training and 94% validation accuracy after 4 epochs.

To retrain the models, please open one of the two ipython notebooks on a GPU device or in Google Colab at 
https://colab.research.google.com/github/williamdaburke/deep_cognito_imdb_sentiment/blob/master/deepCognitoMovieSentiment_allennlp.ipynb
or
https://colab.research.google.com/github/williamdaburke/deep_cognito_imdb_sentiment/blob/master/deepCognitoMovieSentiment_keras.ipynb
and make sure  runtime > change runtime type > harware accelerator is set to GPU.

To test the allennlp model, please run the flask_sentiment file in the command line by typing "python flask_sentiment" and then 
open a browser window to: http://127.0.0.1:5000/,  but make sure you have the model file "model_allen_imdb_twoclass.th" saved in the
directory.  If not, run the deepCognitoMovieSentiment_allennlp.ipynb notebook and save the model at the end in this folder (This will 
requiring saving and then downloading if using google colab).



import torch
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
#from allennlp.predictors import SentenceTaggerPredictor
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.token_embedders import ElmoTokenEmbedder

from LstmTwoClassifier import *
from SentimentPredictor import *
from SentenceDatasetReader import *
from PreprocessText import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
data_dir = ""

def get_predictor():
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 60 #128
    MAX_LEN = 70
    dropout = 0.25
    lstm_layers = 2
    #  pre-trained model
    options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                    '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
    weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                   '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
    
    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    vocab = Vocabulary.from_files(data_dir+"vocabulary_allennlp_imdb_twoclass")
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

    elmo_embedding_dim = 256
    lstm = PytorchSeq2VecWrapper(
        torch.nn.LSTM(elmo_embedding_dim, HIDDEN_DIM, bidirectional=True,num_layers=lstm_layers, 
                                dropout = dropout, batch_first=True))
    model = LstmTwoClassifier(word_embeddings, lstm, vocab)
    net = torch.load("model_allen_imdb_twoclass.th", map_location=str(device))
    model.load_state_dict(net)
    elmo_token_indexer = ELMoTokenCharactersIndexer()
    readerSentence = SentenceDatasetReader(
        token_indexers={'tokens': elmo_token_indexer})

    return SentimentPredictor(model, dataset_reader=readerSentence) 

 
if __name__ == '__main__':
    try:
        sentence = str(sys.argv[1])
    except:
        print('Please pass ')
    try:
        prowler_game_func(filename)
    except Exception as e:
        print("File: ",filename,': \n',e,'\n')

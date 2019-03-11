from typing import Iterator, List, Dict
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.data.vocabulary import Vocabulary
import torch
import torch.optim as optim


class LstmTwoClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        # We need the embeddings to convert word IDs to their vector representations
        self.word_embeddings = word_embeddings


        self.encoder = encoder


        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=2)
        self.accuracy = CategoricalAccuracy()
        #self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
        self.out_act = torch.nn.Sigmoid()
        

        #self.loss_function = torch.nn.BCELoss()   #BCEWithLogitsLoss()   #CrossEntropyLoss()
        self.loss = torch.nn.CrossEntropyLoss()  #torch.nn.BCEWithLogitsLoss()
        

    def forward(self,
                tokens: Dict[str, torch.tensor],
                label: torch.tensor = None) -> torch.tensor:

        mask = get_text_field_mask(tokens)

        # Forward pass
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        linear_out = self.hidden2tag(encoder_out)
        
        logits = self.out_act(linear_out)
        
        output = {"logits": logits}
        if label is not None:
            #y = torch.tensor(label.reshape(-1, 1), dtype=torch.float).cuda()
            self.accuracy(logits, label)
            #output["loss"] = self.loss_function(logits, label)
            output["loss"] = self.loss(logits, label)
            
            
        return output
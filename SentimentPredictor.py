from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.predictors import Predictor
from overrides import overrides
from typing import List
from allennlp.models import Model

#@Predictor.register("sentence_classifier_predictor")
class SentimentPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        #tokens = self._tokenizer.split_words(sentence)
        #return self._dataset_reader.text_to_instance([str(t) for t in tokens])
        return self._dataset_reader.text_to_instance(sentence)
from typing import Iterator, List, Dict
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.fields import LabelField, TextField, Field, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

class SentenceDatasetReader(DatasetReader):

    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer or WordTokenizer(JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self,  # type: ignore
                         text: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokenized_text = self._tokenizer.tokenize(text)
        fields["tokens"] = TextField(tokenized_text, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, sentence: str) -> Iterator[Instance]:
        sentence = sentence.strip()
        yield self.text_to_instance(sentence)
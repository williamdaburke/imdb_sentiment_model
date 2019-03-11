from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import LabelField, TextField, Field, SequenceLabelField
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder


#@DatasetReader.register("imdb_sentiment")
class ImdbSentimentDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        logger.info("Reading instances from lines in file at: %s", file_path)
        with open(cached_path(file_path), "r") as data_file:
            csv_in = csv.reader(data_file, delimiter=',')
            next(csv_in)
            for row in csv_in:
                if len(row) == 4:
                    review_text = preprocess_text(row[2])
                    labels_list.append(row[3])
                    yield self.text_to_instance(text=review_text, label=row[3])
    @overrides
    def text_to_instance(self,  # type: ignore
                         text: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokenized_text = self._tokenizer.tokenize(text)
        #torch.tensor(tokenized_text, dtype=torch.long).cuda()
        fields["tokens"] = TextField(tokenized_text[:MAX_LEN], self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
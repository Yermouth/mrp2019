import json
import logging
import random
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.models import Model
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

UNKNOWN_POS = 'XXX'
UNKNOWN_LABEL = 'XXXXXX'


# @DatasetReader.register("mrp_json_reader")
class MRPDatasetReader(DatasetReader):
    def __init__(
            self,
            lazy: bool = False,
            #             tokenizer: Tokenizer = None,
            word_indexers: Dict[str, TokenIndexer] = None,
            pos_indexers: Dict[str, TokenIndexer] = None,
            parse_feature_window_size: int = 5,
    ) -> None:
        super().__init__(lazy)
        #         self._tokenizer = tokenizer or WordTokenizer()
        self._word_indexers = word_indexers or {
            "word": SingleIdTokenIndexer(namespace='word'),
        }
        self._pos_indexers = pos_indexers or {
            'pos': SingleIdTokenIndexer(namespace='pos'),
        }
        self.parse_feature_window_size = parse_feature_window_size

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s",
                        file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue

                data_instance = json.loads(line)
                yield from self.process_mrp_graph(data_instance)

    def process_mrp_graph(self, data_instance):
        mrp_json = data_instance.get('mrp_json', {})
        parse_json = data_instance.get('parse_json', {})
        mrp_parser_states = data_instance.get('mrp_parser_states', {})
        mrp_meta_data = data_instance.get('mrp_meta_data', {})
        companion_parser_states = data_instance.get('companion_parser_states',
                                                    {})
        companion_meta_data = data_instance.get('companion_meta_data', {})

        # Generate features
        framework = mrp_json.get('framework', '')
        parse_nodes = parse_json.get('nodes', [])
        parse_node_labels = []
        parse_node_lemmas = []
        parse_node_uposs = []
        parse_node_xposs = []

        for node in parse_nodes:
            parse_node_labels.append(Token(node.get('label', UNKNOWN_LABEL)))
            for prop, value in zip(
                    node.get('properties', []), node.get('values', [])):
                if prop == 'lemma':
                    parse_node_lemmas.append(Token(value))
                if prop == 'upos':
                    parse_node_uposs.append(Token(value))
                if prop == 'xpos':
                    parse_node_xposs.append(Token(value))

        label = str(random.randint(0, 1))
        yield self.text_to_instance(parse_node_labels, parse_node_lemmas,
                                    parse_node_uposs, parse_node_xposs, label)

    @overrides
    def text_to_instance(self,
                         parse_node_labels,
                         parse_node_lemmas,
                         parse_node_uposs,
                         parse_node_xposs,
                         label=str) -> Instance:  # type: ignore

        # pylint: disable=arguments-differ
        tokenized_parse_label = parse_node_labels
        tokenized_parse_lemma = parse_node_lemmas
        tokenized_parse_upos = parse_node_uposs
        tokenized_parse_xpos = parse_node_xposs

        parse_label_field = TextField(tokenized_parse_label,
                                      self._word_indexers)
        parse_lemma_field = TextField(tokenized_parse_lemma,
                                      self._word_indexers)
        parse_upos_field = TextField(tokenized_parse_upos, self._pos_indexers)
        parse_xpos_field = TextField(tokenized_parse_xpos, self._pos_indexers)

        instance_fields = {
            'parse_label': parse_label_field,
            'parse_lemma': parse_lemma_field,
            'parse_upos': parse_upos_field,
            'parse_xpos': parse_xpos_field,
        }
        if label:
            instance_fields['label'] = LabelField(label)
        return Instance(instance_fields)

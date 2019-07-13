from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from mrp_library import MRPDatasetReader

TEXT_FIELD_NAMES = ['parse_label', 'parse_lemma', 'parse_upos', 'parse_xpos']


class TestSemanticScholarDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = MRPDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/test.jsonl'))

        instance0 = {
            'input':
            'In the final minute of the game, Johnson had the ball stolen by Celtics center Robert Parish, and then missed two free throws that could have won the game.',
            'parse_label': [
                'In', 'the', 'final', 'minute', 'of', 'the', 'game', ',',
                'Johnson', 'had', 'the', 'ball', 'stolen', 'by', 'Celtics',
                'center', 'Robert', 'Parish', ',', 'and', 'then', 'missed',
                'two', 'free', 'throws', 'that', 'could', 'have', 'won', 'the',
                'game', '.'
            ],
            'parse_lemma': [
                'in', 'the', 'final', 'minute', 'of', 'the', 'game', ',',
                'Johnson', 'have', 'the', 'ball', 'steal', 'by', 'Celtics',
                'center', 'Robert', 'Parish', ',', 'and', 'then', 'miss',
                'two', 'free', 'throw', 'that', 'could', 'have', 'win', 'the',
                'game', '.'
            ],
            'parse_upos': [
                'ADP', 'DET', 'ADJ', 'NOUN', 'ADP', 'DET', 'NOUN', 'PUNCT',
                'PROPN', 'VERB', 'DET', 'NOUN', 'VERB', 'ADP', 'PROPN', 'NOUN',
                'PROPN', 'PROPN', 'PUNCT', 'CONJ', 'ADV', 'VERB', 'NUM', 'ADJ',
                'NOUN', 'PRON', 'AUX', 'AUX', 'VERB', 'DET', 'NOUN', 'PUNCT'
            ],
            'parse_xpos': [
                'IN', 'DT', 'JJ', 'NN', 'IN', 'DT', 'NN', ',', 'NNP', 'VBD',
                'DT', 'NN', 'VBN', 'IN', 'NNPS', 'NN', 'NNP', 'NNP', ',', 'CC',
                'RB', 'VBD', 'CD', 'JJ', 'NNS', 'WDT', 'MD', 'VB', 'VBN', 'DT',
                'NN', '.'
            ],
        }

        assert len(instances) == 1
        fields = instances[0].fields
        assert all([t.text for t in fields[text_field_name].tokens] ==
                   instance0[text_field_name]
                   for text_field_name in TEXT_FIELD_NAMES)

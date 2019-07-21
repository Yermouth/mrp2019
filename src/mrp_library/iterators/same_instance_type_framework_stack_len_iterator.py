import logging
import random
from collections import defaultdict, deque
from typing import Deque, Iterable

from allennlp.common.util import lazy_groups_of
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.iterators.data_iterator import DataIterator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

#logger.setLevel(logging.DEBUG)


def split_by_instance_type_by_framework_by_stack_len(instance_list):
    instance_type2framework2stack_len2insts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))

    for inst in instance_list:
        # logger.debug(('framework', inst.fields['framework'].metadata))
        instance_type = inst.fields['instance_type'].metadata
        framework = inst.fields['framework'].metadata
        stack_len = inst.fields['stack_len'].metadata
        instance_type2framework2stack_len2insts[instance_type][framework][
            stack_len].append(inst)

    instance_list = [
        insts for framework2stack_len2insts in
        instance_type2framework2stack_len2insts.values()
        for stack_len2insts in framework2stack_len2insts.values()
        for insts in stack_len2insts.values() if insts
    ]
    random.shuffle(instance_list)
    return iter(instance_list)


#@DataIterator.register("same_framework")
class SameInstanceTypeFrameworkStackLenIterator(BucketIterator):
    """
    Splits batches into batches containing the same framework.
    The framework of each instance is determined by looking at the 'framework' value
    in the metadata.
    It takes the same parameters as :class:`allennlp.data.iterators.BucketIterator`
    """

    def __init__(self, shuffle, *args, **kwargs):
        super(SameInstanceTypeFrameworkStackLenIterator, self).__init__(
            *args, **kwargs)
        self.shuffle = shuffle

    def _create_batches(self, instances: Iterable[Instance],
                        shuffle) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            if shuffle:
                random.shuffle(instance_list)

            instance_list = split_by_instance_type_by_framework_by_stack_len(
                instance_list)
            for same_repr_batch in instance_list:
                iterator = iter(same_repr_batch)
                excess = deque()
                # Then break each memory-sized list into batches.
                for batch_instances in lazy_groups_of(iterator,
                                                      self._batch_size):
                    for poss_smaller_batches in self._ensure_batch_is_sufficiently_small(  # type: ignore
                            batch_instances, excess):
                        batch = Batch(poss_smaller_batches)
                        yield batch
                if excess:
                    yield Batch(excess)

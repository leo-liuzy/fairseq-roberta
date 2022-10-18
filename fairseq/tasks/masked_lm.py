# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import json
from typing import Optional
from dataclasses import dataclass, field
from fairseq.pdb import distributed_set_trace

import numpy as np
from omegaconf import II, MISSING, OmegaConf

from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    MaskStrategyTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.dataclass import ChoiceEnum, FairseqDataclass

from pdb import set_trace as bp

from .language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES
# MASK_IMPL_CHOICE = ChoiceEnum(["fix", "bernoulli"])

logger = logging.getLogger(__name__)


@dataclass
class MaskedLMConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_prob_range: str = field(
        default="none",
        metadata={"help": "probability range of replacing a token with mask"},
    )
    mask_impl: str = field(
        default="fix",
        metadata={"help": "If set to 'fix', calculate the expected number of mask for the current sequence/sentence"
                  "If set to 'bernoulli', independently determine whether to mask a token by flipping a coin of success rate=`mask_prob`"},
    )
    seqlen_masking_flag: bool = field(
        default=False,
        metadata={"help": "True to adopt sequence length-based uniform masking"},
    )
    seqlen_masking_granularity: str = field(
        default="none",
        metadata={"help": 'If set to "eos", sample once num_mask for each sentence; a sequence may contains many sentences.'
        'If omitted or "none", Follow MLM\'s sample once for entire sequence.'
        'If set to "eos-eos", in addition to "eos", when masking, avoid masking eos'
        },
    )
    seqlen_masking_boundary: Optional[str]= field(
        # default="0.1-0.9",
        default="none",
        metadata={"help": "seq-len sample num_mask from [1,seq_len); this might have large variance,"
                  "this argument is to reset the boundary to be e.g. [0.1*seq_len, 0.9*seq_len)"
        },
    )
    use_learned_unmask_table : Optional[bool]= field(
        default=False,
        metadata={"help": "True to use learned unmask mappings"},
    )
    path_to_learned_unmask_table : Optional[str]= field(
        default=None,
        metadata={"help": "Path to learned unmask mappings. The table is structured as:"
                  "sequence_length -> [[1,...,sequence_length], [prob(1),...., prob(sequence_length)]]"
                  "Given a sequence_length, we assume all tokens are masked, "
                  "and we sample how many tokens to unmask according to the probability in the second list"},
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a token with a random token"},
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on word frequencies"},
    )
    mask_whole_words: bool = field(
        default=False,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"},
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    seed: int = II("common.seed")

    include_target_tokens: bool = field(
        default=False,
        metadata={
            "help": "include target tokens in model input. this is used for data2vec"
        },
    )


@register_task("masked_lm", dataclass=MaskedLMConfig)
class MaskedLMTask(FairseqTask):

    cfg: MaskedLMConfig

    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, cfg: MaskedLMConfig, dictionary):
        super().__init__(cfg)
        self.dictionary = dictionary

        # add mask token
        self.mask_idx = dictionary.add_symbol("<mask>")

    @classmethod
    def setup_task(cls, cfg: MaskedLMConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(cfg, dictionary)

    def _load_dataset_split(self, split, epoch, combine):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
        )
        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.cfg.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.cfg.sample_break_mode,
        )
        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.cfg.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.cfg.sample_break_mode,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))
        return PrependTokenDataset(dataset, self.source_dictionary.bos())

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        dataset = self._load_dataset_split(split, epoch, combine)

        # create masked input and targets
        mask_whole_words = (
            get_whole_word_mask(self.args, self.source_dictionary)
            if self.cfg.mask_whole_words
            else None
        )
        learned_seqlen_unmask_table = json.load(open(self.cfg.path_to_learned_unmask_table, 'r')) if self.cfg.use_learned_unmask_table else None
        # bp()
        src_dataset, tgt_dataset = MaskStrategyTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.cfg.seed,
            mask_prob=self.cfg.mask_prob,
            mask_prob_range=self.cfg.mask_prob_range,
            # seq-len arguments
            seqlen_masking_flag=self.cfg.seqlen_masking_flag,
            seqlen_masking_granularity=self.cfg.seqlen_masking_granularity,
            seqlen_masking_boundary=self.cfg.seqlen_masking_boundary,
            # arguement to use learned unmask
            learned_unmask_table=learned_seqlen_unmask_table,
            max_seq_length=self.cfg.tokens_per_sample,
            bos=self.source_dictionary.bos(),
            eos=self.source_dictionary.eos(),
            leave_unmasked_prob=self.cfg.leave_unmasked_prob,
            random_token_prob=self.cfg.random_token_prob,
            freq_weighted_replacement=self.cfg.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
            mask_multiple_length=self.cfg.mask_multiple_length,
            mask_stdev=self.cfg.mask_stdev,
        )

        # if self.cfg.seqlen_masking_granularity != "none":
        #     # if eos, src and tgt have 1 sentence per samepl, 
        #     # we need to make sure src and tgt follow self.cfg.sample_break_mode
        #     src_dataset, tgt_dataset = MaskStrategyTokensDataset.apply_mask(
        #         dataset,
        #         self.source_dictionary,
        #         pad_idx=self.source_dictionary.pad(),
        #         mask_idx=self.mask_idx,
        #         seed=self.cfg.seed,
        #         mask_prob=self.cfg.mask_prob,
        #         seqlen_masking_flag=self.cfg.seqlen_masking_flag,
        #         seqlen_masking_granularity=self.cfg.seqlen_masking_granularity,
        #         bos=self.source_dictionary.bos(),
        #         eos=self.source_dictionary.eos(),
        #         leave_unmasked_prob=self.cfg.leave_unmasked_prob,
        #         random_token_prob=self.cfg.random_token_prob,
        #         freq_weighted_replacement=self.cfg.freq_weighted_replacement,
        #         mask_whole_words=mask_whole_words,
        #         mask_multiple_length=self.cfg.mask_multiple_length,
        #         mask_stdev=self.cfg.mask_stdev,
        #     )
        # else:
        #     src_dataset, tgt_dataset = MaskStrategyTokensDataset.apply_mask(
        #         dataset,
        #         self.source_dictionary,
        #         pad_idx=self.source_dictionary.pad(),
        #         mask_idx=self.mask_idx,
        #         seed=self.cfg.seed,
        #         mask_prob=self.cfg.mask_prob,
        #         mask_impl=self.cfg.mask_impl,
        #         seqlen_masking_flag=self.cfg.seqlen_masking_flag,
        #         leave_unmasked_prob=self.cfg.leave_unmasked_prob,
        #         random_token_prob=self.cfg.random_token_prob,
        #         freq_weighted_replacement=self.cfg.freq_weighted_replacement,
        #         mask_whole_words=mask_whole_words,
        #         mask_multiple_length=self.cfg.mask_multiple_length,
        #         mask_stdev=self.cfg.mask_stdev,
        #     )
        # src_dataset[0]
        # tgt_dataset[0]
        # distributed_set_trace()
        # print()

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_dataset))

        target_dataset = RightPadDataset(
            tgt_dataset,
            pad_idx=self.source_dictionary.pad(),
        )

        input_dict = {
            "src_tokens": RightPadDataset(
                src_dataset,
                pad_idx=self.source_dictionary.pad(),
            ),
            "src_lengths": NumelDataset(src_dataset, reduce=False),
        }
        if self.cfg.include_target_tokens:
            input_dict["target_tokens"] = target_dataset

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": input_dict,
                    "target": target_dataset,
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = RightPadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.cfg.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

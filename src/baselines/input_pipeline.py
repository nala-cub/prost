# Copyright 2021 The PROST Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

import copy
import multiprocessing as mp
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import datasets
import numpy as np
import toolz.curried as T
import torch
from absl import logging
from icecream import ic
from labtools import hf_one_to_many
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _get_option_encodings(example, tokenizer, strict=False):
  option_words = T.get(list('ABCD'), example)
  fmt_str = ' %s'

  option_encodings = [
    tokenizer(fmt_str % tok, add_special_tokens=False) for tok in option_words]
  if strict:
    _check_encodings(option_encodings, name=example['name'])
  return option_encodings


def _check_encodings(encodings, level=logging.FATAL, **kwargs):
  # check option_ids
  if not all([len(o.input_ids) == 1 for o in encodings]):
    # create table
    err_mapping = {
      str(o.tokens()): o.input_ids for o in encodings if len(o.input_ids) > 1}
    info_str = '&'.join(['='.join([k, v]) for k, v in kwargs.items()])
    logging.log(
      level,
      '[%s] Found invalid options, expected each to have exatly one token. ' +
      'Encodings: %s',
      info_str,
      err_mapping,
    )


def try_get_index(tokens, stoken, s=0, e=-1):
  try:
    target_tok_idx = tokens.index(stoken, s, e)
    return target_tok_idx
  except ValueError:
    logging.fatal('Could not locate token (%s). Tokens: \n%s', stoken, tokens)


@T.curry
@hf_one_to_many
def create_examples_gpt(example, tokenizer) -> List[Dict[str, Any]]:
  # This is batchd so that we can expand each each example in the base dataset
  # to 4 examples of the form: <context><canidate_answer>
  # lint of instances to create.
  instances = []

  context = example['context']

  # substitute mask with <pad>
  example['question'] = re.sub(r'( \[MASK\])|(\[MASK\])', tokenizer.pad_token,
                               example['question'])
  # format text input
  text_input = '{context} {question}'.format_map(example)

  if tokenizer.name_or_path.startswith('gpt'):
    text_input = f'{tokenizer.bos_token}{text_input}{tokenizer.eos_token}'

  # get BatchEncoding
  base_instance = tokenizer(text=text_input, padding='do_not_pad',
                            add_special_tokens=True)
  # locate pad token
  target_tok_idx = try_get_index(base_instance.tokens(), tokenizer.pad_token, 1,
                                 -1)

  # add example-general info
  base_instance['example_idx'] = example['example_idx']
  base_instance['target_tok_idx'] = target_tok_idx

  # get option ids
  option_encodings = _get_option_encodings(example, tokenizer, True)
  option_input_ids = [o.input_ids[0] for o in option_encodings]

  # substitute <pad> with each option
  for i, option_input_id in enumerate(option_input_ids):
    instance = copy.deepcopy(base_instance)
    # index of this option (A -> 0, B-> 1, etc.)
    instance['option_idx'] = i
    # update ids
    instance['input_ids'][target_tok_idx] = option_input_id
    instance['option_input_ids'] = option_input_id
    instances.append(instance)

  return instances


@T.curry
def create_examples_t5(examples, tokenizer):

  example = T.valmap(T.get(0), examples)

  # get option encodings and number of tokens per option.
  option_encodings = _get_option_encodings(example, tokenizer, False)
  option_input_ids = [o.input_ids[0] for o in option_encodings]

  # T5 uses <extra_id_*> for each token to fill
  # since we test that each option is only 1 token, we set mask in the input
  # v sequence to <extra_id_0>
  extra_id_toks = tokenizer.additional_special_tokens
  example['question'] = re.sub(r'( \[MASK\])|(\[MASK\])', extra_id_toks[0],
                               example['question'])

  # format text input
  text_input = '{context} {question}'.format_map(example)

  # get BatchEncoding
  instance = tokenizer(text=text_input, padding='do_not_pad',
                       add_special_tokens=True)
  # locate pad token
  target_tok_idx = try_get_index(instance.tokens(), extra_id_toks[0])

  # now we need to create labels. This will be a sequence of extra_id_toks,
  # except at `target_tok_idx`, however, there must be enough extra ids
  if len(instance.tokens()) > len(extra_id_toks):
    logging.fatal(
      'Cannot create labels, only have %d extra ids available, but need %d',
      len(instance.tokens()), len(extra_id_toks))
  # toks -> labels
  extra_id_ids = tokenizer.convert_tokens_to_ids(extra_id_toks)

  instance['labels'] = [
    extra_id_ids[0], option_input_ids[example['label']], extra_id_ids[1]]

  # add example-general info
  instance['example_idx'] = example['example_idx']
  instance['target_tok_idx'] = target_tok_idx
  instance['option_input_ids'] = option_input_ids
  instance['option_idx'] = list(range(4))
  # indicator to drop
  instance['dropme'] = 'sliding' in example['name']
  return {k: [instance[k]] for k in instance.keys()}


@T.curry
def create_examples_bert(examples, tokenizer):
  example = T.valmap(T.get(0), examples)
  # substitute mask with <mask>

  # get option ids
  option_encodings = _get_option_encodings(example, tokenizer, True)
  option_input_ids = [o.input_ids[0] for o in option_encodings]

  # BERT -> mask token doesn't have a space
  example['question'] = re.sub(r'(\[MASK\])', tokenizer.mask_token,
                               example['question'])

  # format text input
  text_input = '{context} {question}'.format_map(example)

  # get BatchEncoding
  instance = tokenizer(text=text_input, padding='do_not_pad',
                       add_special_tokens=True)
  # locate pad token
  target_tok_idx = try_get_index(instance.tokens(), tokenizer.mask_token)

  # add example-general info
  instance['example_idx'] = example['example_idx']
  instance['option_idx'] = list(range(4))
  instance['target_tok_idx'] = target_tok_idx
  instance['option_input_ids'] = option_input_ids

  return {k: [instance[k]] for k in instance.keys()}


@T.curry
def create_examples_roberta(examples, tokenizer):
  example = T.valmap(T.get(0), examples)
  # substitute mask with <mask>

  # get option encodings and number of tokens per option.
  option_encodings = _get_option_encodings(example, tokenizer, True)
  option_input_ids = [o.input_ids[0] for o in option_encodings]

  # RoBERTa expoects <mask> to be contain the space, e.g. `<mask>=' hi'`.
  example['question'] = re.sub(r'( \[MASK\])|(\[MASK\])', tokenizer.mask_token,
                               example['question'])

  # format text input
  text_input = '{context} {question}'.format_map(example)

  # get BatchEncoding
  instance = tokenizer(text=text_input, padding='do_not_pad',
                       add_special_tokens=True)
  # locate pad token
  target_tok_idx = try_get_index(instance.tokens(), tokenizer.mask_token)

  # add example-general info
  instance['example_idx'] = instance['example_idx']
  instance['option_idx'] = list(range(4))
  instance['target_tok_idx'] = target_tok_idx
  instance['option_input_ids'] = option_input_ids

  return {k: [instance[k]] for k in instance.keys()}


@T.curry
def create_examples_albert(examples, tokenizer):
  """Create examples for Albert.
    Albert uses whole-word masking, so [MASK] should be replaced with the
    number of tokens that the option has. This version only accounts for SINGLE token
    masking, see create_examples_albert_wwm for details on the other approach.

    """
  example = T.valmap(T.get(0), examples)
  # substitute mask with <mask>

  # get option ids
  option_encodings = _get_option_encodings(example, tokenizer, True)
  option_input_ids = [o.input_ids[0] for o in option_encodings]

  # substitute [MASK]
  example['question'] = re.sub(r'( \[MASK\])|(\[MASK\])', tokenizer.mask_token,
                               example['question'])

  # format text input
  text_input = '{context} {question}'.format_map(example)

  # get BatchEncoding
  instance = tokenizer(text=text_input, padding='do_not_pad',
                       add_special_tokens=True)
  # locate pad token
  target_tok_idx = try_get_index(instance.tokens(), tokenizer.mask_token)

  # add example-general info
  instance['example_idx'] = example['example_idx']
  instance['option_idx'] = list(range(4))
  instance['target_tok_idx'] = target_tok_idx
  instance['option_input_ids'] = option_input_ids

  return {k: [instance[k]] for k in instance.keys()}


@T.curry
def create_examples_unifiedqa(examples, tokenizer):
  """Create examples for UnifiedQA. """
  example = T.valmap(T.get(0), examples)

  # substitute mask with <mask>

  # get option ids
  # get option encodings and number of tokens per option.
  option_encodings = _get_option_encodings(example, tokenizer, False)
  option_input_ids = [o.input_ids[0] for o in option_encodings]

  # format text input
  # text_input = '{ex_question} \\n {context} ...'.format_map(example)
  text_input = '{ex_question} \\n (A) {A} (B) {B} (C) {C} (D) {D} \\n {context}'.format_map(
    example)
  # print(text_input)

  # get BatchEncoding
  instance = tokenizer(text=text_input, padding='do_not_pad',
                       add_special_tokens=True)
  # locate pad token
  # target_tok_idx = try_get_index(instance.tokens(), tokenizer.mask_token)
  # ic(instance.tokens(),)
  # maybe not setup
  if example['ex_question'] == '' or 'sliding' in example['name']:
    instance['dropme'] = True
  else:
    instance['dropme'] = False

  # add example-general info
  instance['example_idx'] = example['example_idx']
  instance['option_idx'] = list(range(4))
  instance['target_tok_idx'] = example['label']
  instance['option_input_ids'] = option_input_ids

  return {k: [instance[k]] for k in instance.keys()}


@dataclass
class T5DataCollator(DataCollatorWithPadding):
  """See transformers.DataCollatorWithPadding for args."""
  tokenizer: PreTrainedTokenizerBase
  padding: str = 'longest'

  def __call__(
    self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
  ) -> Dict[str, torch.Tensor]:
    # pop labels
    labels = [f.pop('labels') for f in features]
    batch = super().__call__(features)
    # pad labels
    batch['labels'] = torch.nn.utils.rnn.pad_sequence(
      labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    return batch


def get_prost_ds(model_name: str, model_type: str,
                 tokenizer: PreTrainedTokenizerBase):
  """Creates formatted PROST dataset.
  Args:
      model_name: Name of model
      tokenizer: Tokenizer for model.
  """
  ds_path = str(Path(__file__).parent.parent / 'prost')
  ds = datasets.load_dataset(ds_path, split='test')

  __model_fn_map = {
    'albert': create_examples_albert,
    'gpt2': create_examples_gpt,
    'bert': create_examples_bert,
    'roberta': create_examples_roberta,
    'openai-gpt': create_examples_gpt,
    't5': create_examples_t5,
    'unifiedqa': create_examples_unifiedqa}

  examples_fn = __model_fn_map.get(model_type, None)

  if 'unifiedqa' in model_name:
    examples_fn = create_examples_unifiedqa

  if examples_fn is None:
    logging.fatal('This script is not set up to use %s', model_type)

  logging.info('Using preprocessing fn: %s', examples_fn.__name__)

  # create examples
  ds = ds.map(
    examples_fn(tokenizer=tokenizer), batch_size=1, batched=True,
    remove_columns=ds.column_names)
  if 'dropme' in ds.column_names:
    num_examples = len(ds)
    ds = ds.filter(lambda example: not example['dropme'])
    ds = ds.map(lambda x: x, remove_columns=['dropme'])
    logging.info('Filtered out %d examples for %s', num_examples - len(ds),
                 model_name)

  ds.set_format(type='torch', columns=ds.column_names)
  return ds


def get_prost_iter(model_name: str, model_type: str,
                   tokenizer: PreTrainedTokenizerBase, batch_size: int):
  """Creates formatted PROST dataset.
  Args:
      model_name: Name of model
      tokenizer: Tokenizer for model.
  """
  # block parallelism
  pre_tp = os.environ.get('TOKENIZERS_PARALLELISM', 'true')
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  ds = get_prost_ds(model_name, model_type, tokenizer)
  os.environ['TOKENIZERS_PARALLELISM'] = pre_tp

  # as iter
  if re.match(r'^t5.+', model_name):
    collate_fn = T5DataCollator(tokenizer=tokenizer, padding='longest')
  else:
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

  batch_iter = DataLoader(ds, batch_size=batch_size,
                          num_workers=min(batch_size, mp.cpu_count()),
                          drop_last=False, collate_fn=collate_fn)
  return batch_iter

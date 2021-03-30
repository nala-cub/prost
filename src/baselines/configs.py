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

import labtools
from absl import flags
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForMaskedLM, AutoTokenizer,
                          T5ForConditionalGeneration)

MODEL_BSM = {
  # gpt
  'openai-gpt': 1,  # << set batch size s.t. this runs.
  # gpt2
  'gpt2': 0.75,
  'gpt2-medium': 0.5,
  'gpt2-large': 0.3,
  'gpt2-xl': 0.25,
  # bert
  'bert-base-uncased': 0.75,
  'bert-large-uncased': 0.5,
  # roberta
  'roberta-base': 0.75,
  'roberta-large': 0.5,
  # albert v2
  'albert-base-v2': 0.3,
  'albert-xlarge-v2': 0.3,
  'albert-large-v2': 0.3,
  'albert-xxlarge-v2': 0.25,
  # t5
  't5-small': 1,
  't5-base': 1,
  't5-large': 0.5,
  't5-3b': 0,
  # unifiedqa
  'allenai/unifiedqa-t5-small': 1,
  'allenai/unifiedqa-t5-base': 1,
  'allenai/unifiedqa-t5-large': 0.5,
  'allenai/unifiedqa-t5-3b': 0,}

FLAGS = flags.FLAGS
flags.DEFINE_integer('gpus', 1, '# gpus to use')
flags.DEFINE_integer(
  'batch_size', 512,
  'Batch size for opani-gpt. Will automatically be adjusted for large models. Default is set to fit in 24GB of VRAM.'
)
flags.DEFINE_multi_string('model_name', None,
                          'Model configuration, run all if not provided')
flags.DEFINE_boolean('use_cached_results', False, 'Use cached results.')
flags.DEFINE_string('results_dir', labtools.get_results_dir('prost'),
                    'Directory to store results/checkpoints.')


def hf_auto_configure(model_name):
  model_config = AutoConfig.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  if model_config.model_type in ('openai-gpt', 'gpt2'):
    model_config.is_causal = True
    model_cls = AutoModelForCausalLM
    tokenizer.pad_token = tokenizer.unk_token
  else:
    model_cls = AutoModelForMaskedLM
    model_config.is_causal = False

  # add tokens
  if model_config.model_type == 't5':
    model_cls = T5ForConditionalGeneration

  return model_config, tokenizer, model_cls

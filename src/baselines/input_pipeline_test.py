# Copyright 2021 The PROST Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Tests for Baseline input pipelines."""

import copy

from absl import logging
from absl.testing import absltest, parameterized
from transformers import AutoTokenizer

from baselines import configs, input_pipeline


@parameterized.parameters(list(configs.MODEL_BSM.keys()))
class DataTokenizationTest(parameterized.TestCase):
  """ Checks Tokenization for each model.  """
  def test_create_datset(self, model_name):
    default_tokenizer = AutoTokenizer.from_pretrained(model_name)
    default_special_tokens = copy.deepcopy(default_tokenizer.special_tokens_map)
    default_num_tokens = len(default_tokenizer)

    # load tokenizer and add special tokens
    model_config, tokenizer, _ = configs.hf_auto_configure(model_name)
    ds = input_pipeline.get_prost_ds(model_type=model_config.model_type,
                                     model_name=model_name, tokenizer=tokenizer,
                                     batch_size=1)

    special_token_diff = {
      k: f'"{default_special_tokens.get(k, None)}" -> "{v}"'
      for k, v in tokenizer.special_tokens_map.items()
      if default_special_tokens.get(k, None) != v}
    if special_token_diff != {}:
      logging.info('Special Token Changes: %s', special_token_diff)
      sample_level = logging.INFO
    elif model_config.model_type == 'albert':
      sample_level = logging.INFO
    else:
      sample_level = logging.DEBUG
    iids = ds['input_ids'][0]
    logging.log(
      sample_level,
      'Sample:\n Decoded: %s\n Tokens: %s',
      tokenizer.decode(iids),
      tokenizer.convert_ids_to_tokens(iids),
    )

    # make sure no tokens were added.
    self.assertEqual(len(tokenizer), default_num_tokens)


if __name__ == '__main__':
  absltest.main(failfast=True)

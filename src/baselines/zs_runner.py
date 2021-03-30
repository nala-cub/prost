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
""" Zeroshot Baseline code for all models. """
from __future__ import annotations

import time
from pathlib import Path

import labtools
import numpy as np
import pandas as pd
import toolz.curried as T
import torch
import torch.nn.functional as F
import tree
from absl import flags, logging
from einops import rearrange, reduce, repeat
from tqdm import tqdm

from baselines import configs, unifiedqa
from baselines.input_pipeline import get_prost_iter

FLAGS = flags.FLAGS


def run_experiment(model_name, batch_size, save_dir):
  model_config, tokenizer, model_cls = configs.hf_auto_configure(model_name)

  # maybe skip
  pred_path = save_dir / f'prost_{model_name.replace("/", "-")}_preds.csv'
  if pred_path.is_file() and FLAGS.use_cached_results:
    logging.info('Results already exist for %s, Skipping.', model_name)
    return

  batch_iter = get_prost_iter(model_name, model_config.model_type, tokenizer,
                              batch_size)

  is_generative = model_name.startswith('allenai/unifiedqa')

  # prep

  meta_vars = [
    'example_idx', 'option_input_ids', 'target_tok_idx', 'option_idx']

  # load model
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = model_cls.from_pretrained(model_name).to(device).eval()

  # stage step fn with static args.
  step_fn = test_step(model, is_generative, model_config.is_causal)

  def _inner(batch):
    batch = tree.map_structure(lambda x: x.to(device), batch)
    output, model_input = labtools.split_by_keys(batch, keys=meta_vars)
    return step_fn(output, model_input)

  with torch.no_grad():
    results = list(T.map(_inner, tqdm(batch_iter)))

  # flatten and save
  save_results(results, is_generative, tokenizer, pred_path)


@T.curry
def test_step(model, is_generative, is_causal, output, model_input):
  """ """
  # pop non-input vars (these are metadata)
  if is_generative:
    result = model.generate(model_input['input_ids'])
    output['generated'] = result
    return output

  # feed through model
  # lm_logits: <bs, seq_len, vacab_size>
  logits_lm = model(**model_input).logits

  if is_causal:
    # Shift so that tokens < n predict n (from transformers.modeling_gpt2)
    shift_logits = logits_lm[..., :-1, :].contiguous()
    shift_labels = model_input['input_ids'][..., 1:].contiguous()
    # set labels at pad indices to -100 s.t. loss=0 for pad
    shift_labels[model_input['attention_mask'][..., 1:].eq(0)] = -100
    # Flatten the tokens (no reduction)
    loss = F.cross_entropy(
      rearrange(shift_logits, 'b s v -> (b s) v'),
      rearrange(shift_labels, 'b s -> (b s)'), reduction='none',
      ignore_index=-100)
    # avg loss over each sequence
    loss = reduce(loss, '(b s) -> b', 'mean',
                  b=model_input['input_ids'].size(0))
    output['loss'] = loss
  else:
    num_options = output['option_input_ids'].shape[-1]

    if model.config.model_type == 't5':
      if logits_lm.shape[1] != 3:
        logging.fatal('Expected logits to have exatly 3 tokens, found %d',
                      logits_lm.shape[1])

      # subset by vocab for each sequeuence in the batch
      # logits_subset_options: <bs, seq_len, num_options>
      logits_lm = logits_lm[:, 1, :].squeeze(1)
      option_ids = output['option_input_ids']
      logits_subset_target_tok = torch.gather(logits_lm, -1, option_ids)
    else:
      # subset by vocab for each sequeuence in the batch
      # logits_subset_options: <bs, seq_len, num_options>
      option_ids = repeat(output['option_input_ids'],
                          'b num_options -> b seq_len num_options',
                          seq_len=logits_lm.shape[1])

      logits_subset_options = torch.gather(logits_lm, -1, option_ids)

      # pull the logits for the target token(s)
      # logits_subset_target_tok: <bs, 1, num_options>
      target_tok_idx = repeat(output['target_tok_idx'], 'b -> b num_options',
                              num_options=num_options).unsqueeze(-2)
      logits_subset_target_tok = torch.gather(logits_subset_options, 1,
                                              target_tok_idx)

    # get probs
    probs_subset_target_tok = F.softmax(logits_subset_target_tok.squeeze(1), -1)
    output['probs'] = probs_subset_target_tok

    # repeat to match fmt
    output['example_idx'] = repeat(output['example_idx'], 'b -> b vs',
                                   vs=num_options)
    output['target_tok_idx'] = repeat(output['target_tok_idx'], 'b -> b vs',
                                      vs=num_options)

  return output


def save_results(outputs, is_generative, tokenizer, pred_path):
  """ Flatten batched outputs and save predictions. """

  outputs = tree.map_structure(lambda *x: [*x], *outputs)
  # generated outputs may be misshaped
  # ic(outputs)

  if is_generative:
    generated = outputs.pop('generated')

  outputs = T.valmap(torch.cat, outputs)

  option_ids = outputs.pop('option_input_ids')

  if is_generative:
    option_tokens = T.pipe(option_ids, tokenizer.batch_decode, T.map(str.split))

    # generated idxs -> tokens
    generated = [g for gs in generated for g in gs]
    generated = tokenizer.batch_decode(generated, skip_special_tokens=True)

    idxs_and_scores = T.map(unifiedqa.get_pred_idx, generated, option_tokens)
    idxs, scores = list(zip(*list(idxs_and_scores)))

    outputs['pred_idx'] = torch.as_tensor(idxs)
    outputs['max_score'] = torch.as_tensor(scores)
    outputs['generated'] = generated

    # pop useless tags
    outputs.pop('option_idx')
  else:
    option_tokens = T.pipe(option_ids, torch.flatten, tokenizer.batch_decode,
                           T.map(str.split), T.map(T.get(0)))
    option_tokens = list(option_tokens)
    # flatten
    outputs = tree.map_structure(torch.flatten, outputs)
    # get tokens
    outputs['token'] = option_tokens

  # convert from tensors -> df
  outputs = tree.map_structure(labtools.topylist, outputs)
  results = pd.DataFrame(outputs)

  try:
    results['correct'] = results['pred_idx'] == results['target_tok_idx']
    logging.info('ACC: %0.03f', results['correct'].mean())
  except:
    pass
  logging.debug(results.head())
  results.to_csv(pred_path, index=False)


def run_zeroshot():
  """Runs zeroshot experiment(s) """
  save_dir = Path(FLAGS.results_dir) / 'preds'
  save_dir.mkdir(exist_ok=True, parents=True)
  logging.info('Saving results to %s', save_dir)

  # if none, runall
  model_names = FLAGS.model_name or list(configs.MODEL_BSM)

  tick = time.time()
  for model_name in model_names:
    batch_size = max([int(FLAGS.batch_size * configs.MODEL_BSM[model_name]), 1])
    logging.info('Running probe on %s with bs=%d', model_name, batch_size)

    with labtools.catch_exp_failures(model_name):
      run_experiment(model_name, batch_size, save_dir)

  logging.info('Total Runtime: %ds', time.time() - tick)

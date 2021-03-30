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
from __future__ import annotations

import re
from typing import Any, List, Tuple, Union, overload

import numpy as np
import toolz.curried as T
from absl import logging


def prep_example_prost_gcp(example):
  """ Prepare PROST example for the T5 Colab Notebook """

  template = '{ex_question} \\n (A) {A} (B) {B} (C) {C} (D) {D} \\n {context}'

  instance = {
    'input': template.format_map(example),
    'target': example[list('ABCD')[example['label']]],
    'target_idx': example['label'],
    **example}
  return instance


def normalize_answer(s: str) -> str:
  """Normalize UnifiedQA Generated Text [1]. """
  import re
  import string

  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  # Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
  def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  return T.pipe(s, str.lower, remove_punc, fix_buggy_characters,
                remove_articles, white_space_fix)


def score_unifiedqa(prediction: str, groundtruth: str) -> float:
  # Better than perfect token match
  if prediction == groundtruth:
    return 3.0
  prediction = normalize_answer(prediction)
  groundtruth = normalize_answer(groundtruth)

  # stipped matches
  if prediction == groundtruth:
    return 2.0

  # token overlap
  if ' ' in prediction or ' ' in groundtruth:
    prediction_split = prediction.split(' ')
    groundtruth_split = groundtruth.split(' ')
    overlap = list(set(prediction_split) & set(groundtruth_split))
    return len(overlap) / max(len(prediction_split), len(groundtruth_split))

  # single word, no match
  else:
    return 0.0


def get_pred_idx(prediction: str, groundtruth: list[str]) -> Tuple[int, float]:
  if len(groundtruth) == 0:
    return -1, -1.0
  scores = np.asarray([score_unifiedqa(prediction, gt) for gt in groundtruth])
  max_score: float = scores.max()
  pred_idx: int = -1 if max_score == 0 else scores.argmax()
  return pred_idx, max_score


def prep_example_piqa(example, index):
  """ Prepare PIQA example for scoring preds from the T5 notebook """
  match = re.match(r'^(.+) \\n \(A\) (.+) \(B\) (.+)$', example['input'])

  instance = {}
  instance['sol1'] = match[2].strip().lower()
  instance['sol2'] = match[3].strip().lower()

  example['target'] = example['target'].strip().lower()

  instance['example_idx'] = index
  instance['target'] = example['target']
  instance['target_idx'] = (instance['sol1'], instance['sol2'],
                            example['target']).index(example['target'])

  if instance['target_idx'] > 1:
    logging.warning('failed to parse solution index.')

  return instance

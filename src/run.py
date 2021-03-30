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
""" Baseline code for all models. 

  This is a wrapper around 3 scripts in baselines. See baselines.configs for
  other relavent flags, or use --helpfull.
  Note that uqa_finetune is not setup out of the box - we used the T5 Notebook
  for finetuning UnifiedQA on PIQA, the `uqa_finetune` script is dor uploading 
  and downloading data from GCP and scoring the predictions.
"""
from __future__ import annotations

from pathlib import Path

import labtools
from absl import app, flags, logging

from baselines.results import generate_all_results
from baselines.uqa_runner import run_uqa_finetune
from baselines.zs_runner import run_zeroshot

FLAGS = flags.FLAGS

flags.DEFINE_boolean('zeroshot', True, 'Run Zeroshot Baselines.')
flags.DEFINE_boolean('uqa_finetune', False, 'Run UnifiedQA Finetuning.')
flags.DEFINE_boolean('results', True, 'Generate all results.')


def main(_):
  """Parses flags and runs experiment(s). """
  labtools.configure_logging()
  save_dir = Path(FLAGS.results_dir) / 'preds'
  save_dir.mkdir(exist_ok=True, parents=True)
  logging.info('Saving results to %s', save_dir)

  if FLAGS.zeroshot:
    run_zeroshot()
  if FLAGS.uqa_finetune:
    run_uqa_finetune()
  if FLAGS.results:
    generate_all_results()


if __name__ == '__main__':
  app.run(main)

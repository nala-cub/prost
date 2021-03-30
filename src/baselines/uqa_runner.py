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

import re
from pathlib import Path

import labtools
import pandas as pd
from absl import flags, logging
from datasets import load_dataset
from google.cloud import storage

from baselines import unifiedqa

FLAGS = flags.FLAGS

flags.DEFINE_boolean('upload_data', False, 'Upload Datasets to GCP')
flags.DEFINE_boolean('download_preds', False, 'Download preds from GCP')


def _upload_ds_as_tsv(ds, key, name='dev'):
  lines = ['{input}\t{target}'.format_map(ex) for ex in ds]
  text = '\n'.join(lines)
  bucket = storage.Client('hfbeam').bucket('hfbeam-models')
  blob = bucket.blob(f'data/{key}/{name}.tsv')
  blob.upload_from_string(text)
  raise Exception


def upload_prost():
  ds = load_dataset('src/prost')['test']
  ds.cleanup_cache_files()
  ds = ds.map(unifiedqa.prep_example_prost_gcp)
  _upload_ds_as_tsv(ds, 'prost')


def get_uqa_preds_gcp():
  model_dirs = ['models/3B', 'models/3B-2']
  client = storage.Client('hfbeam')
  # download dir
  local_dir = Path(FLAGS.results_dir, 'finetune_preds')
  local_dir.mkdir(exist_ok=True)

  for model_dir in model_dirs:
    # get blobs
    prefix = f'{model_dir}/validation_eval/'
    blobs = client.list_blobs('hfbeam-models', prefix=prefix, delimiter='/')
    # save all matching to files
    for blob in blobs:
      fname = Path(blob.name).name
      if re.match(r'^(.+)_(\d+)_predictions$', fname):
        logging.info('Downloading preds: %s', fname)
        blob.download_to_filename(local_dir / fname.replace('_predictions', ''))


def score_piqa():
  # load & prep piqa
  gcp_path = 'data/physical_iqa/dev.tsv'
  local_path = Path().cwd() / gcp_path.replace('/', '')
  if not local_path.is_file():
    logging.info('Fetching PIQA from UnifiedQA bucket')
    bucket = storage.Client.create_anonymous_client().bucket('unifiedqa')
    bucket.get_blob(gcp_path).download_to_filename(local_path)

  piqa_ds = load_dataset('csv', sep='\t', names=['input', 'target'],
                         data_files=str(local_path))['train']
  piqa_ds.cleanup_cache_files()
  piqa_ds = piqa_ds.map(unifiedqa.prep_example_piqa, with_indices=True,
                        new_fingerprint='prep_examples')
  # load preds
  pred_dir = Path(FLAGS.results_dir, 'finetune_preds')
  checkpoint_preds = list(pred_dir.glob('physical_iqa_*'))

  savedir = Path(FLAGS.results_dir, 'finetune')
  savedir.mkdir(exist_ok=True)
  [p.unlink() for p in savedir.iterdir()]

  keys = ['sol1', 'sol2']

  for pred_path in checkpoint_preds:
    preds = pred_path.read_text().split('\n')
    outputs = []
    for ex, pred in labtools.safe_zip(piqa_ds, preds):
      pred_idx, max_score = unifiedqa.get_pred_idx(pred, [ex[k] for k in keys])
      output = {
        **ex, 'pred': pred,
        'pred_idx': pred_idx,
        'max_score': max_score}
      outputs.append(output)

    results = pd.DataFrame.from_records(outputs)
    results.to_csv((savedir / pred_path.name).with_suffix('.csv'), index=False)


def score_prost():
  # load & prep piqa
  ds = load_dataset('src/prost', split='test')
  ds = ds.map(unifiedqa.prep_example_prost_gcp)
  # load preds
  pred_dir = Path(FLAGS.results_dir, 'finetune_preds')
  checkpoint_preds = list(pred_dir.glob('prost_*'))

  savedir = Path(FLAGS.results_dir, 'finetune')
  savedir.mkdir(exist_ok=True)

  keys = list('ABCD')

  for pred_path in checkpoint_preds:
    preds = pred_path.read_text().split('\n')
    outputs = []
    for ex, pred in labtools.safe_zip(ds, preds):
      pred_idx, max_score = unifiedqa.get_pred_idx(pred, [ex[k] for k in keys])
      output = {
        **ex, 'pred': pred,
        'pred_idx': pred_idx,
        'max_score': max_score}
      outputs.append(output)

    results = pd.DataFrame.from_records(outputs)
    results.to_csv((savedir / pred_path.name).with_suffix('.csv'), index=False)


def run_uqa_finetune():
  Path(FLAGS.results_dir).mkdir(exist_ok=True, parents=True)
  if FLAGS.upload_data:
    upload_prost()

  if FLAGS.download_preds:
    get_uqa_preds_gcp()

  # score preds
  score_piqa()
  score_prost()

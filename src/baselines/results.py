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
""" Baselines Result Analysis """
from __future__ import annotations

import functools
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import altair as alt
import datasets
import labtools
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from absl import flags, logging
from icecream import ic
from transformers import AutoModelForPreTraining

from baselines import configs

FLAGS = flags.FLAGS


def generate_all_results(cachedir=None):
  # get dirs
  cachedir = cachedir or Path(FLAGS.results_dir)
  tex_dir = cachedir / 'tabs'
  tex_dir.mkdir(exist_ok=True)
  fig_dir = cachedir / 'figs'
  fig_dir.mkdir(exist_ok=True)

  # make result tables
  make_fullresults(cachedir, tex_dir)
  make_position_results(cachedir, tex_dir)
  make_inverse_results(cachedir, tex_dir)

  # collect metadata
  collect_metadata(cachedir, tex_dir)

  # make result figures
  labtools.setup_plotting_themes()
  make_scaling_figure(cachedir)
  make_uqa_ft_figure(cachedir)


def load_raw_preds(cachedir):
  preds_dir = cachedir / 'preds'

  dfs = []
  for pred_file in preds_dir.glob('*_preds.csv'):
    match = re.match(r'^(\w+)_(.+)_preds.csv$', pred_file.name)
    model_name = match[2]
    if not model_name.endswith('v1'):
      _df = pd.read_csv(pred_file)
      _df['model_name'] = model_name
      dfs.append(_df)

  # join results
  df = pd.concat(dfs)
  # print(df)
  return df


def load_preds(cachedir):
  # load dataset
  ds_path = str(Path(__file__).parent.parent / 'prost')
  ds = datasets.load_dataset(ds_path, split='test')
  # load preds
  df = load_raw_preds(cachedir)
  # set dataset as df
  ds.set_format('pd')
  info = ds[:]
  info['example_idx'] = info.index

  df = df.merge(info, how='left', on='example_idx')
  return df, info


pretty_column_map = {
  'model': {
    'gpt': 'GPT',
    'gpt2': 'GPT-2',
    'bert': 'BERT',
    'roberta': 'RoBERTa',
    'albert v1': 'ALBERT V1',
    'albert v2': 'ALBERT V2',
    't5': 'T5',
    'unifiedqa': 'UnifiedQA'},
  'model_type': {
    'gpt': 'GPT',
    'gpt2': 'GPT-2',
    'bert': 'BERT',
    'roberta': 'RoBERTa',
    'albert v1': 'ALBERT V1',
    'albert v2': 'ALBERT V2',
    't5': 'T5',
    'unifiedqa': 'UnifiedQA'}}
model_cmap = {
  'gpt': '#AA0DFE',
  'gpt2': '#782AB6',
  'bert': '#16FF32',
  'roberta': '#1CBE4F',
  'albert v1': '#2ED9FF',
  'albert v2': '#3283FE',
  't5': '#FEAF16'}

task_order = [
  'Directions',
  'Mass',
  'Height',
  'Circum.',
  'Stacking',
  'Rolling',
  'Grasping',
  'Breaking',
  'Sliding',
  'Bouncing',]


def get_model_type(x):
  if x == 'openai-gpt' or x.startswith('allenai-unifiedqa'):
    return x.split('-')[1]
  elif x.startswith('albert'):
    return ' '.join([x.split('-')[0], x.split('-')[-1]])
  else:
    return x.split('-')[0]


def _splitsort(cols):
  split_cols = [c.split('-') for c in cols]

  prefix = [
    'openai', 'gpt2', 'bert', 'roberta', 'albert', 't5', 'allenai',
    *[c[0] for c in split_cols]]
  suffix = [
    'gpt', 'gpt2', 'small', 'base', 'medium', 'large',
    *[c[-1] for c in split_cols]]
  # print(split_cols)

  split_cols = sorted(split_cols, key=lambda x:
                      (prefix.index(x[0]), suffix.index(x[-1])))
  return split_cols


def sort_columns(cols):
  split_cols = _splitsort(cols)
  return ['-'.join(c) for c in split_cols]


def make_nice_rows(cols):
  split_cols = _splitsort(cols)
  nice_cols = []
  for parts in split_cols:
    if parts[0] == 'openai':
      nice_cols.append('GPT & ')
      continue
    # 1st token is model name
    if parts[0] == 'roberta':
      name = 'RoBERTa'
    elif parts[0] == 'allenai':
      name = 'UnifiedQA'
    else:
      name = parts[0].upper()
    if parts[0] == 'albert':
      name += ' ' + parts[2].upper()
    if parts[0] == 'allenai':
      name += ' ' + parts[-1]
    elif len(parts) > 1:
      name += ' ' + parts[1]  #.upper()
    else:
      name += ' & '

    # do replace
    rmap = [
      ('small', '& S'),
      ('base', '& B'),
      ('medium', '& M'),
      ('xxlarge', '& XXL'),
      ('xlarge', '& XL'),
      ('xl', '& XL'),
      ('large', '& L'),
      ('3b', '& 3B'),]
    for i, o in rmap:
      name = name.replace(i, o)
    #name = name.replace

    nice_cols.append(name)
  # join
  #print(split_cols)
  #print(nice_cols)
  return nice_cols


def pivoted_to_longtex(df, bold=None, dest=None):
  """ Takes in a pivoted DF  and produces a LaTeX Table
        Each row should be a model and each column a Task.
    """
  dfl = df.copy(deep=True)

  header = r'\begin{tabular}{rl' + 'c' * (len(dfl.columns) + 2) + '}\n'
  header += r'\toprule' + '\n'
  # column headers
  dfl = dfl.reindex(sort_columns(dfl.index))

  # make and order tasks
  colmap = {c: c.capitalize() for c in dfl.columns}
  colmap['circumference'] = 'Circum.'
  dfl = dfl.rename(columns=colmap)
  #cols =

  dfl = dfl[[c for c in task_order if c in dfl.columns]]

  dfl['Average'] = dfl.mean(1)
  # print(dfl.head)
  # dfl = dfl[sort_columns(dfl.index)]
  # dfl = dfl.reindex(sort_columns(dfl.index))
  nice_rows = make_nice_rows(dfl.index)
  # print(nice_rows)
  # nice_cols = [i.capitalize()  for i in dfl.columns]

  ## Exlcluding unifiedqa
  dfl_no_uqa = dfl[~dfl.index.str.startswith('allenai-unifiedqa')]

  header += r'\multicolumn{2}{c}{Model} & ' + ' & '.join(
    dfl.columns) + r' \\' + '\n' + r'\midrule' + '\n'
  if bold == 'max':
    best_models_idxs = dfl_no_uqa.idxmax(0)
  elif bold == 'min':
    best_models_idxs = dfl_no_uqa.idxmin(0)
  else:
    best_models_idxs = defaultdict(lambda: None)

  # print(best_models_idxs)
  last_model = nice_rows[0].split('&')[0]
  # dfl = dfl.replace()

  ic(dfl)

  for ri, (i, row) in enumerate(dfl.iterrows()):
    #print(row)
    strrow = [('---' if np.isnan(r) else f'{r*100:.1f}') for r in row]
    for j, task in enumerate(row.index):
      if best_models_idxs[task] == i:
        strrow[j] = r'\textbf{' + strrow[j] + '}'

    if nice_rows[ri].split('&')[0] != last_model:
      header += r'\midrule' + '\n'
      last_model = nice_rows[ri].split('&')[0]
    elif ri == 0:
      pass
    else:
      nice_rows[ri] = '&' + nice_rows[ri].split('&')[1]
      #print(last_model, nice_rows[ri])

    # best per row is bold
    #print(row.argmin())
    #strrow[row.argmax()] = r'{\bf ' + strrow[row.argmax()] + '}'
    header += nice_rows[ri] + ' & ' + ' & '.join(strrow) + r' \\' + '\n'

    #print(i, row)
  header += r'\midrule' + '\n' + r'Task Avg. & & '
  # average over tasks
  # uf_task_avg = dfl.mean(0)
  task_avg = dfl[~dfl.index.str.startswith('allenai-unifiedqa')].mean(0)
  # ic(task_avg, uf_task_avg, dfl[~dfl.index.str.startswith('allenai-unifiedqa')])

  strrow = [('-' if np.isnan(r) else f'{r*100:.1f}') for r in task_avg]

  header += ' & '.join(strrow) + r' \\' + '\n'

  header += r'\bottomrule' + '\n' + r'\end{tabular}'
  if dest is not None:
    with open(dest, 'w') as f:
      f.write(header)

  return header


def to_wide_rankings(df):
  # df = df.parallel_apply(_get_option_idx, axis=1)
  dfp = df.copy()
  vcols = [c for c in ['probs', 'loss', 'target_tok_idx'] if c in dfp.columns]
  dfp = dfp.pivot_table(index=['example_idx', 'model_name'],
                        columns=['option_idx'], values=vcols)
  # min loss (gpt)
  try:
    dfp[('pred_idx', 'loss')] = dfp[('loss',)][[0, 1, 2, 3]].idxmin(1)
  except KeyError:
    logging.exception('Cannot aggregate results for gpt,gpt2')
  # max probs (bert, roberta, albert, t5)
  try:
    dfp[('pred_idx', 'probs')] = dfp[('probs',)][[0, 1, 2, 3]].idxmax(1)
  except KeyError:
    logging.exception('Cannot aggregate results for bert, roberta, albert, t5')
  dfp = dfp.reset_index()
  dfp[('pred_idx', 'final')] = dfp[('pred_idx',)].fillna(-1).max(1)
  return dfp


def process_preds_base(df, info):
  """ Processes all Predictions into Longform DF with Dataset Examples + Correct Tags.

  """
  dfp = to_wide_rankings(df)
  dfp = dfp[[('example_idx', ''), ('model_name', ''), ('pred_idx', 'final')]]
  dfp.columns = dfp.columns.droplevel(1)
  # add unifiedqa
  uqa_df = df[df['model_name'].apply(lambda x: x.startswith('allenai'))]
  if len(uqa_df.index) > 0:
    # uqa_df = uqa_df[['example_idx', 'model_name', 'pred_idx', 'generated']]
    uqa_df = uqa_df[['example_idx', 'model_name', 'pred_idx']]
    dfp = pd.concat([dfp, uqa_df])
  # add info
  dfm = dfp.merge(info, on='example_idx', how='left')
  # get correct (per example)
  dfm['correct'] = dfm['pred_idx'] == dfm['label']
  # get groupings
  dfm['question_group'] = dfm['name'].apply(
    lambda x: x.split('_')[0].replace('non', ''))
  return dfm


def process_preds(df, info):
  dfm = process_preds_base(df, info)
  dfm = dfm.groupby(['model_name', 'question_group', 'name'],
                    as_index=False).mean()
  dfm = dfm.groupby(['model_name', 'question_group'], as_index=False).mean()
  dfm = dfm.pivot('question_group', 'model_name', 'correct')
  return dfm


def make_fullresults(cachedir, dest_dir):
  df, info = load_preds(cachedir)
  dfm = process_preds(df, info)
  dfm = dfm[sort_columns(dfm.columns)]
  dfm.to_csv(dest_dir / 'fullresults.csv')
  # dfm['task_avg'] = dfm.mean(1)
  dfm = dfm.T
  dfm['avg'] = dfm.mean(1)
  # Accuracy

  pivoted_to_longtex(dfm, bold='max', dest=dest_dir / 'fullresults.tex')


def process_preds_positions(df, info, group_models=True):
  dfm = process_preds_base(df, info)
  # get the index of the corrrect option within the context
  # names are in the format:
  # <basename>_i or <basename>_i_j or non<basename>_i or non<basename>_i_j
  # in all cases i represents the correct_tok_position
  dfm['correct_tok_position'] = dfm['name'].apply(lambda x: x.split('_')[1])
  # take the avg over each tok position for each template (with the same name) for each model
  dfm = dfm.groupby(['model_name', 'name', 'correct_tok_position'],
                    as_index=False).mean()

  if group_models:
    dfm['model'] = dfm['model_name'].apply(get_model_type)
    # avg by model group
    # dfs = dfm.groupby(['model','correct_tok_position'], as_index=False).std()
    dfm = dfm.groupby(['model', 'correct_tok_position'], as_index=False).mean()
    #dfm = dfm.merge(dfm, dfs)
    dfm = dfm.replace(pretty_column_map)
    dfm = dfm.pivot('model', 'correct_tok_position', 'correct') * 100
  else:
    dfm = dfm.groupby(['model_name', 'correct_tok_position'],
                      as_index=False).mean()
    dfm = dfm.pivot('model_name', 'correct_tok_position', 'correct') * 100

  return dfm


def add_avg_row(df):
  avg = df.mean(0)
  avg.name = 'Average'
  df = df.append(avg)
  return df


def make_position_results(cachedir, dest_dir):
  # Model Grouping
  df, info = load_preds(cachedir)
  dfm = df.pipe(process_preds_positions, info, True)\
          .pipe(add_avg_row)
  save_table(dfm, dest_dir / 'position_acc_m')

  # Model+size grouping
  df, info = load_preds(cachedir)
  dfm = df.pipe(process_preds_positions, info, False)\
          .pipe(add_avg_row)
  save_table(dfm, dest_dir / 'position_acc_ms')


def save_table(df, path):
  df.to_csv(path.with_suffix('.csv'))
  df.to_latex(path.with_suffix('.tex'), float_format="%.2f")


# ------------------------------------------------------------------------------
def is_inverse(name):
  # directions has no inverses
  if name.startswith('directions'):
    return np.nan
  # e.g. nonbouncing
  if name.startswith('non'):
    return True
  # e.g. height_1_b (least instad of most)
  if name.split('_')[-1] == 'b':
    return True
  return False


def make_inverse_results(cachedir, dest_dir):
  """ Inverse Superlatives """
  # Model+size grouping
  df, info = load_preds(cachedir)
  dfm = process_preds_inverse(df, info)
  dfm.to_csv(dest_dir / 'inverses.csv', index=False)
  pivoted_to_longtex(dfm, bold='min', dest=dest_dir / 'inverses.tex')


def process_preds_inverse(df, info):
  dfm = process_preds_base(df, info)
  # get model
  dfm['model'] = dfm['model_name'].apply(get_model_type)
  # filter out examples which have no inverse
  dfm['is_inverse'] = dfm['name'].apply(is_inverse)
  dfm = dfm.dropna(subset=['is_inverse'])
  # take avg by model_name
  dfp = dfm.groupby(['model', 'model_name', 'question_group', 'is_inverse'],
                    as_index=False).mean()
  # partition into disjoint sets
  dfp1 = dfp[~dfp['is_inverse']]
  dfp2 = dfp[dfp['is_inverse']]
  # check size
  assert len(dfp1) == len(dfp2)
  # merge and abs diff
  dff = dfp1.merge(
    dfp2,
    how='left',
    on=['model', 'model_name', 'question_group'],
  )
  dff['inverse_diff'] = (dff['correct_y'] - dff['correct_x']).abs()
  dffp = dff.pivot('model_name', 'question_group', 'inverse_diff')  # * 100
  return dffp


@functools.lru_cache(None)
def get_model_metadata(model_name):
  model_name = model_name.replace('allenai-', 'allenai/')
  return AutoModelForPreTraining.from_pretrained(model_name).num_parameters()


def collect_model_metadata(names):
  metadata = {}
  for model_name in names:
    metadata[model_name] = get_model_metadata(model_name)
  return metadata


def get_meta_df(cachedir, model_names):
  sdir = cachedir / 'meta'
  sdir.mkdir(exist_ok=True)
  sp = sdir / 'metadata.json'
  if not sp.is_file():
    metadata = collect_model_metadata(model_names)
    sp.write_text(json.dumps(metadata))
  else:
    metadata = json.loads(sp.read_text())
  metadata = {k: {'num_params': v} for k, v in metadata.items()}
  pub_info = {
    'gpt2': date(2019, 2, 14),  # February 14, 2019
    'bert': date(2018, 10, 11),  # Oct 11, 2018
    'gpt': date(2018, 6, 11),  # June 11, 2018
    'albert v1': date(2019, 9, 26),  # Sep 26, 2019
    'albert v2': date(2019, 9, 26),  # Sep 26, 2019
    'roberta': date(2019, 6, 26),  # Jul 26, 2019
    't5': date(2019, 10, 29),  # Oct 29, 2019
    'unifiedqa': date(2020, 5, 2),  #  May 2, 2020
  }
  data_info = {
    'gpt': 2,  # BookCorpus Only
    'gpt2': 40,  # Webtext
    'bert': 13,  # Wiki+Books
    'roberta': 160,  # 160 GB
    'albert v1': 13,  # same as bert
    'albert v2': 160,  # same as roberta
    't5': 170,  # C4
    'unifiedqa': 170,  # C4
  }
  # create df.
  dfp = pd.DataFrame.from_dict(metadata, orient='index')
  # add other data
  dfp['model_name'] = dfp.index.str.replace('allenai/', 'allenai-')
  dfp['model_type'] = dfp['model_name'].apply(get_model_type)
  dfp['date_of_publication'] = dfp['model_type'].apply(lambda x: pub_info[x])
  dfp['dataset_size'] = dfp['model_type'].apply(lambda x: data_info[x])
  dfp['dataset_size_text'] = dfp['dataset_size'].apply(lambda x: f'{x} GB')
  return dfp


def collect_metadata(cachedir, dest_dir):
  # get model_names
  df = load_raw_preds(cachedir)
  model_names = list(configs.MODEL_BSM)
  meta_df = get_meta_df(cachedir, model_names)
  meta_df.to_csv(dest_dir / 'metadata.csv', index=False)


def reformat(df):
  df = df.T
  df.columns = df.iloc[0]
  df = df[1:] * 100
  colmap = {c: c.capitalize() for c in df.columns}
  colmap['circumference'] = 'Circum.'
  df = df.rename(columns=colmap)
  df = df[[c for c in task_order if c in df.columns]]
  df['Average'] = df.mean(1)
  return df


def make_scaling_figure(cachedir):
  df = pd.read_csv(cachedir / 'tabs/fullresults.csv')
  meta = pd.read_csv(cachedir / 'tabs/metadata.csv')

  dfp = reformat(df)
  dfp['model_name'] = dfp.index
  dfp = dfp[['Average', 'model_name']]
  dfp = dfp.merge(meta)
  color_discrete_map = {
    'GPT': '#AA0DFE',
    'GPT-2': '#782AB6',
    'BERT': '#16FF32',
    'RoBERTa': '#1CBE4F',
    'ALBERT V1': '#2ED9FF',
    'ALBERT V2': '#3283FE',
    'T5': '#FEAF16',
    'UnifiedQA': '#FA0087'}
  dfp = dfp.replace(pretty_column_map)
  dfp = dfp[dfp['model_type'] != 'ALBERT V1']

  df = dfp.sort_values('num_params', ascending=False)

  legend_names = [
    ('GPT', '#AA0DFE'),
    ('GPT-2', '#782AB6'),
    ('BERT', '#16FF32'),
    ('RoBERTa', '#1CBE4F'),
    ('ALBERT V2', '#3283FE'),
    ('T5', '#FEAF16'),
    ('UnifiedQA', '#FA0087'),]

  df.to_hdf(cachedir / 'results.h5', 'scaling_results')

  df['marker_size'] = (2 * (df['num_params'] / 1e6)) + 50

  df['pcolor'] = df['model_type'].map(color_discrete_map.get)
  fig = go.Figure()
  for _, row in df.iterrows():
    fig.add_trace(
      go.Scatter(x=[row['dataset_size']], y=[row['Average']], mode='markers',
                 marker=dict(size=[row['marker_size']],
                             color=row['pcolor']), showlegend=False))
  for name, color in legend_names:
    fig.add_trace(
      go.Scatter(x=[None], y=[None], mode='markers',
                 marker=dict(size=10, color=color), showlegend=True, name=name))

  fig.update_layout(xaxis_title='<b>Amount of Pretraining Data (GB)<b>',
                    yaxis_title='<b>Accuracy<b>', width=500, height=600)
  fig.update_traces(
    marker=dict(sizemode='area', sizemin=4, line={
      'color': '#333333',
      'width': 1}))

  fig.write_image(
    str(cachedir / 'figs/scaling_params_data.pdf'), engine="kaleido", scale=5)
  return fig


def make_uqa_ft_figure(cachedir):
  records = []
  for fpath in (cachedir / 'finetune').iterdir():
    m = re.match(r'^(.+)_task_(\d+)$', fpath.stem)
    df = pd.read_csv(fpath)
    df['correct'] = df['target_idx'] == df['pred_idx']
    df['accuracy'] = df['correct']
    # macro
    if m[1] == 'prost':
      # get groupings
      df['question_group'] = df['name'].apply(
        lambda x: x.split('_')[0].replace('non', ''))
      df = df.groupby(['question_group', 'name'], as_index=False).mean()
      df = df.groupby(['question_group'], as_index=False).mean()
      # drop sliding
      df = df[df['question_group'] != 'sliding']

    metrics = {
      'Task': m[1],
      'Step': m[2],
      'Finetune Step': int(m[2]) - 1_120_000,
      'Accuracy': df['accuracy'].mean() * 100}
    records.append(metrics)
  df = pd.DataFrame.from_records(records)
  df = df.replace({'Task': {'physical_iqa': 'PIQA', 'prost': 'PROST'}})
  df.head()

  ## sepearate
  ns = df.sort_values('Accuracy').groupby('Task').tail(
    1)['Finetune Step'].values

  df['label'] = df['Accuracy'].round(2)

  piqa = df[df.Task == 'PIQA']
  prost = df[df.Task == 'PROST']

  piqa_points = piqa[piqa['Finetune Step'].isin(ns)]
  prost_points = prost[prost['Finetune Step'].isin(ns)]

  xscale = alt.Scale(type='linear')

  x_axis = alt.X('Finetune Step', scale=xscale)
  y_axis = alt.Y('Accuracy', scale=alt.Scale(type='linear', zero=False,
                                             domain=(70, 90)),
                 axis=alt.Axis(title='Accuracy (PIQA)', titleColor='#5276A7'))

  y2 = alt.Y('Accuracy', scale=alt.Scale(type='linear', zero=False,
                                         domain=(40, 60)),
             axis=alt.Axis(title='Accuracy (PROST)', titleColor='#57A44C'))

  # draw lines
  line = alt.Chart(piqa).mark_line(color='#5276A7')
  line = line.encode(x=x_axis, y=y_axis)

  point = alt.Chart(piqa_points).mark_point(size=80, opacity=1, color='#5276A7')
  point = point.encode(x=x_axis, y=y_axis)

  line2 = alt.Chart(prost).mark_line(color='#57A44C')
  line2 = line2.encode(x=x_axis, y=y2)

  point2 = alt.Chart(prost_points).mark_point(size=80, color='#57A44C')
  point2 = point2.encode(x=x_axis, y=y2)

  piqa_text = point.mark_text(align='left', baseline='middle', dx=10,
                              dy=-4).encode(text='label')

  prost_text = point2.mark_text(align='left', baseline='middle', dx=10,
                                dy=-4).encode(text='label')

  piqa_chart = alt.layer(line, point, piqa_text).resolve_scale()
  prost_chart = alt.layer(line2, point2, prost_text).resolve_scale()
  chart = alt.layer(piqa_chart, prost_chart)
  chart = chart.resolve_scale(color='independent', shape='independent',
                              y='independent')
  chart = chart.configure_legend(orient='right')
  try:
    altair_saver = labtools.altair_saver()
    altair_saver.save(chart, str(cachedir / 'figs/acc-uqa-ft.pdf'))
  except:
    pass
  return chart

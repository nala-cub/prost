# Copyright 2020 The HuggingFace Datasets Authors and The PROST Authors
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TODO: Add a description here."""
from __future__ import annotations

import copy
import itertools
import json
import re
import string
from collections.abc import Sequence
from enum import IntEnum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import datasets
import numpy as np
import toolz.curried as T
import tree
import yaml

logging = datasets.logging.get_logger(__name__)

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
"""
_HOMEPAGE = 'https://github.com/nala-cub/prost'
_LICENSE = 'Apache 2.0'

# TODO swap with URL
# _URL = 'https://raw.githubusercontent.com/nala-cub/prost/data/prost'
_URL = '/home/corypaik/research/prost/data/prost'

_TEMPLATE_GROUPS = [
  'bouncing', 'breaking', 'circumference', 'directions', 'grasping', 'height',
  'mass', 'rolling', 'sliding', 'stacking']

_URLs = {
  'scenario_ymls': [f'{_URL}/{name}.yml' for name in _TEMPLATE_GROUPS],
  'lexicon_yml': f'{_URL}/globals/lexicons.yml',}

MC_LABELS = list('ABCD')


class Prost(datasets.GeneratorBasedBuilder):
  """TODO: Short description of my dataset."""

  VERSION = datasets.Version('1.0.0')

  def _info(self):
    features = datasets.Features({
      'A': datasets.Value('string'),
      'B': datasets.Value('string'),
      'C': datasets.Value('string'),
      'D': datasets.Value('string'),
      'context': datasets.Value('string'),
      'question': datasets.Value('string'),
      'ex_question': datasets.Value('string'),
      'group': datasets.Value('string'),
      'label': datasets.ClassLabel(names=MC_LABELS),
      'example_idx': datasets.Value('int32'),
      'name': datasets.Value('string'),})
    return datasets.DatasetInfo(description=_DESCRIPTION, features=features,
                                supervised_keys=None, homepage=_HOMEPAGE,
                                license=_LICENSE, citation=_CITATION)

  def _split_generators(self, dl_manager):
    """ Returns SplitGenerators."""
    data_paths = dl_manager.download_and_extract(_URLs)
    return [datasets.SplitGenerator(datasets.Split.TEST, gen_kwargs=data_paths)]

  def _generate_examples(self, scenario_ymls, lexicon_yml):

    # fetch global lexicons
    with open(lexicon_yml, mode='r') as f:
      global_variables = yaml.safe_load(f)

    builder_fn = build_examples_from_config(variables=global_variables)

    #  build examples.
    # we sort by a json dump to make it hermetic.
    examples = T.pipe(scenario_ymls, T.mapcat(load_and_check_config_yml),
                      T.mapcat(builder_fn),
                      T.sorted(key=lambda x: json.dumps(x, sort_keys=True)))

    for id_, example in enumerate(examples):
      example['group'] = example['name'].split('_')[0].replace('non', '')
      example['example_idx'] = id_
      yield id_, example


def load_and_check_config_yml(path):
  logging.debug('Loading config YML from %s', path)
  # collect all tasks
  with open(path, mode='r') as f:
    loaded = yaml.safe_load(f)
  try:
    task_configs = loaded['configs']
  except KeyError:
    logging.fatal('Failed to load from %s', path)
  return task_configs


@T.curry
def build_examples_from_config(config, variables, product=True,
                               remove_duplicates=True):
  """Construct Test cases for a task.

  Args:
    config: raw dictionary configurary read in from scenario yml file.
    variables: global and scenario lexicons.
  Returns:
    Formatted examples for the provided template
  """
  # get answer key function, prep evs
  expect_fn = globals()[config.pop('expect_fn')]
  enum_variables = T.itemmap(lambda x: make_enum_vars(*x), variables)
  logging.debug('variables: %s, ev: %s', variables, enum_variables)

  # find keys for the template
  template = copy.deepcopy(config)
  template.update(template.pop('choices'))
  all_keys = find_all_keys(template)

  # build value sets for the template
  # fillin_items: {lexicon: <values>, }
  fillin_items = _get_fillin_items(all_keys, **variables)
  fillin_values = ensure_listlike(list(fillin_items.values()))
  val_sets = (itertools.product if product else zip)(*fillin_values)

  # remove all sets with duplicate values and restructure
  if remove_duplicates:
    val_sets = T.filter(lambda x: len(set(x)) == len(x), val_sets)

  # restructure as var -> value maps
  mappings = T.map(lambda vals: dict(zip(list(fillin_items), vals)), val_sets)

  def generate_example(mapping):
    data = recursive_format(template, mapping)
    example = {**data, 'label': expect_fn(data, mapping, ev=enum_variables)}
    # Capitilize first word.
    for k in ('context', 'question', 'ex_question'):
      example[k] = example[k][0].upper() + example[k][1:]
    return example

  examples = list(T.map(generate_example, mappings))
  logging.debug('Built %d from task %s', len(examples), config['name'])
  return examples


def make_enum_vars(k, vals):
  return k, IntEnum(k, {to_ek(v): i for i, v in enumerate(vals)})


def to_ek(s: str) -> str:
  return s.replace(' ', '').upper()


def is_listlike(x: Any) -> bool:
  return isinstance(x, Sequence) and not isinstance(x, str)


def ensure_listlike(x: Any):
  return x if is_listlike(x) else [x]


def enum_meta(meta: dict[str, str], ev: dict[str, IntEnum]):
  def _enum_meta(item) -> Tuple[str, dict[str, Any]]:
    k, v = item
    lexicon = _remove_counts(k)
    value = ev[lexicon][to_ek(v)]  # pytype: disable=unsupported-operands
    return k, {'text': v, 'value': value, 'lexicon': lexicon}

  return T.itemmap(_enum_meta, meta)


def preprocess_meta(fn: Callable):
  """ Preprocess meta dicts -> IntEnum objects"""
  @wraps(fn)
  @T.curry
  def _preprocess_meta(ex, meta, ev, **kwargs):
    # enumerate meta
    meta = enum_meta(meta, ev)

    # get options
    options = []
    for i, k in enumerate('ABCD'):
      option_meta = T.valfilter(lambda x: x['text'] == ex[k], meta)
      option_meta = list(option_meta.values())
      if len(option_meta) == 0:
        continue
      assert len(option_meta) == 1, (option_meta, meta)
      # get the enum for that obj
      options.append({'key': k, 'idx': i, **option_meta[0]})
    meta['options'] = options
    return fn(ex, meta, **kwargs)

  return _preprocess_meta


@T.curry
def _constant(label, ex, meta, ev):
  return label


# Constant Functions
constant_a = _constant(0)
constant_b = _constant(1)
constant_c = _constant(2)
constant_d = _constant(3)


@preprocess_meta
def turning(ex, meta):
  """ For directions. """
  return (meta['coord']['value'] + (meta['turn']['value'] + 1)) % 4


@preprocess_meta
def pick_odd_one_out(ex, meta):
  """ Pick object with different property """
  lexicon_groups = T.groupby('lexicon', meta['options'])

  odd_one_outs = T.valfilter(lambda x: len(x) == 1, lexicon_groups)
  assert len(odd_one_outs) == 1, odd_one_outs
  label = list(odd_one_outs.values())[0][0]['idx']

  return label


@preprocess_meta
def apply_enum_func(ex, meta, fn: Callable[[list[int]], int], max_n=4):
  # all matching numbered keys where keyidx <= max_n
  label = T.pipe(meta['options'], T.sorted(key=T.get('idx')), T.take(max_n),
                 T.map(T.get('value')), list, fn, int)
  return label


# Ranked objects
pick_ranked_min = apply_enum_func(fn=np.argmin)
pick_ranked_max = apply_enum_func(fn=np.argmax)
pick_ranked_filtered_min = apply_enum_func(fn=np.argmin, max_n=2)
pick_ranked_filtered_max = apply_enum_func(fn=np.argmax, max_n=2)

################################################################################
# MIT License
#
# Copyright (c) 2020 Marco Tulio Correia Ribeiro
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
""" Template Formatting for PROST. 

  A large portion of this section is based on the Checklist Implementation [1].
  The original Checklist licence is provided above.

  Sources:
      [1] https://github.com/marcotcr/checklist 
"""

TemplateObj = TypeVar('TemplateObj')


def add_article(noun: str) -> str:
  article = 'an' if noun[0].lower() in list('aeiou') else 'a'
  return '%s %s' % (article, noun)


def _remove_articles(k: str) -> str:
  """ Remove article prefixes [1]."""
  k = re.sub(r'\..*', '', k)
  k = re.sub(r'\[.*\]', '', k)
  # article
  k = re.sub(r'.*?:', '', k)
  return k


def _remove_counts(k: str) -> str:
  """ Remove count suffixes [1]."""
  return re.sub(r'\d+$', '', k)


def keys_to_var_names(keys):
  """ Convert list of keys to list of variable names. """
  # remove articles and clear duplicates
  var_keys = [_remove_counts(_remove_articles(k)) for k in keys]
  # the lists should be the same length
  assert len(keys) == len(var_keys)
  return var_keys


def _get_fillin_items(all_keys, max_count: Optional[int] = None, **kwargs):
  # items = {}
  for k in kwargs:
    if re.search(r'\d+$', k):
      raise Exception(
        'Keys cannot end in integers, we use that to index multiple copies'
        'of the same key (offending key: "%s")' % k)
  var_names = keys_to_var_names(all_keys)
  items = T.valmap(lambda x: kwargs.get(x, None),
                   dict(zip(T.map(_remove_articles, all_keys), var_names)))
  missing_lexicons = list(T.valfilter(lambda x: x is None, items))
  if missing_lexicons != []:
    raise Exception('Error: Missing keys in lexicons: %s' % missing_lexicons)
  if max_count:
    items = T.map(T.take(max_count), items)
  return items


def find_all_keys(obj) -> set[str]:
  """Finds all tag keys in object (with options) """
  return T.pipe(obj, tree.flatten, set,
                T.mapcat(lambda x: string.Formatter().parse(x)),
                T.filter(T.get(1)),
                T.map(lambda x: x[1] if not x[2] else '%s:%s' % (x[1], x[2])),
                list, set)


class SafeFormatter(string.Formatter):
  def vformat(self, format_string, args, kwargs):
    args_len = len(args)  # for checking IndexError
    tokens = []
    for (lit, name, spec, conv) in self.parse(format_string):
      # re-escape braces that parse() unescaped
      lit = lit.replace('{', '{{').replace('}', '}}')
      # only lit is non-None at the end of the string
      if name is None:
        tokens.append(lit)
      else:
        # but conv and spec are None if unused
        conv = '!' + conv if conv else ''
        spec = ':' + spec if spec else ''
        # name includes indexing ([blah]) and attributes (.blah)
        # so get just the first part
        fp = name.split('[')[0].split('.')[0]
        # treat as normal if fp is empty (an implicit
        # positional arg), a digit (an explicit positional
        # arg) or if it is in kwargs
        if not fp or fp.isdigit() or fp in kwargs:
          tokens.extend([lit, '{', name, conv, spec, '}'])
        # otherwise escape the braces
        else:
          tokens.extend([lit, '{{', name, conv, spec, '}}'])
    format_string = ''.join(tokens)  # put the string back together
    # finally call the default formatter
    return string.Formatter.vformat(self, format_string, args, kwargs)


def recursive_format(obj: TemplateObj, mapping: Dict,
                     ignore_missing: bool = False) -> TemplateObj:
  """Formats all strings within an object, using mapping
    
  Args:
    obj: Object (leaves must be strings, regardless of type)
    mapping: format dictionary, maps keys to values
    ignore_missing:  If True, will not throw exception if a string contains a 
      tag not present in mapping, and will keep the tag instead.
  Returns:
    Object of the same type as obj, with strings formatted (tags replaced
    by their value)
  """
  def formatfn(x):
    fmt = SafeFormatter()
    formatz = (lambda x, m: x.format(**m)
               if not ignore_missing else fmt.format(x, **m))
    options = re.compile(r'{([^}]+):([^}]+)}')

    def mysub(match):
      options, thing = match.group(1, 2)
      ret = ''
      if 'a' in options:
        if ignore_missing and thing not in mapping:
          return match.group()
        else:
          word = formatz('{%s}' % thing, mapping)
          ret += '%s ' % add_article(word).split()[0]
      ret += '{%s}' % thing
      return ret

    x = options.sub(mysub, x)
    return formatz(x, mapping)

  return tree.map_structure(formatfn, obj)

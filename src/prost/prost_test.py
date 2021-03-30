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
""" Test PROST Dataset """

import json
from pathlib import Path

import tree
import yaml
from absl import logging
from absl.testing import absltest, parameterized

EXPECTED_STRUCTURE = {
  'name': '',
  'context': '',
  'question': '',
  'ex_question': '',
  'expect_fn': '',
  'choices': {
    'A': '',
    'B': '',
    'C': '',
    'D': ''}}


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


def load_configs(scenario_yml_paths):
  configs = [load_and_check_config_yml(p) for p in scenario_yml_paths]
  configs = [c for cg in configs for c in cg]
  return configs


class TestTemplates(parameterized.TestCase):
  def setUp(self):
    self.scenario_yml_files = list(Path('data/prost').glob('*.yml'))
    self.lexicons_yml_files = list(Path('data/prost/globals').glob('*.yml'))

  def test_valid_yml_files(self):
    for path in self.scenario_yml_files + self.lexicons_yml_files:
      try:
        with path.open('r') as f:
          loaded = yaml.safe_load(f)
      except:
        self.fail(f'Invalid YML file: {path}')

  def test_unique_names(self):
    configs = load_configs(self.scenario_yml_files)
    names = [c.get('name', None) for c in configs]

    # check all have names
    all_have_names = all([c != None] for c in names)
    self.assertEqual(all_have_names, True)

    # check unique names
    name_set = set(names)
    self.assertEqual(len(names), len(name_set))

  def test_structures(self):
    """ Check templates have consistent structure """
    configs = load_configs(self.scenario_yml_files)
    for config in configs:
      if 'label' in config:
        del config['label']
      tree.assert_same_structure(config, EXPECTED_STRUCTURE)

  def test_unique_templates(self):
    """ Dataset components (context,question,choices) should be unique."""
    # we don't care about names, we already check if those are unique
    configs = load_configs(self.scenario_yml_files)
    names = [c.pop('name') for c in configs]
    configs = [json.dumps(c, sort_keys=True) for c in configs]
    configs_set = set(configs)
    self.assertCountEqual(configs, configs_set, msg='Found duplicate Templates')

  def test_spacing(self):
    """ Check for double spacing 
      This is usally a result of space before a line break in yml templates
    """
    configs = load_configs(self.scenario_yml_files)

    def _test_leaf_spacing(x):
      if isinstance(x, str):
        self.assertNotIn('  ', x, msg='Found incorrect spacing')

    tree.map_structure(_test_leaf_spacing, configs)

  def test_consistent(self):
    """ Dataset is sorted by context, creation should be deterministic. 
    
    Tempdir doesn't work bc HF Datasets uses a full path to create the filename
    On Linux the limits are 255/4096 for name and path respectively, so 
    a long path breaks if you try to use it as a filename. 

    Here we just are manually calling and checking the split generators.
    
    This test still might
    fail depending on your bazel installation (where the cache is).
    """

    # self.create_tempdir('hf_cache')
    # ds_1 = datasets.load_dataset('src/prost', cache_dir=ds_dir,
    #                              download_mode='force_redownload')['test']
    # ds_2 = datasets.load_dataset('src/prost', cache_dir=ds_dir,
    #                              download_mode='force_redownload')['test']

    # Workaround
    import prost

    ds_1 = prost.Prost._generate_examples(None, self.scenario_yml_files,
                                          self.lexicons_yml_files[0])
    ds_2 = prost.Prost._generate_examples(None, self.scenario_yml_files,
                                          self.lexicons_yml_files[0])

    # check deterministic
    ds_1, ds_2 = list(ds_1), list(ds_2)
    self.assertEqual(ds_1, ds_2, 'Dataset creation is not deterministic.')


if __name__ == '__main__':
  absltest.main()

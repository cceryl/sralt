import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from scripts import comparison_report, defense_report, reproduce_sal
from sralt.utils.system import get_object_repr


class DefenseSettingsTests(unittest.TestCase):
    def test_pipeline_defaults_formal_defaults_and_overrides_agree(self):
        expected = {'ciao': .05, 'epinions': .01, 'flickr': .1, 'yelp': .01}
        for dataset, weight in expected.items():
            with self.subTest(dataset=dataset), initialize_config_module(config_module='sralt.configs', version_base=None):
                config = compose(config_name='config', overrides=[
                    f'target/data={dataset}', 'target/model=design_sal', 'shadow/model=smia', 'attack/model=mlp'
                ])
                self.assertEqual(config.target.model.attribute.defense.loss_weight, weight)
                explicit = compose(config_name='config', overrides=[
                    f'target/data={dataset}', 'target/model=design_sal', 'shadow/model=smia', 'attack/model=mlp',
                    'target.model.attribute.defense.loss_weight=.2'
                ])
                self.assertEqual(explicit.target.model.attribute.defense.loss_weight, .2)
        cases = list(reproduce_sal.cases(reproduce_sal.parse_args(['--phase', 'formal', '--seeds', '42'])))
        self.assertEqual({dataset: weight for dataset, _, weight in cases if weight is not None}, expected)
        self.assertEqual([w for _, _, w in reproduce_sal.cases(reproduce_sal.parse_args(['--phase', 'minimal']))],
                         [None, .01])
        overridden = list(reproduce_sal.cases(reproduce_sal.parse_args([
            '--phase', 'formal', '--datasets', 'ciao', '--seeds', '42', '--weights', '.3'
        ])))
        self.assertEqual(overridden, [('ciao', 42, None), ('ciao', 42, .3)])

    def test_literal_model_config_preserves_previous_cache_representation(self):
        # Rebuild the original YAML without today's defaults selector, then apply
        # the explicit weight used in the completed formal experiments.
        directory = Path(comparison_report.ROOT / 'sralt/configs/target/model')
        original = OmegaConf.merge(OmegaConf.load(directory / 'design.yaml'),
                                   OmegaConf.load(directory / 'design_sal.yaml'))
        del original.defaults
        for dataset, weight in [('ciao', .05), ('flickr', .1), ('epinions', .01), ('yelp', .01)]:
            original.attribute.defense.loss_weight = weight
            actual = comparison_report.compose_case(dataset, 'design_sal', 'smia', 'mlp', 42).target.model
            self.assertEqual(get_object_repr(actual), get_object_repr(original))

    def test_shared_comparison_report_uses_dataset_defaults(self):
        args = comparison_report.parse_args([
            '--datasets', 'ciao', 'flickr', '--targets', 'design_sal', '--pairs', 'smia:mlp',
            '--seeds', '42', '--dry-run'
        ])
        payload = comparison_report.collect(args)
        self.assertEqual([row['defense'] for row in payload['rows']], ['SAL λ=0.05', 'SAL λ=0.1'])


class DefenseReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.args = defense_report.parse_args([
            '--datasets', 'ciao', '--targets', 'design', 'gdmsr', '--seeds', '42', '43',
            '--metric-cache', str(self.directory / 'metrics')
        ])
        self.plan = next(defense_report.plans(self.args))
        self.counter = 0

    def store(self, plan=None, seed=42, variant='baseline', source='v1', stage=None, base=.8, date='2026-10-23T01:00:00Z'):
        plan = plan or self.plan
        config = copy.deepcopy(plan[f'{variant}_config'])
        config.seed = seed
        for current in ([stage] if stage else defense_report.STAGES):
            identity = defense_report.expected_identity(config, current, self.args.namespace)
            identity['sources'] = source
            metrics = ({'all': {'Recall@10': .3 if variant == 'baseline' else .27, 'NDCG@10': .2}}
                       if current == 'target' else {group: {'AUROC': base, 'TPR@FPR=1%': .1}
                                                   for group in ['all', 'head', 'tail']})
            self.counter += 1
            path = self.args.metric_cache / current / f'{self.counter}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(identity=identity, created_at=date, metrics=metrics)))

    def populate(self):
        for seed in (42, 43):
            self.store(seed=seed, variant='baseline', base=.8 if seed == 42 else .7)
            self.store(seed=seed, variant='defense', base=.6 if seed == 42 else .63)
        placeholder = list(defense_report.plans(self.args))[1]
        for seed in (42, 43):
            self.store(plan=placeholder, seed=seed, base=.72)

    def test_paired_metrics_and_missing_adapter_results(self):
        self.populate()
        with patch.object(defense_report, 'evaluate_existing', side_effect=AssertionError('Unexpected evaluation')):
            payload = defense_report.collect(self.args)
        implemented, placeholder = payload['comparisons']
        metric = next(row for row in implemented['summary'] if row['metric'] == 'AUROC' and row['group'] == 'all')
        self.assertAlmostEqual(metric['relative_drop']['mean'], 17.5)
        self.assertEqual(metric['relative_drop']['n'], 2)
        self.assertEqual(metric['decreased'], 2)
        self.assertEqual(implemented['setting_status'], 'confirmed')
        self.assertTrue(placeholder['implemented'])
        self.assertEqual(placeholder['setting_status'], 'provisional')
        self.assertTrue(all(row['baseline']['state'] == 'available' and row['defense']['state'] == 'missing'
                            for row in placeholder['cases']))
        self.assertFalse(placeholder['summary'])

    def test_sources_are_not_mixed_and_coverage_precedes_recency(self):
        self.populate()
        self.store(seed=42, source='v2', date='2026-10-24', base=.9)
        self.store(seed=42, variant='defense', source='v2', date='2026-10-24', base=.1)
        result = defense_report.collect(self.args)['comparisons'][0]
        self.assertEqual(result['available_cohorts'], 2)
        self.assertEqual(result['provenance']['sources'], 'v1')
        self.args.source = 'v2'
        result = defense_report.collect(self.args)['comparisons'][0]
        self.assertEqual(result['cases'][1]['baseline']['state'], 'missing')
        self.assertEqual(result['summary'][0]['relative_drop']['n'], 1)

    def test_config_and_namespace_mismatch_are_missing_and_partial_is_explicit(self):
        self.store(variant='defense', stage='attack')
        result = defense_report.collect(self.args)['comparisons'][0]
        self.assertEqual(result['cases'][0]['defense']['state'], 'partial')
        self.assertFalse(result['summary'])
        self.args.namespace = 'another-experiment'
        result = defense_report.collect(self.args)['comparisons'][0]
        self.assertEqual(result['cases'][0]['defense']['state'], 'missing')
        self.args.namespace = 'default'
        self.args.weights = [.2]
        result = defense_report.collect(self.args)['comparisons'][0]
        self.assertEqual(result['cases'][0]['defense']['state'], 'missing')
        self.assertEqual(result['setting_status'], 'provisional')

    def test_opt_in_evaluation_never_trains_and_dry_run_never_evaluates(self):
        self.args.evaluate_missing = True
        self.args.dry_run = True
        with patch.object(defense_report, 'evaluate_existing', side_effect=AssertionError('Unexpected evaluation')):
            defense_report.collect(self.args)
        with patch.object(comparison_report, 'source_digest', return_value='new-source'), \
                patch.object(comparison_report, 'evaluate_missing', side_effect=comparison_report.MissingTrainingCache('missing')) as evaluator:
            defense_report.evaluate_existing(self.plan, self.args)
        self.assertEqual(evaluator.call_count, 4)
        self.assertTrue(all(call.kwargs['allow_training'] is False for call in evaluator.call_args_list))

    def test_html_and_json_export_preserve_labels_as_data(self):
        self.populate()
        payload = defense_report.collect(self.args)
        payload['comparisons'][0]['dataset'] = '</script><script>alert(1)</script>'
        output = self.directory / 'defense.html'
        defense_report.write_report(payload, output)
        text = output.read_text()
        self.assertNotIn(payload['comparisons'][0]['dataset'], text)
        self.assertEqual(json.loads(output.with_suffix('.json').read_text()), payload)
        self.assertNotIn('<script src=', text)

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for actual client execution')
    def test_client_pairing_filters_missing_results_chart_and_csv(self):
        self.populate()
        payload = defense_report.collect(self.args)
        harness = r'''
const assert=require('node:assert/strict');
class Element{constructor(){this.children=[];this.checked=false;this.value='';this.innerHTML='';this.textContent='';}append(...children){this.children.push(...children)}click(){}}
const elements=new Map();global.document={getElementById:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id)},createElement:()=>new Element(),createTextNode:text=>({textContent:text})};
document.getElementById('aggregate').checked=true;document.getElementById('metric').value='AUROC';document.getElementById('group').value='all';
document.getElementById('data').textContent=PAYLOAD;
'''.replace('PAYLOAD', json.dumps(json.dumps(payload)))
        script = defense_report.HTML.split('<script>')[1].split('</script>')[0]
        assertions = r'''
assert.equal(currentTable.rows.length,2);
assert.match(currentTable.rows[0][8],/17.5000 ± 10.6066/);
assert.match(currentTable.rows[1][5],/配对 0\/2/);
assert.equal(currentTable.rows[1][4],'待验证');
assert.match(currentTable.rows[1][6],/0.7200/);
assert.match(currentTable.rows[1][7],/n=0/);
assert.match($('chart').innerHTML,/<svg/);
choices.seed.delete('43');render();assert.match(currentTable.rows[0][8],/25.0000 \(n=1\)/);
$('aggregate').checked=false;render();assert.equal(currentTable.rows.length,2);
choices.seed.add('43');render();assert.equal(currentTable.rows.length,4);
choices.target.delete('gdmsr');render();assert.equal(currentTable.rows.length,2);
const zero=structuredClone(report.comparisons[0].cases[0]);zero.baseline.results.attack.all.AUROC=0;
assert.equal(metricStats([zero],'attack','all','AUROC').drop.n,0);
let exported;download=(name,content,type)=>exported={name,content,type};$('csv').onclick();assert.match(exported.content,/AUROC/);
$('json').onclick();assert.equal(JSON.parse(exported.content).comparisons.length,2);
'''
        path = self.directory / 'client.js'
        path.write_text(harness + script + assertions)
        result = subprocess.run(['node', str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()

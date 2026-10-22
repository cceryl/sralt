import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import comparison_report as report


class ComparisonReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cache = patch.object(report, "CACHE_DIR", self.directory / "metrics")
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.environment = patch.dict(os.environ, {"SRALT_CACHE_VERSION": "report-test", "SRALT_CACHE_REFRESH": "0"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.config = report.compose_case("ciao", "lightgcn", "miner", "miner", 42)

    def args(self, *extra):
        return report.parse_args(["--datasets", "ciao", "--targets", "lightgcn", "--pairs", "miner:miner",
                                  "--seeds", "42", "--no-target-metrics", "--no-shadow-metrics", *extra])

    def test_default_targets_include_lightgcn_and_cli_can_override(self):
        self.assertEqual(report.parse_args([]).targets, ["design", "gdmsr", "egcl", "lightgcn"])
        self.assertEqual(report.parse_args(["--targets", "design"]).targets, ["design"])
        self.assertEqual(report.compose_case("ciao", "lightgcn", "csa", "mlp", 42).target.model.name, "LightGCN")

    def test_dry_run_never_evaluates(self):
        for args in [self.args("--dry-run"), self.args("--allow-training", "--dry-run")]:
            with patch.object(report, "evaluate_missing", side_effect=AssertionError("Unexpected evaluation")):
                payload = report.collect(args)
            self.assertEqual(payload["rows"][0]["status"], {"attack": "missing"})
            self.assertIsNone(payload["rows"][0]["results"]["attack"])
        self.assertFalse(report.CACHE_DIR.exists())

    def test_default_evaluates_existing_models_and_marks_missing_training_cache(self):
        with patch.object(report, "evaluate_missing", side_effect=report.MissingTrainingCache("shadow/MINER")) as evaluator:
            payload = report.collect(self.args())
        self.assertFalse(evaluator.call_args.kwargs["allow_training"])
        self.assertEqual(payload["rows"][0]["status"], {"attack": "missing"})
        self.assertEqual(payload["rows"][0]["missing_reason"], "shadow/MINER")
        self.assertFalse(report.CACHE_DIR.exists())

    def test_empty_model_family_skips_upstream_loading(self):
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.directory)
        identities = {"attack": {}}
        with self.assertRaisesRegex(report.MissingTrainingCache, "target/LightGCN"):
            report.evaluate_missing(self.config, identities, {"attack": None})

    def test_metrics_hit_bypasses_evaluation_even_when_training_allowed(self):
        identity = report.cache_identity(self.config, "attack", report.source_digest())
        values = [[0.6, 0.5, float("nan"), 0.4, 0.3]] * 3
        report.save_metrics(report.cache_path(identity), identity, values, self.config.attack)
        for args in [self.args(), self.args("--allow-training")]:
            with patch.object(report, "evaluate_missing", side_effect=AssertionError("Unexpected evaluation")):
                payload = report.collect(args)
            row = payload["rows"][0]
            self.assertEqual(row["status"]["attack"], "cached")
            self.assertEqual(row["results"]["attack"]["tail"]["AUROC"], 0.6)
            self.assertIsNone(row["results"]["attack"]["head"]["TPR@FPR=1%"])
            self.assertEqual(payload["attack_metrics"], ["AUROC", "F1Score", "TPR@FPR=1%"])

    def test_stage_keys_share_only_their_actual_dependencies(self):
        other_attack = report.compose_case("ciao", "lightgcn", "miner", "enminer", 42)
        other_shadow = report.compose_case("ciao", "lightgcn", "csa", "mlp", 42)
        key = lambda cfg, stage: report.cache_path(report.cache_identity(cfg, stage, "sources"))
        self.assertEqual(key(self.config, "target"), key(other_shadow, "target"))
        self.assertEqual(key(self.config, "shadow"), key(other_attack, "shadow"))
        self.assertNotEqual(key(self.config, "attack"), key(other_attack, "attack"))
        changed = copy.deepcopy(self.config)
        changed.seed = 43
        self.assertNotEqual(key(self.config, "attack"), key(changed, "attack"))
        changed = copy.deepcopy(self.config)
        changed.target.test.evaluation.test_batch_size += 1
        self.assertEqual(key(self.config, "attack"), key(changed, "attack"))
        self.assertNotEqual(key(self.config, "target"), key(changed, "target"))
        with patch.dict(os.environ, {"SRALT_CACHE_VERSION": "new-namespace"}):
            changed_key = key(self.config, "attack")
        self.assertNotEqual(key(self.config, "attack"), changed_key)
        self.assertNotEqual(report.cache_path(report.cache_identity(self.config, "attack", "v1")),
                            report.cache_path(report.cache_identity(self.config, "attack", "v2")))

    def test_explicit_training_and_refresh_reach_evaluation(self):
        def evaluate(config, identities, records, allow_training=False):
            for stage, identity in identities.items():
                if records[stage] is None:
                    values = [[0.5] * 5] * 3 if stage == "attack" else [0.5, 0.4]
                    records[stage] = report.save_metrics(report.cache_path(identity), identity, values, config[stage])

        args = self.args("--allow-training", "--target-metrics", "--shadow-metrics")
        with patch.object(report, "evaluate_missing", side_effect=evaluate) as evaluator:
            payload = report.collect(args)
            self.assertEqual(evaluator.call_count, 1)
            self.assertEqual(set(payload["rows"][0]["status"].values()), {"computed"})
            report.collect(args)
            self.assertEqual(evaluator.call_count, 1)
            args.refresh_metrics = True
            report.collect(args)
            self.assertEqual(evaluator.call_count, 2)
            args.allow_training = False
            report.collect(args)
            self.assertEqual(evaluator.call_count, 3)
            self.assertFalse(evaluator.call_args.kwargs["allow_training"])

    def test_html_keeps_embedded_labels_as_data_and_has_no_external_assets(self):
        payload = report.collect(self.args("--dry-run"))
        payload["rows"][0]["shadow"] = '</script><script>alert("injected")</script>'
        output = self.directory / "report.html"
        report.write_report(payload, output)
        html = output.read_text()
        self.assertNotIn(payload["rows"][0]["shadow"], html)
        embedded = html.split('<script type="application/json" id="report-data">')[1].split('</script>')[0]
        self.assertEqual(json.loads(embedded), payload)
        self.assertNotIn('<script src=', html)
        self.assertNotIn('<link ', html)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the inline JavaScript check")
    def test_client_aggregation_filters_baseline_and_exports(self):
        """Exercise the actual inline client with a minimal DOM, without network assets."""
        payload = report.collect(self.args("--dry-run"))
        first = payload["rows"][0]
        first["results"]["attack"] = {group: dict(AUROC=0.6, F1Score=0.4, **{"TPR@FPR=1%": 0.1})
                                      for group in report.GROUPS}
        first["status"]["attack"] = "cached"
        second = copy.deepcopy(first)
        second["seed"] = 43
        second["results"]["attack"]["all"]["AUROC"] = 0.8
        missing = copy.deepcopy(first)
        missing["seed"] = 44
        missing["results"]["attack"] = None
        missing["status"]["attack"] = "missing"
        payload["rows"] = [first, second, missing]
        script = report.HTML.split('<script>')[1].split('</script>')[0]
        harness = r'''
const assert=require('node:assert/strict');
class Element {
  constructor(){this.children=[];this.dataset={};this.checked=false;this.value='';this.innerHTML='';this.textContent='';}
  append(...children){this.children.push(...children);if(!this.value&&children[0]?.value)this.value=children[0].value;}
  querySelectorAll(selector){const descendants=this.children.flatMap(c=>c instanceof Element?[c,...c.querySelectorAll(selector)]:[]);return selector==='input'?descendants.filter(c=>c.type==='checkbox'):[];}
  click(){}
}
const elements=new Map();
global.document={getElementById:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id)},createElement:()=>new Element(),createTextNode:text=>({textContent:text})};
document.getElementById('aggregate').checked=true;
document.getElementById('chart-group').value='all';
document.getElementById('report-data').textContent=PAYLOAD;
'''.replace("PAYLOAD", json.dumps(json.dumps(payload)))
        assertions = r'''
assert.equal(currentTable.rows.length,1);
assert.equal(currentTable.rows[0][5],'2/3');
assert.match(currentTable.rows[0][6],/0.7000 ± 0.1414 \(n=2\)/);
assert.equal(stats([null,undefined,NaN]).n,0);
assert.match($('charts').children.at(-1).innerHTML,/<svg/);
$('hide-empty').checked=true;render();
assert.equal(currentTable.rows.length,1);
assert.equal(currentTable.rows[0][5],'2/3');
choices.seed.delete('43');render();
assert.match(currentTable.rows[0][6],/0.6000 \(n=1\)/);
$('aggregate').checked=false;render();assert.equal(currentTable.rows.length,1);
$('hide-empty').checked=false;render();assert.equal(currentTable.rows.length,2);
choices.dataset.clear();render();assert.equal(currentTable.rows.length,0);
$('reset').onclick();assert.equal(currentTable.rows.length,3);
let exported;download=(name,content,type)=>exported={name,content,type};
$('csv').onclick();assert.match(exported.content,/AUROC/);assert.match(exported.content,/ciao/);
$('json').onclick();assert.equal(JSON.parse(exported.content).rows.length,3);
const absent=structuredClone(report.rows[0]);absent.shadow='miner';absent.attack='enminer';absent.results.attack=null;
report.rows.push(absent);choices.shadow.add('miner');choices.attack.add('enminer');
$('aggregate').checked=true;$('hide-empty').checked=false;render();assert.equal(currentTable.rows.length,2);
$('hide-empty').checked=true;render();assert.equal(currentTable.rows.length,1);
assert.equal(currentTable.rows[0][5],'2/3');
$('csv').onclick();assert.doesNotMatch(exported.content,/enminer/);
$('json').onclick();assert.equal(JSON.parse(exported.content).rows.length,4);

// A baseline compares the displayed means, independently within each facet.
function fixture(dataset,target,shadow,seed,auroc){return {dataset,target,defense:'None',shadow,attack:'mlp',seed,
  status:{attack:'cached'},results:{attack:auroc===null?null:Object.fromEntries(['all','head','tail'].map(group=>
    [group,{AUROC:auroc,F1Score:shadow==='smia'?0:0.5,'TPR@FPR=1%':0.1}]))}};}
report.rows=[fixture('ciao','design','smia',42,0.4),fixture('ciao','design','smia',43,0.6),
  fixture('ciao','design','csa',42,0.6),fixture('ciao','design','csa',43,0.8),
  fixture('flickr','design','smia',42,0.2),fixture('flickr','design','csa',42,0.1),
  fixture('ciao','lightgcn','smia',42,0.8),fixture('ciao','lightgcn','csa',42,0.4),
  fixture('ciao','egcl','csa',42,0.8)];
for(const key of Object.keys(dimensions))choices[key]=new Set(report.rows.map(row=>String(row[key])));
$('aggregate').checked=true;$('hide-empty').checked=false;$('baseline').value=JSON.stringify(['smia','mlp']);
$('baseline').onchange();
let ciao=aggregate(selectedRows()).find(row=>row.dataset==='ciao'&&row.target==='design'&&row.shadow==='csa');
assert.equal(relativeChange(ciao,'all','AUROC').text,'+40.00%');
assert.equal(relativeChange(ciao,'all','TPR@FPR=1%').text,'0.00%');
assert.equal(relativeChange(ciao,'all','F1Score').value,null);
assert.match(relativeChange(ciao,'all','F1Score').reason,/baseline 值为 0/);
assert.match($('attack-table').innerHTML,/Δ \+40.00%/);
assert.equal($('baseline-note').hidden,false);
const baseline=aggregate(selectedRows()).find(row=>row.target==='design'&&row.shadow==='smia');
assert.equal(relativeChange(baseline,'all','AUROC').text,'baseline');
for(const [dataset,target] of [['flickr','design'],['ciao','lightgcn']]){
  const row=aggregate(selectedRows()).find(row=>row.dataset===dataset&&row.target===target&&row.shadow==='csa');
  assert.equal(relativeChange(row,'all','AUROC').text,'-50.00%');
}
const noBaseline=aggregate(selectedRows()).find(row=>row.target==='egcl');
assert.equal(relativeChange(noBaseline,'all','AUROC').value,null);
assert.match(relativeChange(noBaseline,'all','AUROC').reason,/没有 baseline 指标记录/);
const csvRow=currentTable.rows.find(row=>row[0]==='ciao'&&row[1]==='design'&&row[3]==='csa');
assert.equal(csvRow[currentTable.headers.indexOf('Baseline')],'smia / mlp');
assert.equal(csvRow[currentTable.headers.indexOf('全部 · AUROC · 相对 baseline (%)')],'+40.00%');
$('csv').onclick();assert.match(exported.content,/\+40.00%/);
choices.shadow.delete('smia');render();
assert.equal(relativeChange(ciao,'all','AUROC').text,'+40.00%');
assert.equal(currentTable.rows.some(row=>row[3]==='smia'),false);
choices.seed.delete('43');render();
ciao=aggregate(selectedRows()).find(row=>row.target==='design'&&row.dataset==='ciao');
assert.equal(relativeChange(ciao,'all','AUROC').text,'+50.00%');
choices.seed.add('43');$('aggregate').checked=false;render();
const ciao43=aggregate(selectedRows()).find(row=>row.target==='design'&&row.dataset==='ciao'&&row.seed===43);
assert.equal(relativeChange(ciao43,'all','AUROC').text,'+33.33%');
report.rows[0].results.attack=null;render();
const ciao42=aggregate(selectedRows()).find(row=>row.target==='design'&&row.dataset==='ciao'&&row.seed===42);
assert.equal(relativeChange(ciao42,'all','AUROC').value,null);
assert.equal(relativeChange(ciao43,'all','AUROC').text,'+33.33%');
ciao42.runs[0].results.attack=null;
assert.match(relativeChange(ciao42,'all','AUROC').reason,/当前组合缺少/);
$('baseline').value='';$('baseline').onchange();
assert.equal(relativeChange(ciao43,'all','AUROC'),null);
assert.equal(currentTable.headers.includes('Baseline'),false);

// SAL variants aggregate separately, and relative losses match seeds before averaging.
const sal42=fixture('ciao','design','smia',42,0.3);sal42.defense='SAL λ=0.01';
const sal43=fixture('ciao','design','smia',43,0.48);sal43.defense='SAL λ=0.01';
const salOther=fixture('ciao','design','smia',42,0.2);salOther.defense='SAL λ=0.02';
report.rows=[fixture('ciao','design','smia',42,0.4),fixture('ciao','design','smia',43,0.6),sal42,sal43,salOther];
for(const key of Object.keys(dimensions))choices[key]=new Set(report.rows.map(row=>String(row[key])));
$('aggregate').checked=true;render();
assert.equal(currentTable.rows.length,3);
assert.match($('defense-table').innerHTML,/22.5000 ± 3.5355/);
assert.match($('defense-table').innerHTML,/50.0000/);
choices.defense.delete('None');render();
assert.equal(currentTable.rows.length,2);
assert.match($('defense-table').innerHTML,/22.5000 ± 3.5355/);
$('aggregate').checked=false;render();
assert.equal(currentTable.rows.length,3);
assert.match($('defense-table').innerHTML,/25.0000/);
'''
        program = self.directory / "client-test.js"
        program.write_text(harness + script + assertions)
        result = subprocess.run(["node", str(program)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_small_three_stage_evaluation_and_cached_report(self):
        """Real training/test functions on tiny CPU graphs, never real datasets."""
        import torch
        from omegaconf import OmegaConf
        from torch_geometric.utils import to_undirected
        from sralt.target.data import TargetData

        data = TargetData(name="synthetic", directory="unused")
        data.num_users, data.num_items = 12, 20
        users = torch.arange(data.num_users)
        data.social_edge_index = to_undirected(torch.cat([
            torch.stack([users, (users + 1) % data.num_users]),
            torch.stack([users, (users + 2) % data.num_users])], dim=1))
        items = data.num_users + (2 * users[:, None] + torch.arange(5)) % data.num_items
        data.behavioral_train_edge_index = torch.stack([users.repeat_interleave(3), items[:, :3].flatten()])
        data.behavioral_val_edge_index = torch.stack([users, items[:, 3]])
        data.behavioral_test_edge_index = torch.stack([users, items[:, 4]])
        config = copy.deepcopy(self.config)
        config.target.data.directory = str(report.ROOT / "data/ciao")
        config.device = OmegaConf.create({"_target_": "torch.device", "type": "cpu"})
        for stage in ["target", "shadow"]:
            cfg = config[stage]
            cfg.model.attribute.embed_dim = 4
            cfg.model.attribute.num_layers = 1
            cfg.model.train.num_epochs = 1
            cfg.model.train.train_batch_size = 16
            cfg.model.train.val_num_negatives = 4
            cfg.test.evaluation.test_num_negatives = 4
            for metric in [*cfg.model.train.metrics, *cfg.test.metrics]:
                metric.k = 2
        config.shadow.data.num_recommendations = 4
        config.shadow.data.shadow_social_ratio = 0.5
        config.attack.data.val_ratio = 0.25
        config.attack.tail_ratio = 0.5
        config.attack.model.train.num_epochs = 1
        config.attack.model.train.train_batch_size = 5
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.directory)
        with patch.object(report, "compose_case", return_value=config), patch("sralt.target.load_target_data", return_value=data):
            args = self.args("--allow-training", "--target-metrics", "--shadow-metrics")
            payload = report.collect(args)
            self.assertEqual(set(payload["rows"][0]["status"].values()), {"computed"})
            self.assertEqual(set(payload["rows"][0]["results"]["attack"]), set(report.GROUPS))
            with patch.object(report, "evaluate_missing", side_effect=AssertionError("Unexpected evaluation")):
                cached = report.collect(args)
            self.assertEqual(cached["rows"][0]["results"], payload["rows"][0]["results"])
            # Delete only metric caches. All three real model caches must now be
            # reused with training disabled, including interpolated MINER config.
            for path in report.CACHE_DIR.rglob("*.json"):
                path.unlink()
            args.allow_training = False
            from sralt.target.model.lightgcn import LightGCN
            from sralt.shadow.model.miner import MINER as ShadowMINER
            from sralt.attack.model.miner import MINER as AttackMINER
            with patch.object(LightGCN, "train_model", side_effect=AssertionError("Target retrained")), \
                 patch.object(ShadowMINER, "train_model", side_effect=AssertionError("Shadow retrained")), \
                 patch.object(AttackMINER, "train_model", side_effect=AssertionError("Attack retrained")):
                retested = report.collect(args)
                self.assertEqual(retested["rows"][0]["results"], payload["rows"][0]["results"])
                self.assertEqual(set(retested["rows"][0]["status"].values()), {"computed"})
                # --refresh-metrics also retests cached models without granting
                # training permission. The environment refresh cannot bypass
                # this report's training guard either.
                args.refresh_metrics = True
                with patch.dict(os.environ, {"SRALT_CACHE_REFRESH": "1"}):
                    refreshed = report.collect(args)
                self.assertEqual(refreshed["rows"][0]["results"], payload["rows"][0]["results"])
                args.refresh_metrics = False
                # Missing attack cache preserves already available target/shadow
                # metrics and records the reason without starting any training.
                for path in Path(".cache/attack/model/MINER").glob("*.pkl"):
                    path.unlink()
                for path in (report.CACHE_DIR / "attack").glob("*.json"):
                    path.unlink()
                missing = report.collect(args)["rows"][0]
                self.assertEqual(missing["status"], {"attack": "missing", "target": "cached", "shadow": "cached"})
                self.assertIn("attack/MINER", missing["missing_reason"])


if __name__ == "__main__":
    unittest.main()

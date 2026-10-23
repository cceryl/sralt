#!/usr/bin/env python3
"""Build an offline paired privacy/utility report from existing metric caches.

Defaults: four datasets, DESIGN/GDMSR/EGCL, SMIA+MLP, seeds42–46. All three targets implement SAL; GDMSR/EGCL weights remain provisional.
By default no models, data tensors or GPUs are loaded. --evaluate-missing tests
existing trained model caches, using comparison_report's loaders without training.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import itertools
import json
import logging
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import comparison_report as comparison

TARGETS = ['design', 'gdmsr', 'egcl']
VERIFIED_DATASETS = {'ciao', 'flickr'}
STAGES = ('target', 'attack')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=comparison.DATASETS)
    parser.add_argument('--targets', nargs='+', choices=TARGETS, default=TARGETS)
    parser.add_argument('--seeds', nargs='+', type=int, default=comparison.SEEDS)
    parser.add_argument('--weights', nargs='+', type=float, help='override dataset SAL defaults')
    parser.add_argument('--shadow', default='smia')
    parser.add_argument('--attack', default='mlp')
    parser.add_argument('--source', help='use one exact recorded source fingerprint')
    parser.add_argument('--namespace', default=os.environ.get('SRALT_CACHE_VERSION', 'default'))
    parser.add_argument('--metric-cache', type=Path, default=comparison.CACHE_DIR)
    parser.add_argument('--evaluate-missing', action='store_true', help='test trained caches only; never train')
    parser.add_argument('--dry-run', action='store_true', help='inspect plans/cache coverage without evaluating or writing')
    parser.add_argument('--output', type=Path, default=Path('runs/defense.html'))
    args = parser.parse_args(argv)
    for key in ('datasets', 'targets', 'seeds'):
        setattr(args, key, list(dict.fromkeys(getattr(args, key))))
    if args.weights is not None:
        args.weights = list(dict.fromkeys(args.weights))
    if args.attack in {'miner', 'enminer'} and args.shadow != 'miner':
        parser.error(f'{args.attack} requires --shadow miner')
    if args.evaluate_missing and args.namespace != os.environ.get('SRALT_CACHE_VERSION', 'default'):
        parser.error('--evaluate-missing requires SRALT_CACHE_VERSION to match --namespace')
    return args


def adapter_config(target):
    """Resolve an implemented adapter through its Hydra config."""
    name = f'{target}_sal'
    return name if (ROOT / 'sralt/configs/target/model' / f'{name}.yaml').is_file() else None


def plans(args):
    for dataset, target in itertools.product(args.datasets, args.targets):
        adapter = adapter_config(target)
        base = comparison.compose_case(dataset, target, args.shadow, args.attack, args.seeds[0])
        defended = (comparison.compose_case(dataset, adapter, args.shadow, args.attack, args.seeds[0])
                    if adapter else None)
        weights = args.weights or ([float(defended.target.model.attribute.defense.loss_weight)] if defended else [None])
        for weight in weights:
            config = copy.deepcopy(defended)
            if config is not None:
                config.target.model.attribute.defense.loss_weight = weight
            yield dict(dataset=dataset, target=target, adapter=adapter, weight=weight,
                       baseline_config=base, defense_config=config,
                       setting_status=('not_implemented' if adapter is None else 'confirmed'
                                       if target == 'design' and dataset in VERIFIED_DATASETS
                                       and weight == float(defended.target.model.attribute.defense.loss_weight)
                                       else 'provisional'))


def expected_identity(config, stage, namespace):
    """Use the shared identity rules; composition/stat calls never load tensors."""
    identity = comparison.cache_identity(config, stage, sources='')
    identity['namespace'] = namespace
    # Match the JSON representation of tuples in dataset metadata.
    return json.loads(json.dumps(identity))


def load_records(directory):
    return [(path, json.loads(path.read_text()))
            for stage in STAGES for path in sorted((directory / stage).glob('*.json'))]


def cohort_key(identity):
    """Never pool different source revisions or different data file snapshots."""
    return json.dumps({key: identity[key] for key in
                       ('sources', 'namespace', 'deterministic', 'dataset_files')}, sort_keys=True)


def candidates(plan, args, records):
    wanted = {}
    for variant, config in (('baseline', plan['baseline_config']), ('defense', plan['defense_config'])):
        if config is None:
            continue
        for seed in args.seeds:
            config = copy.deepcopy(config)
            config.seed = seed
            for stage in STAGES:
                wanted[variant, seed, stage] = expected_identity(config, stage, args.namespace)
    cohorts = {}
    for path, record in records:
        identity = record['identity']
        if args.source and identity['sources'] != args.source:
            continue
        for key, expected in wanted.items():
            if all(identity.get(field) == expected[field] for field in
                   ('schema', 'stage', 'seed', 'namespace', 'deterministic', 'config', 'dataset_files')):
                bucket = cohorts.setdefault(cohort_key(identity), {})
                if key not in bucket or record['created_at'] > bucket[key][1]['created_at']:
                    bucket[key] = (path, record)
    return cohorts


def select_cohort(cohorts, seeds):
    """Prefer complete matched seed coverage, then coverage, then recency."""
    def quality(item):
        records = item[1]
        paired = sum(all((variant, seed, stage) in records for variant in ('baseline', 'defense')
                         for stage in STAGES) for seed in seeds)
        return paired, len(records), max(record['created_at'] for _, record in records.values())
    return max(cohorts.items(), key=quality) if cohorts else (None, {})


def evaluate_existing(plan, args):
    """Explicit opt-in only. Model-cache misses remain missing, never train."""
    sources = comparison.source_digest()
    for template in (plan['baseline_config'], plan['defense_config']):
        if template is None:
            continue
        for seed in args.seeds:
            config = copy.deepcopy(template)
            config.seed = seed
            identities = {stage: comparison.cache_identity(config, stage, sources) for stage in STAGES}
            records = {stage: json.loads(comparison.cache_path(identity).read_text())
                       if comparison.cache_path(identity).is_file() else None for stage, identity in identities.items()}
            if any(record is None for record in records.values()):
                try:
                    comparison.evaluate_missing(config, identities, records, allow_training=False)
                except comparison.MissingTrainingCache as error:
                    logging.info('Missing trained cache: %s', error)


def statistics_for(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return dict(mean=statistics.mean(values) if values else None,
                sd=statistics.stdev(values) if len(values) > 1 else None, n=len(values))


def summarize(cases):
    """Summarize only metric-level valid, nonzero, matched-seed pairs."""
    metrics = {}
    for case in cases:
        baseline, defended = case['baseline']['results'], case['defense']['results']
        for stage, groups in baseline.items():
            for group, values in groups.items():
                for metric, base in values.items():
                    value = defended.get(stage, {}).get(group, {}).get(metric)
                    if base is None or value is None or not math.isfinite(base) or not math.isfinite(value) or base == 0:
                        continue
                    row = metrics.setdefault((stage, group, metric), [])
                    row.append(dict(seed=case['seed'], baseline=base, defended=value,
                                    relative_drop_percent=100 * (base - value) / base))
    return [dict(stage=stage, group=group, metric=metric, pairs=pairs,
                 baseline=statistics_for([p['baseline'] for p in pairs]),
                 defense=statistics_for([p['defended'] for p in pairs]),
                 relative_drop=statistics_for([p['relative_drop_percent'] for p in pairs]),
                 decreased=sum(p['defended'] < p['baseline'] for p in pairs))
            for (stage, group, metric), pairs in metrics.items()]


def collect(args):
    planned = list(plans(args))
    if args.evaluate_missing and not args.dry_run:
        for plan in planned:
            evaluate_existing(plan, args)
    records = load_records(args.metric_cache)
    comparisons = []
    for plan in planned:
        cohorts = candidates(plan, args, records)
        cohort, selected = select_cohort(cohorts, args.seeds)
        cases = []
        for seed in args.seeds:
            case = dict(seed=seed)
            for variant in ('baseline', 'defense'):
                results, origins, times = {}, {}, {}
                for stage in STAGES:
                    item = selected.get((variant, seed, stage))
                    if item:
                        path, record = item
                        results[stage] = record['metrics']
                        origins[stage] = str(path)
                        times[stage] = record['created_at']
                state = ('not_implemented' if variant == 'defense' and plan['adapter'] is None else
                         'available' if len(results) == len(STAGES) else 'partial' if results else 'missing')
                case[variant] = dict(state=state, results=results, metric_files=origins, evaluated_at=times)
            cases.append(case)
        comparisons.append(dict(dataset=plan['dataset'], target=plan['target'], weight=plan['weight'],
                                shadow=args.shadow, attack=args.attack, setting_status=plan['setting_status'],
                                implemented=plan['adapter'] is not None,
                                provenance=json.loads(cohort) if cohort else None,
                                available_cohorts=len(cohorts), cases=cases, summary=summarize(cases)))
    return dict(generated_at=datetime.now().astimezone().isoformat(), namespace=args.namespace,
                metric_cache=str(args.metric_cache), evaluated_existing=args.evaluate_missing and not args.dry_run,
                comparisons=comparisons,
                note='同一行只使用一个源码/数据版本；优先匹配种子最多的版本。历史结果不等于当前源码重测。')


def write_report(payload, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    output.write_text(HTML.replace('__DATA__', data))


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SRALT · 防御报告</title><style>
body{font:14px system-ui,sans-serif;background:#f3f6fa;color:#193248;margin:0;padding:24px;max-width:1700px;margin:auto}h1{margin-bottom:6px}h2{font-size:18px}.card{background:white;padding:20px;border:1px solid #dbe4ed;border-radius:12px;margin:18px 0}.muted,small{color:#5b6b7d}p{line-height:1.7}.toolbar{display:flex;gap:16px;flex-wrap:wrap;align-items:center}select,button{padding:7px;border:1px solid #b8c7d6;border-radius:6px;background:white}fieldset{display:inline-block;vertical-align:top;border:1px solid #dbe4ed;margin:8px;padding:10px}fieldset label{margin-right:10px;white-space:nowrap}.wrap{overflow:auto;max-height:65vh}table{border-collapse:collapse;width:100%;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{padding:10px;border-bottom:1px solid #e4ebf1;text-align:right}th{position:sticky;top:0;background:#edf3fa}th:first-child,td:first-child{text-align:left}.missing{color:#9b5a2c}.positive{color:#087760}.negative{color:#b44b3c}svg{width:100%;max-width:1000px;min-width:500px}svg text{font-family:system-ui,sans-serif;fill:#29465b}@media(max-width:600px){body{padding:12px}}
</style><h1>SAL 防御报告</h1><p id="meta" class="muted"></p>
<section class="card"><div id="filters"></div><div class="toolbar"><label><input id="aggregate" type="checkbox" checked>合并种子</label><label>攻击指标 <select id="metric"><option>AUROC</option><option>F1Score</option><option>TPR@FPR=1%</option><option>Precision</option><option>Recall</option></select></label><label>分组 <select id="group"><option value="all">All</option><option value="head">Head</option><option value="tail">Tail</option></select></label><button id="csv">导出当前表格 CSV</button><button id="json">导出完整 JSON</button></div></section>
<section class="card"><p id="coverage"></p><p>相对下降 =（无防御 − SAL）/ 无防御 × 100%。攻击指标下降表示防御收益，推荐指标下降表示效用代价。按相同种子配对后汇总均值 ± 样本标准差；相对下降的 n 为有效配对数，零基线不参与百分比计算。无配对时仍显示已有单组指标，其 n 为该组记录数；单种子不估计标准差。</p><p>DESIGN/GDMSR/EGCL 均已接入 SAL；缺少实验指标时显示缺失。GDMSR/EGCL 的所有默认 λ、DESIGN 在 Epinions/Yelp 的默认 λ 均为待验证起点。固定0.5阈值的F1/Recall变化与低FPR保护可能不同。</p><p class="muted" id="provenance"></p></section>
<section class="card"><h2>隐私与推荐效用</h2><div class="wrap" id="table"></div></section>
<section class="card"><h2>隐私—效用取舍</h2><p class="muted">横轴：Recall@10相对损失；纵轴：当前攻击指标相对下降。每个点使用两项指标都有结果的同一批种子；左上方对应较低效用损失和较高防御收益。</p><div class="wrap" id="chart"></div></section>
<section class="card"><h2>版本与结果来源</h2><div class="wrap" id="sources"></div></section>
<script type="application/json" id="data">__DATA__</script><script>
'use strict';
const report=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const choices={},unique=a=>[...new Set(a)],label=r=>r.weight===null?'未接入':`λ=${r.weight}`;
for(const [key,title] of [['dataset','数据集'],['target','目标模型'],['weight','SAL权重']]){
 const values=unique(report.comparisons.map(r=>String(r[key]))),field=document.createElement('fieldset'),legend=document.createElement('legend');legend.textContent=title;field.append(legend);choices[key]=new Set(values);
 for(const value of values){const node=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=true;input.value=value;input.onchange=()=>{input.checked?choices[key].add(value):choices[key].delete(value);render()};node.append(input,document.createTextNode(value==='null'?'未接入':value));field.append(node);}$('filters').append(field);
}
const seeds=unique(report.comparisons.flatMap(r=>r.cases.map(c=>String(c.seed))));choices.seed=new Set(seeds);
const seedField=document.createElement('fieldset');const seedLegend=document.createElement('legend');seedLegend.textContent='种子';seedField.append(seedLegend);
for(const seed of seeds){const node=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=true;input.onchange=()=>{input.checked?choices.seed.add(seed):choices.seed.delete(seed);render()};node.append(input,document.createTextNode(seed));seedField.append(node);}$('filters').append(seedField);
function stats(values){const a=values.filter(Number.isFinite),n=a.length;if(!n)return {mean:null,sd:null,n};const mean=a.reduce((x,y)=>x+y,0)/n;return {mean,sd:n>1?Math.sqrt(a.reduce((s,x)=>s+(x-mean)**2,0)/(n-1)):null,n};}
const num=x=>Number.isFinite(x)?x.toFixed(4):'—',format=s=>s.n?`${num(s.mean)}${s.sd===null?'':` ± ${num(s.sd)}`} (n=${s.n})`:'— (n=0)';
const value=(c,variant,stage,group,metric)=>c[variant].results[stage]?.[group]?.[metric];
function metricStats(cases,stage,group,metric){
 const pairs=cases.filter(c=>Number.isFinite(value(c,'baseline',stage,group,metric))&&Number.isFinite(value(c,'defense',stage,group,metric))&&value(c,'baseline',stage,group,metric)!==0);
 const displayed=pairs.length?pairs:cases;
 return {baseline:stats(displayed.map(c=>value(c,'baseline',stage,group,metric))),defense:stats(displayed.map(c=>value(c,'defense',stage,group,metric))),drop:stats(pairs.map(c=>100*(value(c,'baseline',stage,group,metric)-value(c,'defense',stage,group,metric))/value(c,'baseline',stage,group,metric))),decreased:pairs.filter(c=>value(c,'defense',stage,group,metric)<value(c,'baseline',stage,group,metric)).length};
}
function selected(){return report.comparisons.filter(r=>['dataset','target','weight'].every(key=>choices[key].has(String(r[key])))).flatMap(r=>{const cases=r.cases.filter(c=>choices.seed.has(String(c.seed)));return $('aggregate').checked?[{...r,cases}]:cases.map(c=>({...r,cases:[c]}));});}
let currentTable={headers:[],rows:[]};
function render(){
 const rows=selected(),metric=$('metric').value,group=$('group').value;
 const columns=[['attack',group,metric],['target','all','Recall@10'],['target','all','NDCG@10']];
 currentTable={headers:['数据集','Target','λ','种子','设置','状态',...columns.flatMap(c=>[`${c[2]} 无防御`,`${c[2]} SAL`,`${c[2]} 相对下降(%)`,`${c[2]} 下降种子`])],rows:[]};
 let html='<table><thead><tr>'+currentTable.headers.map(h=>`<th>${escapeHTML(h)}</th>`).join('')+'</tr></thead><tbody>',points=[],complete=0;
 for(const row of rows){
  const paired=row.cases.filter(c=>c.baseline.state==='available'&&c.defense.state==='available').length;complete+=paired;
  const state=!row.implemented?'未实现':paired===row.cases.length&&paired?'完整':`配对 ${paired}/${row.cases.length}`;
  const cells=[row.dataset,row.target,label(row),row.cases.map(c=>c.seed).join(','),row.setting_status==='confirmed'?'已验证候选':row.setting_status==='not_implemented'?'未接入':'待验证',state];
  for(const [stage,g,m] of columns){const s=metricStats(row.cases,stage,g,m);cells.push(format(s.baseline),format(s.defense),format(s.drop),`${s.decreased}/${s.drop.n}`);}
  currentTable.rows.push(cells);html+='<tr>'+cells.map(c=>`<td>${escapeHTML(c)}</td>`).join('')+'</tr>';
  const matched=row.cases.filter(c=>columns.slice(0,2).every(([s,g,m])=>Number.isFinite(value(c,'baseline',s,g,m))&&value(c,'baseline',s,g,m)!==0&&Number.isFinite(value(c,'defense',s,g,m))));
  const privacy=metricStats(matched,'attack',group,metric),utility=metricStats(matched,'target','all','Recall@10');
  if(privacy.drop.n&&utility.drop.n)points.push({x:utility.drop.mean,y:privacy.drop.mean,label:`${row.dataset}/${row.target}/${label(row)}`,n:matched.length});
 }
 $('table').innerHTML=rows.length?html+'</tbody></table>':'没有符合筛选的结果。';
 $('coverage').textContent=`${rows.length} 个展示组合，${complete} 个完整种子配对。缺失记录保持为空，不按零值处理。`;
 let origins='<table><tr><th>数据集 / Target / λ</th><th>版本</th><th>种子</th><th>来源</th></tr>';
 for(const r of rows)origins+='<tr>'+[`${r.dataset}/${r.target}/${label(r)}`,r.provenance?.sources||'无记录',r.cases.map(c=>c.seed).join(','),unique(r.cases.flatMap(c=>['baseline','defense'].flatMap(v=>Object.values(c[v].metric_files)))).join(' · ')||'缺失'].map(v=>`<td>${escapeHTML(v)}</td>`).join('')+'</tr>';$('sources').innerHTML=origins+'</table>';
 renderChart(points);
}
function renderChart(points){
 if(!points.length){$('chart').textContent='尚无可配对的隐私与效用指标。';return;}
 const width=960,height=360,left=70,top=25,plotW=790,plotH=265;
 const xmin=Math.min(0,...points.map(p=>p.x)),xmax=Math.max(1,...points.map(p=>p.x)),ymin=Math.min(0,...points.map(p=>p.y)),ymax=Math.max(1,...points.map(p=>p.y));
 const x=v=>left+(v-xmin)/(xmax-xmin)*plotW,y=v=>top+plotH-(v-ymin)/(ymax-ymin)*plotH;
 let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="隐私与推荐效用取舍">`;
 for(let i=0;i<=5;i++){const a=xmin+(xmax-xmin)*i/5,b=ymin+(ymax-ymin)*i/5;svg+=`<path d="M ${x(a)} ${top} V ${top+plotH} M ${left} ${y(b)} H ${left+plotW}" stroke="#e2e9f1" fill="none"/><text x="${x(a)}" y="${top+plotH+22}" text-anchor="middle">${a.toFixed(1)}%</text><text x="${left-10}" y="${y(b)+4}" text-anchor="end">${b.toFixed(1)}%</text>`;}
 for(const p of points)svg+=`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="6" fill="#257d9b"><title>${escapeHTML(p.label+` · n=${p.n} · 效用损失${p.x.toFixed(2)}% · 防御收益${p.y.toFixed(2)}%`)}</title></circle><text x="${x(p.x)+9}" y="${y(p.y)-9}" font-size="12">${escapeHTML(p.label)}</text>`;
 $('chart').innerHTML=svg+`<text x="${left+plotW/2}" y="${height-12}" text-anchor="middle">Recall@10 相对损失 (%)</text></svg>`;
}
function download(name,content,type){const url=URL.createObjectURL(new Blob(['\ufeff',content],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('csv').onclick=()=>{const quote=v=>'"'+String(v).replace(/"/g,'""')+'"';download('defense.csv',[currentTable.headers,...currentTable.rows].map(row=>row.map(quote).join(',')).join('\r\n'),'text/csv;charset=utf-8');};
$('json').onclick=()=>download('defense.json',JSON.stringify(report,null,2),'application/json');
for(const id of ['aggregate','metric','group'])$(id).onchange=render;
$('meta').textContent=`生成 ${report.generated_at} · 命名空间 ${report.namespace} · ${report.evaluated_existing?'已按需测试现有模型，禁止训练':'只读指标缓存，未加载模型'}`;
$('provenance').textContent=report.note;render();
</script></html>'''


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    if args.evaluate_missing and args.source and args.source != comparison.source_digest():
        raise ValueError('--evaluate-missing can only evaluate the current source revision; omit --source or disable evaluation.')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # Shared evaluators write to the same explicitly selected metric directory.
    comparison.CACHE_DIR = args.metric_cache.resolve()
    payload = collect(args)
    if args.dry_run:
        for row in payload['comparisons']:
            count = sum(c['baseline']['state'] == c['defense']['state'] == 'available' for c in row['cases'])
            print(row['dataset'], row['target'], row['weight'], f'paired={count}/{len(row["cases"])}',
                  'implemented' if row['implemented'] else 'not_implemented')
    else:
        write_report(payload, args.output.resolve())
        logging.info('Defense report: %s', args.output.resolve())


if __name__ == '__main__':
    main()

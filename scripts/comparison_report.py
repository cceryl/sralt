#!/usr/bin/env python3
r"""Compare configured model combinations and write a self-contained offline HTML report.

Run from any directory with the project's Python environment (inside Apptainer):
    python scripts/comparison_report.py
    python scripts/comparison_report.py --datasets ciao --targets lightgcn \
        --pairs smia:mlp csa:mlp miner:miner miner:enminer --seeds 42 43
    python scripts/comparison_report.py --target-metrics --shadow-metrics
    python scripts/comparison_report.py --allow-training
    python scripts/comparison_report.py --pairs ensemble:ensemble ensemble:shadow_user_fusion

Metric records are stored in .cache/comparison_report. Model/data caches use
the existing loaders. A metrics hit bypasses all model imports and loading.
"""

from __future__ import annotations
from typing import Any
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import os
import math
import logging
import json
import itertools
import hashlib
import functools
import copy
import argparse

# Edit these defaults, then run without arguments. CLI options override each field.
DATASETS = ["ciao", "epinions", "flickr", "yelp"]
TARGET_MODELS = ["design", "gdmsr", "egcl"] # lightgcn
MODEL_PAIRS = [("ensemble", "ensemble"), ("ensemble", "shadow_user_fusion"),
               ("smia", "mlp"), ("csa", "mlp"), ("tail_proxy", "tail_proxy"),
               ("item_prefix", "item_prefix"), ("item_prefix_csa", "item_prefix_csa")]
SEEDS = [42, 43, 44, 45, 46]
INCLUDE_TARGET_METRICS = True
INCLUDE_SHADOW_METRICS = True
ALLOW_TRAINING = False  # Test existing models; leave missing training caches untrained.
OUTPUT = "runs/comparison.html"


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache/comparison_report"
CACHE_SCHEMA = 1
GROUPS = ("all", "head", "tail")
ATTACK_SUBGROUPS = {
    "head_head": "HH", "head_tail": "HT", "tail_tail": "TT",
    "isolated_tail": "isolated_Tail",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="target/data config names")
    parser.add_argument("--targets", nargs="+", default=TARGET_MODELS, help="target/model config names")
    parser.add_argument("--pairs", nargs="+", default=[f"{s}:{a}" for s, a in MODEL_PAIRS],
                        help="shadow:attack config pairs, e.g. csa:mlp miner:enminer")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--sal-weights", nargs="+", type=float,
                        help="override SAL weights for gdmsr_sal/egcl_sal")
    parser.add_argument("--target-metrics", action=argparse.BooleanOptionalAction, default=INCLUDE_TARGET_METRICS)
    parser.add_argument("--shadow-metrics", action=argparse.BooleanOptionalAction, default=INCLUDE_SHADOW_METRICS)
    parser.add_argument("--allow-training", action=argparse.BooleanOptionalAction, default=ALLOW_TRAINING,
                        help="train on model cache misses; existing models are tested even without this flag")
    parser.add_argument("--refresh-metrics", action="store_true", help="retest via existing loaders; do not force retraining")
    parser.add_argument("--dry-run", action="store_true", help="list combinations and metric cache hits without evaluation")
    parser.add_argument("--output", type=Path, default=Path(OUTPUT))
    args = parser.parse_args(argv)
    for pair in args.pairs:
        if len(pair.split(":")) != 2 or not all(pair.split(":")):
            parser.error(f"expected shadow:attack, got {pair!r}")
        shadow, attack = pair.split(":")
        if attack in {"miner", "enminer"} and shadow != "miner":
            parser.error(f"{attack} requires shadow/model=miner")
        if (shadow == "ensemble") != (attack in {"ensemble", "shadow_user_fusion"}):
            parser.error('ensemble settings require ensemble:ensemble or ensemble:shadow_user_fusion')
    for name in ("datasets", "targets", "pairs", "seeds"):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    return args


def source_digest() -> str:
    """Small source files only: never walk model caches or hash training tensors."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / "sralt").rglob("*")):
        if path.suffix in {".py", ".yaml"}:
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    for name in ('comparison_report.py', 'ensemble_pipeline.py', 'csa_itemprefix_ensemble_case.py',
                 'csa_itemprefix_ensemble_recipe.py', 'tail_proxy_case.py'):
        digest.update(name.encode())
        digest.update((ROOT / 'scripts' / name).read_bytes())
    return digest.hexdigest()


@functools.lru_cache(maxsize=None)
def compose_template(dataset: str, target: str, shadow: str, attack: str) -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "sralt/configs"), version_base=None):
        config = compose(config_name="config", overrides=[
            f"target/data={dataset}", f"target/model={target}", f"shadow/model={shadow}",
            f"attack/model={attack}"
        ])
    # The pipeline writer uses Hydra runtime interpolation, absent in compose().
    del config.writer
    return config


def compose_case(dataset: str, target: str, shadow: str, attack: str, seed: int,
                 sal_weight: float | None = None) -> Any:
    config = copy.deepcopy(compose_template(dataset, target, shadow, attack))
    config.seed = seed
    if sal_weight is not None:
        config.target.model.attribute.defense.loss_weight = sal_weight
    # Keep interpolation syntax intact: disk_cache includes repr(DictConfig).
    return config


def metric_name(metric: Any) -> str:
    name = str(metric["_target_"]).rsplit(".", 1)[-1].removeprefix("Binary").removeprefix("LinkPred")
    if name == "TPRatFPR":
        return f"TPR@FPR={metric['target_fpr']:.0%}"
    return f"{name}@{metric['k']}" if "k" in metric else name


def cache_identity(config: Any, stage: str, sources: str) -> dict:
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(config, resolve=True)
    assert isinstance(cfg, dict)
    directory = Path(cfg["target"]["data"]["directory"]).resolve()
    # Same dataset metadata convention as load_target_data; no dataset deserialization.
    files = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
             for p in sorted(directory.iterdir()) if p.is_file()]
    target = dict(cfg["target"])
    shadow = dict(cfg["shadow"])
    if stage != "target":
        target.pop("test")
    if stage == "attack":
        shadow.pop("test")
    stages = {"target": target}
    if stage in {"shadow", "attack"}:
        stages["shadow"] = shadow
    if stage == "attack":
        stages["attack"] = cfg["attack"]
        # Older records contain only All/Head/Tail; they cannot supply subgroups.
        stages["attack_report_groups"] = [*GROUPS, *ATTACK_SUBGROUPS]
    if config.attack.model.get('ensemble_condition') is not None and stage != 'target':
        from ensemble_pipeline import resolved_experiment_config
        stages['ensemble_experts'] = resolved_experiment_config(config)
    return {"schema": CACHE_SCHEMA, "stage": stage, "sources": sources, "dataset_files": files,
            "namespace": os.environ.get("SRALT_CACHE_VERSION", "default"),
            "seed": cfg["seed"], "deterministic": cfg["deterministic"], "config": stages}


def cache_path(identity: dict) -> Path:
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return CACHE_DIR / identity["stage"] / f"{key}.json"


def finite_metrics(names: list[str], values: list[float]) -> dict:
    return {name: float(value) if math.isfinite(value) else None
            for name, value in zip(names, values, strict=True)}


def save_metrics(path: Path, identity: dict, values: Any, config: Any,
                 subgroup_metrics: dict | None = None) -> dict:
    names = [metric_name(m) for m in config.test.metrics]
    metrics = ({group: finite_metrics(names, group_values)
                for group, group_values in zip(GROUPS, values, strict=True)}
               if identity["stage"] == "attack" else {"all": finite_metrics(names, values)})
    record = {"identity": identity, "created_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics}
    if identity["stage"] == "attack" and subgroup_metrics is not None:
        # grouped_attack_report uses numeric FPR labels; the report uses percentages.
        subgroup_names = [
            f"TPR@FPR={m['target_fpr']:g}" if str(m["_target_"]).rsplit(".", 1)[-1] == "TPRatFPR"
            else metric_name(m) for m in config.test.metrics
        ]
        counts = {}
        for group, source in ATTACK_SUBGROUPS.items():
            result = subgroup_metrics[source]
            metrics[group] = {}
            for name, key in zip(names, subgroup_names, strict=True):
                value = result[key]
                metrics[group][name] = float(value) if value is not None and math.isfinite(value) else None
            counts[group] = {key: result[key] for key in ("samples", "positives", "negatives")}
        record["group_counts"] = counts
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write a complete record atomically, including for interrupted multi-seed runs.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(record, handle, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)
    return record


class MissingTrainingCache(RuntimeError):
    """Expected model cache miss when the report is not allowed to train."""


def load_report_model(stage: str, loader: Any, allow_training: bool,
                      cache_directory: str | None = None, cache_ignore: list[str] | None = None,
                      **kwargs: Any) -> Any:
    """Reuse disk_cache's exact key and hooks without entering a training function."""
    if allow_training:
        return loader(**kwargs)

    from sralt.utils import disk_cache

    # wraps preserves the original signature/module/qualname, hence the cache key.
    # On a miss, the decorator calls this function instead of the training loader.
    @functools.wraps(loader)
    def unavailable(*args: Any, **bound_kwargs: Any) -> Any:
        label = kwargs['config'].name if 'config' in kwargs else kwargs['condition']
        raise MissingTrainingCache(f"{stage}/{label}: no matching trained model cache")

    reader = disk_cache(
        cache_dir=cache_directory or f".cache/{stage}/model/{{config.name}}",
        ignore=cache_ignore if cache_ignore is not None else ["device", "writer"],
        hook=lambda model, arguments, _: model.to(arguments["device"]),
        hit=f'Load trained {stage} model from cache at "{{__cache__.path}}".',
        miss=f'No trained {stage} model cache at "{{__cache__.path}}"; training disabled.'
    )(unavailable)
    # A refresh request must never turn this read into training or bypass an
    # available trained model. Metric refresh is handled separately by collect().
    refresh = os.environ.pop("SRALT_CACHE_REFRESH", None)
    try:
        return reader(**kwargs)
    finally:
        if refresh is not None:
            os.environ["SRALT_CACHE_REFRESH"] = refresh


def evaluate_missing(config: Any, identities: dict, records: dict, allow_training: bool = False) -> None:
    """Use public load/test functions unchanged; skip unrequested stage evaluations."""
    if not allow_training:
        order = ["target", "shadow", "attack"]
        first_pending = next(i for i, stage in enumerate(order) if stage in identities and records[stage] is None)
        # If an entire model family has no checkpoints, avoid loading upstream
        # tensors on every report run. Earlier pending metrics still get tested.
        for stage in order[:first_pending + 1]:
            name = config[stage].model.name
            if not any(Path(f".cache/{stage}/model/{name}").glob("*.pkl")):
                raise MissingTrainingCache(f"{stage}/{name}: no matching trained model cache")

    from hydra.utils import instantiate
    from sralt.target import load_target_data, load_target_model, test_target_model

    instantiate(config.filter_warnings)
    device = instantiate(config.device)
    common = dict(seed=config.seed, deterministic=config.deterministic)
    target_data = load_target_data(**common, config=config.target.data).to(device)
    target = load_report_model("target", load_target_model, allow_training,
                               **common, device=device, config=config.target.model, data=target_data, writer=None)
    if "target" in identities and records["target"] is None:
        values = test_target_model(**common, device=device, model=target, data=target_data, config=config.target.test)
        records["target"] = save_metrics(cache_path(identities["target"]), identities["target"], values, config.target)
    need_shadow = any(records[stage] is None for stage in ("shadow", "attack") if stage in identities)
    if not need_shadow:
        return

    from sralt.shadow import load_shadow_data, load_shadow_model, test_shadow_model

    shadow_data = load_shadow_data(**common, target_data=target_data, target_model=target, config=config.shadow.data).to(device)
    shadow = load_report_model("shadow", load_shadow_model, allow_training,
                               **common, device=device, config=config.shadow.model, data=shadow_data, writer=None)
    if "shadow" in identities and records["shadow"] is None:
        values = test_shadow_model(**common, device=device, model=shadow, data=shadow_data, config=config.shadow.test)
        records["shadow"] = save_metrics(cache_path(identities["shadow"]), identities["shadow"], values, config.shadow)
    if records["attack"] is not None:
        return

    from sralt.attack import load_attack_data, load_attack_model, test_attack_model

    attack_data = load_attack_data(**common, target_data=target_data, shadow_data=shadow_data,
                                   shadow_model=shadow, config=config.attack.data).to(device)
    attack = load_report_model("attack", load_attack_model, allow_training,
                               **common, device=device, config=config.attack.model, data=attack_data,
                               tail_ratio=config.attack.tail_ratio, writer=None)
    values = test_attack_model(**common, device=device, model=attack, data=attack_data,
                               config=config.attack.test, tail_ratio=config.attack.tail_ratio)
    records["attack"] = save_metrics(cache_path(identities["attack"]), identities["attack"], values, config.attack,
                                     subgroup_metrics=attack_data.subgroup_metrics)


def collect_ensemble(config: Any, identities: dict, records: dict, status: dict,
                     args: argparse.Namespace, dataset: str, target: str) -> str | None:
    """Two-expert attack rows; recommendation metrics retain their real owners."""
    from ensemble_pipeline import saved_result, execute
    condition = str(config.attack.model.ensemble_condition)
    missing = []
    if records['attack'] is None:
        refresh = args.refresh_metrics or os.environ.get('SRALT_CACHE_REFRESH') == '1'
        artifact = None if refresh else saved_result(config, condition, dataset, target)
        computed = False
        if artifact is None and not args.dry_run:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            try:
                with tempfile.TemporaryDirectory(prefix='ensemble-', dir=CACHE_DIR) as temporary:
                    artifact = execute(config, temporary, allow_training=args.allow_training,
                                       cache_loader=load_report_model)
                computed = True
            except MissingTrainingCache as error:
                missing.append(str(error))
        if artifact is not None:
            document, result = artifact['document'], artifact['result']
            group_map = dict(all='All', head='Head', tail='Tail', **ATTACK_SUBGROUPS)
            metrics, counts = {}, {}
            for group, native_group in group_map.items():
                values = result['groups'].get(native_group)
                if values is None:
                    continue  # Historical experiments did not export HH/HT.
                metrics[group] = {}
                for spec in config.attack.test.metrics:
                    name = metric_name(spec)
                    native_name = (f"TPR@FPR={spec.target_fpr:g}"
                                   if str(spec._target_).rsplit('.', 1)[-1] == 'TPRatFPR' else name)
                    value = values.get(native_name)
                    metrics[group][name] = value if value is not None and math.isfinite(value) else None
                counts[group] = {k: values[k] for k in ('samples', 'positives', 'negatives')}
            record = dict(identity=identities['attack'], created_at=document['updated_at'],
                metrics=metrics, group_counts=counts,
                ensemble=dict(condition=condition, origin='evaluated' if computed else 'saved_experiment',
                    artifact=None if computed else artifact['path'],
                    source_fingerprint=document['source_fingerprint'], evaluation_id=document['evaluation_id'],
                    experts=document['experts'], training=result.get('training')))
            records['attack'] = record
            status['attack'] = 'computed' if computed else 'cached'
            if not args.dry_run:
                path = cache_path(identities['attack'])
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
                    json.dump(record, handle, ensure_ascii=False, allow_nan=False)
                    temporary = Path(handle.name)
                temporary.replace(path)
    if 'shadow' in identities:
        # Neither setting defines a single combined Shadow recommender.
        records['shadow'] = None
        status['shadow'] = 'not_applicable'
    if 'target' in identities and records['target'] is None and not args.dry_run:
        target_records = {'target': None}
        try:
            evaluate_missing(config, {'target': identities['target']}, target_records, args.allow_training)
            records['target'] = target_records['target']
            status['target'] = 'computed'
        except MissingTrainingCache as error:
            missing.append(str(error))
    return '; '.join(missing) or None


def collect(args: argparse.Namespace) -> dict:
    sources = source_digest()
    rows = []
    stages = ["attack"] + (["target"] if args.target_metrics else []) + (["shadow"] if args.shadow_metrics else [])
    combinations = [(*case, weight)
                    for case in itertools.product(args.datasets, args.targets, args.pairs, args.seeds)
                    for weight in (list(dict.fromkeys(args.sal_weights)) if case[1] in {'gdmsr_sal', 'egcl_sal'} and args.sal_weights else [None])]
    for index, (dataset, target, pair, seed, sal_weight) in enumerate(combinations, 1):
        shadow, attack = pair.split(":")
        config = compose_case(dataset, target, shadow, attack, seed)
        if sal_weight is not None:
            config.target.model.attribute.defense.loss_weight = sal_weight
        selected_weight = float(config.target.model.attribute.defense.loss_weight) if target in {'gdmsr_sal', 'egcl_sal'} else None
        defense = f'SAL λ={selected_weight:g}' if selected_weight is not None else 'None'
        identities = {stage: cache_identity(config, stage, sources) for stage in stages}
        records, status = {}, {}
        refresh = args.refresh_metrics or os.environ.get("SRALT_CACHE_REFRESH") == "1"
        for stage, identity in identities.items():
            path = cache_path(identity)
            hit = path.is_file() and not refresh
            records[stage] = json.loads(path.read_text(encoding="utf-8")) if hit else None
            status[stage] = "cached" if hit else "missing"
        logging.info("[%d/%d] %s / %s / %s / seed=%d metric cache: %s", index, len(combinations),
                     dataset, target, pair, seed, status)
        missing_reason = None
        if config.attack.model.get('ensemble_condition') is not None:
            missing_reason = collect_ensemble(config, identities, records, status, args, dataset, target)
        elif not args.dry_run and any(r is None for r in records.values()):
            try:
                evaluate_missing(config, identities, records, allow_training=args.allow_training)
            except MissingTrainingCache as error:
                missing_reason = str(error)
                logging.info("Skip unavailable results: %s", missing_reason)
            status = {stage: "computed" if value == "missing" and records[stage] is not None else value
                      for stage, value in status.items()}
            logging.info("Results: %s", status)
        rows.append(dict(dataset=dataset, target=target.removesuffix('_sal'),
                         target_config=target, defense=defense, shadow=shadow, attack=attack, seed=seed,
                         tail_ratio=float(config.attack.tail_ratio), status=status, missing_reason=missing_reason,
                         results={stage: record["metrics"] if record else None for stage, record in records.items()},
                         attack_group_counts=records["attack"].get("group_counts", {}) if records["attack"] else {},
                         ensemble=records['attack'].get('ensemble') if records['attack'] else None,
                         evaluated_at={stage: record["created_at"] if record else None for stage, record in records.items()}))
    return {"rows": rows, "stages": stages, "generated_at": datetime.now().astimezone().isoformat(),
            "namespace": os.environ.get("SRALT_CACHE_VERSION", "default"), "sources": sources,
            "allow_training": args.allow_training,
            "attack_metrics": [metric_name(m) for m in config.attack.test.metrics][:3]}  # type: ignore


def write_report(payload: dict, output: Path) -> None:
    # JSON is data, never executable script; prevent labels from ending the element.
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(HTML.replace("__REPORT_DATA__", data), encoding="utf-8")


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SRALT · 对比实验报告</title>
<style>
:root{font-family:system-ui,sans-serif;color:#173047;background:#f2f5f9;font-size:14px}body{margin:0;padding:28px;max-width:1800px;margin:auto}
h1{font-size:28px;margin:0 0 8px}h2{font-size:18px;margin:0 0 16px}.muted,small{color:#53677b}.card{background:white;border:1px solid #dce4ed;border-radius:12px;padding:20px;margin-top:20px}
.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:start}label{display:inline-flex;gap:7px;align-items:center}fieldset{border:1px solid #dce4ed;border-radius:8px;min-width:120px}fieldset label{display:flex;margin:6px}select,button{padding:7px 12px;border:1px solid #b8c7d6;border-radius:6px;background:white;color:inherit}button{cursor:pointer}button:hover{background:#eef5ff}
.table-wrap{overflow:auto;max-height:65vh}table{border-collapse:collapse;white-space:nowrap;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:11px;border-bottom:1px solid #e5ebf1}th{background:#edf3fa;cursor:pointer;position:sticky;top:0;z-index:1}td:first-child,th:first-child{text-align:left}.best{color:#08674e;font-weight:700;background:#edfaf4}.missing{color:#9c4550}details{margin-top:14px}summary{cursor:pointer}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,450px),1fr));gap:18px}.chart{border:1px solid #e1e8f0;border-radius:8px;padding:14px;overflow:auto}svg{display:block;width:100%;min-width:410px}svg text{font-family:system-ui,sans-serif;fill:#354c61}.empty{text-align:center;padding:32px;color:#53677b}.badge{background:#e6effa;border-radius:20px;padding:4px 10px}header p{line-height:1.7}#summary{line-height:1.8}.toolbar{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-bottom:18px}.error{background:#fff1f2;padding:16px}
@media(max-width:600px){body{padding:14px}.card{padding:14px}h1{font-size:23px}}@media print{.controls,.toolbar button{display:none}.table-wrap{max-height:none}body{padding:0}}
</style>
<header><h1>SRALT <span class="muted">/ 对比实验</span></h1><p class="muted" id="meta"></p><p class="muted">ensemble / ensemble：默认S0，两完整专家原始概率等权平均。ensemble / shadow_user_fusion：U＋公开度，投影两路Shadow用户向量后分类。两项没有单一的合并Shadow推荐指标；历史产物未记录的人群指标显示为缺失。</p></header>
<section class="card"><h2>筛选与显示</h2><div class="controls" id="filters"></div>
<div class="toolbar" style="margin-top:18px;margin-bottom:0"><label><input type="checkbox" id="aggregate" checked>合并种子（均值 ± 样本标准差）</label><label><input type="checkbox" id="recommendation">显示 Target / Shadow 指标</label><label><input type="checkbox" id="hide-empty">隐藏整行无记录的组合</label><label>Baseline（Shadow / Attack）<select id="baseline"><option value="">不比较</option></select></label><button id="reset">重置筛选</button><button id="csv">导出当前表格 CSV</button><button id="json">导出完整结果 JSON</button></div></section>
<section class="card"><div id="summary"></div><p class="muted">数值范围 0–1；攻击指标越高表示攻击效果越强、隐私泄漏越大。均值按种子等权计算，误差线表示 ±1 样本标准差；单种子不估计标准差。缺失与非有限值显示为 —，不参与均值。</p><details><summary>评估口径与缓存信息</summary><p>头尾用户按现有 pipeline 的社交度划分，尾部比例见完整结果。Head/Tail 按关系端点计权：一个同组端点计一次，两个同组端点计两次；All 每条关系计一次。Head–Head / Head–Tail / Tail–Tail 分别表示两端均为头部、一头一尾、两端均为尾部，每条关系计一次。Isolated Tail 为 Shadow 社交图中度为 0 的尾部用户，按符合条件的端点计权，两个端点均符合时计两次；空组或需要正负两类但缺少一类时显示 —。不同数据集、Target 分面展示，不混合平均。LightGCN 仅使用用户—物品交互，是非社交对照；其关系推断效果不能直接解释为使用社交边训练导致的成员泄漏。</p><p>缓存独立于训练 pipeline，键包含配置、种子、代码、数据文件元信息及 SRALT_CACHE_VERSION。相同配置重训后，请使用 --refresh-metrics 更新；该选项仍复用现有训练缓存。页面筛选不会运行实验。结果状态 cached/computed/missing 与测试时间见 JSON。</p><p id="provenance" class="muted"></p></details></section>
<section class="card"><h2>攻击效果</h2><p class="muted">点击列标题排序；绿色标记相同数据集与 Target（逐种子模式再按 seed 分组）内的最高值。n 为参与该指标均值的有效种子数。</p><p id="baseline-note" class="muted" hidden>Δ =（当前值 − baseline 值）/ baseline 值 × 100%，是相对变化，不是百分点。仅比较相同数据集、Target；逐种子时还匹配 seed。合并时使用所选种子各自的有效均值（有效种子数可能不同）。Baseline 不受 Shadow/Attack 行筛选影响；缺失或为零时显示 Δ —，悬停可查看原因。正数表示攻击指标提高，隐私泄漏更强。</p><div class="table-wrap" id="attack-table"></div></section>
<section class="card"><h2>SAL 配对效果</h2><p class="muted">逐种子匹配无防御基线，显示相对下降 (基线 − SAL) / 基线。攻击指标下降为防御效果，推荐指标下降为效用损失。先计算各匹配种子的变化，再汇总均值与样本标准差。</p><div class="table-wrap" id="defense-table"></div></section>
<section class="card"><h2>指标图表</h2><div class="toolbar"><label>指标 <select id="chart-metric"></select></label><label>人群 <select id="chart-group"><option value="all">全部</option><option value="head">头部</option><option value="tail">尾部</option><option value="head_head">Head–Head</option><option value="head_tail">Head–Tail</option><option value="tail_tail">Tail–Tail</option><option value="isolated_tail">Isolated Tail</option></select></label></div><div class="charts" id="charts"></div></section>
<section class="card" id="rec-section" hidden><h2>Target / Shadow 推荐指标</h2><p class="muted">同一阶段相同配置的结果复用。若生成时未启用，请用 --target-metrics / --shadow-metrics 重新运行脚本。</p><div class="table-wrap" id="rec-table"></div></section>
<script type="application/json" id="report-data">__REPORT_DATA__</script>
<script>
'use strict';
const report=JSON.parse(document.getElementById('report-data').textContent);
const $=id=>document.getElementById(id), groups={all:'全部',head:'头部',tail:'尾部',head_head:'Head–Head',head_tail:'Head–Tail',tail_tail:'Tail–Tail',isolated_tail:'Isolated Tail'};
for(const row of report.rows)row.defense??='None';
const dimensions={dataset:'数据集',target:'Target',defense:'防御',shadow:'Shadow',attack:'Attack',seed:'种子'};
const escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const choices={}, initial={}; let sortKey='dataset',sortDirection=1,currentTable={headers:[],rows:[]};
const unique=items=>[...new Set(items)];
for(const [key,label] of Object.entries(dimensions)){
  const values=unique(report.rows.map(r=>String(r[key]))); choices[key]=new Set(values);initial[key]=values;
  const field=document.createElement('fieldset'),legend=document.createElement('legend');legend.textContent=label;field.append(legend);
  for(const value of values){const labelNode=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=true;input.dataset.dimension=key;input.value=value;labelNode.append(input,document.createTextNode(value));field.append(labelNode);
    input.onchange=()=>{input.checked?choices[key].add(value):choices[key].delete(value);render()};}
  $('filters').append(field);
}
for(const [key,label,values] of [['groups','表格人群',Object.keys(groups)],['metrics','表格攻击指标',report.attack_metrics]]){
  choices[key]=new Set(values);initial[key]=values;const field=document.createElement('fieldset'),legend=document.createElement('legend');legend.textContent=label;field.append(legend);
  for(const value of values){const node=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=true;input.dataset.dimension=key;input.value=value;node.append(input,document.createTextNode(groups[value]||value));field.append(node);input.onchange=()=>{input.checked?choices[key].add(value):choices[key].delete(value);render()};}
  $('filters').append(field);
}
$('meta').textContent=`生成时间 ${report.generated_at} · 命名空间 ${report.namespace} · 单文件离线报告`;
$('provenance').textContent=`源码指纹 ${report.sources} · 模式 ${report.allow_training?'指标缓存优先；模型缺失时允许训练':'指标缓存优先；复用已有模型测试，禁止训练'}`;
for(const metric of report.attack_metrics){const option=document.createElement('option');option.value=metric;option.textContent=metric;$('chart-metric').append(option);}
for(const pair of unique(report.rows.map(row=>JSON.stringify([row.shadow,row.attack])))){
  const option=document.createElement('option');option.value=pair;option.textContent=JSON.parse(pair).join(' / ');$('baseline').append(option);
}
const defaultBaseline=report.rows.some(row=>row.shadow==='ensemble'&&row.attack==='ensemble')?JSON.stringify(['ensemble','ensemble']):'';
$('baseline').value=defaultBaseline;
$('recommendation').checked=report.stages.some(stage=>stage!=='attack');
const number=n=>Number.isFinite(n)?n.toFixed(4):'—';
function stats(values){const valid=values.filter(Number.isFinite),n=valid.length;if(!n)return {mean:null,sd:null,n:0};const mean=valid.reduce((a,b)=>a+b,0)/n;return {mean,sd:n>1?Math.sqrt(valid.reduce((s,v)=>s+(v-mean)**2,0)/(n-1)):null,n};}
function statisticText(s){return s.n?`${number(s.mean)}${s.sd===null?'':` ± ${number(s.sd)}`} (n=${s.n})`:'— (n=0)';}
function selectedRows(){return report.rows.filter(row=>Object.keys(dimensions).every(key=>choices[key].has(String(row[key]))));}
function aggregate(rows){const buckets=new Map();for(const row of rows){const key=JSON.stringify([row.dataset,row.target,row.defense,row.shadow,row.attack,...($('aggregate').checked?[]:[row.seed])]);if(!buckets.has(key))buckets.set(key,{...row,runs:[]});buckets.get(key).runs.push(row);}return [...buckets.values()];}
function metricStats(row,stage,group,metric){return stats(row.runs.map(r=>r.results[stage]?.[group]?.[metric]));}
function relativeChange(row,group,metric){
  if(!$('baseline').value)return null;
  const [shadow,attack]=JSON.parse($('baseline').value);
  if(row.shadow===shadow&&row.attack===attack)return {text:'baseline',isBaseline:true,value:null,reason:'当前选择的 baseline'};
  // Keep the reference available when its rows are hidden by model filters.
  const reference={runs:report.rows.filter(run=>run.dataset===row.dataset&&run.target===row.target&&run.defense===row.defense&&
    run.shadow===shadow&&run.attack===attack&&choices.seed.has(String(run.seed))&&
    ($('aggregate').checked||run.seed===row.seed))};
  const current=metricStats(row,'attack',group,metric),base=metricStats(reference,'attack',group,metric);
  if(current.mean===null)return {text:'—',value:null,reason:'当前组合缺少该指标记录'};
  if(base.mean===null)return {text:'—',value:null,reason:'相同数据集、Target 和所选种子下没有 baseline 指标记录'};
  if(base.mean===0)return {text:'—',value:null,reason:'baseline 值为 0，无法计算相对变化百分比'};
  const value=(current.mean-base.mean)/base.mean*100;
  return {text:`${value>0?'+':''}${value.toFixed(2)}%`,value,
    reason:`baseline ${shadow} / ${attack}: ${statisticText(base)}；当前: ${statisticText(current)}`};
}
function changeLabel(change){return change.isBaseline?'baseline':`Δ ${change.text}`;}
function columns(){return [...choices.groups].flatMap(group=>[...choices.metrics].map(metric=>({group,metric,key:`${group}|${metric}`,label:`${groups[group]} · ${metric}`})));}
function sorted(rows,cols){return [...rows].sort((a,b)=>{const col=cols.find(c=>c.key===sortKey);const av=col?metricStats(a,'attack',col.group,col.metric).mean:a[sortKey],bv=col?metricStats(b,'attack',col.group,col.metric).mean:b[sortKey];if(av==null)return bv==null?0:1;if(bv==null)return -1;return (typeof av==='number'?av-bv:String(av).localeCompare(String(bv)))*sortDirection;});}
function renderTable(rows){const cols=columns(),dims=['dataset','target','defense','shadow','attack',...($('aggregate').checked?[]:['seed'])],hasBaseline=Boolean($('baseline').value),headers=[...dims.map(k=>dimensions[k]),'有效/所选种子',...(hasBaseline?['Baseline']:[]),...cols.flatMap(c=>hasBaseline?[c.label,`${c.label} · 相对 baseline (%)`]:[c.label])];
  currentTable={headers,rows:[]};if(!rows.length){$('attack-table').innerHTML='<div class="empty">没有符合筛选的组合。</div>';return;}
  const best=new Map(),bucket=r=>JSON.stringify([r.dataset,r.target,...($('aggregate').checked?[]:[r.seed])]);
  for(const row of rows)for(const col of cols){const value=metricStats(row,'attack',col.group,col.metric).mean,key=bucket(row)+col.key;if(value!==null)best.set(key,Math.max(best.get(key)??-Infinity,value));}
  let html='<table><thead><tr>'+dims.map(k=>`<th data-sort="${k}">${dimensions[k]}</th>`).join('')+'<th>有效/所选种子</th>'+cols.map(c=>`<th data-sort="${escapeHTML(c.key)}">${escapeHTML(c.label)}</th>`).join('')+'</tr></thead><tbody>';
  for(const row of sorted(rows,cols)){const valid=row.runs.filter(r=>r.results.attack).length;const values=dims.map(k=>row[k]);values.push(`${valid}/${row.runs.length}`);if(hasBaseline)values.push(JSON.parse($('baseline').value).join(' / '));html+='<tr>'+dims.map(k=>`<td>${escapeHTML(row[k])}</td>`).join('')+`<td class="${valid<row.runs.length?'missing':''}">${valid}/${row.runs.length}</td>`;
    for(const col of cols){const s=metricStats(row,'attack',col.group,col.metric),change=relativeChange(row,col.group,col.metric);values.push(statisticText(s));if(change)values.push(change.text);html+=`<td class="${s.mean!==null&&s.mean===best.get(bucket(row)+col.key)?'best':''}">${escapeHTML(statisticText(s))}${change?`<br><small title="${escapeHTML(change.reason)}">${escapeHTML(changeLabel(change))}</small>`:''}</td>`;}html+='</tr>';currentTable.rows.push(values);}
  $('attack-table').innerHTML=html+'</tbody></table>';$('attack-table').querySelectorAll('[data-sort]').forEach(th=>th.onclick=()=>{sortDirection=sortKey===th.dataset.sort?-sortDirection:1;sortKey=th.dataset.sort;render();});
}
function renderCharts(rows){const metric=$('chart-metric').value,group=$('chart-group').value,facets=new Map();for(const row of rows){const key=JSON.stringify([row.dataset,row.target]);if(!facets.has(key))facets.set(key,[]);facets.get(key).push(row);}
  $('charts').innerHTML='';if(!rows.length){$('charts').textContent='没有符合筛选的组合。';return;}
  for(const facet of facets.values()){const width=760,left=245,right=65,height=64+facet.length*42,scale=width-left-right;let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHTML(metric+' '+groups[group])}">`;
    for(let i=0;i<=5;i++){const x=left+i*scale/5;svg+=`<line x1="${x}" x2="${x}" y1="25" y2="${height-25}" stroke="#e1e8f0"/><text x="${x}" y="16" text-anchor="middle" font-size="12">${(i/5).toFixed(1)}</text>`;}
    facet.forEach((row,i)=>{const y=36+i*42,s=metricStats(row,'attack',group,metric),change=relativeChange(row,group,metric),label=`${row.defense} / ${row.shadow} / ${row.attack}${$('aggregate').checked?'':` / ${row.seed}`}`,color=['#3978d2','#1b927f','#9866c9','#da8242'][initial.shadow.indexOf(row.shadow)%4];
      svg+=`<text x="${left-12}" y="${y+12}" text-anchor="end" font-size="13">${escapeHTML(label)}</text>`;
      if(s.n){const x=left+s.mean*scale,lo=left+Math.max(0,s.mean-(s.sd||0))*scale,hi=left+Math.min(1,s.mean+(s.sd||0))*scale;svg+=`<rect x="${left}" y="${y}" width="${Math.max(0,s.mean*scale)}" height="20" rx="3" fill="${color}"><title>${escapeHTML(label+' · '+statisticText(s)+(change?' · '+changeLabel(change):''))}</title></rect>`;if(s.sd!==null)svg+=`<path d="M ${lo} ${y+10} H ${hi} M ${lo} ${y+5} V ${y+15} M ${hi} ${y+5} V ${y+15}" stroke="#173047" fill="none"/>`;svg+=`<text x="${width-right+8}" y="${y+13}" font-size="12">${number(s.mean)}</text>`;}else svg+=`<text x="${left+8}" y="${y+13}" font-size="12">无有效结果</text>`;
    });svg+='</svg>';const node=document.createElement('div');node.className='chart';node.innerHTML=`<h2>${escapeHTML(facet[0].dataset)} / ${escapeHTML(facet[0].target)}</h2><small>${escapeHTML(metric)} · ${groups[group]}</small>${svg}`;$('charts').append(node);}
}
function renderRecommendation(rows){$('rec-section').hidden=!$('recommendation').checked;if($('rec-section').hidden)return;let html='<table><thead><tr><th>数据集</th><th>Target</th><th>防御</th><th>Shadow</th><th>阶段</th><th>指标</th><th>均值 ± 标准差</th></tr></thead><tbody>';const seen=new Set();let count=0;
  for(const row of rows)for(const stage of ['target','shadow']){const metrics=unique(row.runs.flatMap(r=>Object.keys(r.results[stage]?.all||{})));const base=[row.dataset,row.target,row.defense,...(stage==='shadow'?[row.shadow]:[]),...($('aggregate').checked?[]:[row.seed])];for(const metric of metrics){const key=JSON.stringify([stage,metric,...base]);if(seen.has(key))continue;seen.add(key);count++;const matching=selectedRows().filter(r=>r.dataset===row.dataset&&r.target===row.target&&r.defense===row.defense&&(stage==='target'||r.shadow===row.shadow)&&($('aggregate').checked||r.seed===row.seed));const perSeed=new Map(matching.filter(r=>r.results[stage]).map(r=>[r.seed,r]));const s=stats([...perSeed.values()].map(r=>r.results[stage]?.all?.[metric]));html+='<tr>'+[row.dataset,row.target,row.defense,stage==='shadow'?row.shadow:'—',stage,metric].map(v=>`<td>${escapeHTML(v)}</td>`).join('')+`<td>${escapeHTML(($('aggregate').checked?'':`seed=${row.seed} · `)+statisticText(s))}</td></tr>`;}}
  $('rec-table').innerHTML=count?html+'</tbody></table>':'<div class="empty">当前结果没有推荐指标。生成时启用 --target-metrics / --shadow-metrics。</div>';
}
function renderDefense(rows){
  const buckets=new Map();
  for(const row of rows){
    if(!row.defense.startsWith('SAL '))continue;
    const baseline=report.rows.find(r=>r.dataset===row.dataset&&r.target===row.target&&r.defense==='None'&&r.shadow===row.shadow&&r.attack===row.attack&&r.seed===row.seed);
    for(const stage of ['attack','target'])for(const [group,metrics] of Object.entries(row.results[stage]||{}))for(const [metric,value] of Object.entries(metrics)){
      if(stage==='attack'&&(!choices.groups.has(group)||!choices.metrics.has(metric)))continue;
      const base=baseline?.results[stage]?.[group]?.[metric];
      const key=JSON.stringify([row.dataset,row.target,row.defense,row.shadow,row.attack,stage,group,metric,...($('aggregate').checked?[]:[row.seed])]);
      if(!buckets.has(key))buckets.set(key,{row,stage,group,metric,base:[],value:[],changes:[],seeds:[]});
      const bucket=buckets.get(key);
      if(Number.isFinite(base)&&Number.isFinite(value)&&base!==0){bucket.base.push(base);bucket.value.push(value);bucket.changes.push((base-value)/base*100);bucket.seeds.push(row.seed);}
    }
  }
  let html='<table><thead><tr>'+['数据集','Target','防御','Shadow / Attack','阶段 / 人群','指标','匹配种子','无防御','SAL','相对下降 (%)'].map(v=>`<th>${v}</th>`).join('')+'</tr></thead><tbody>';
  for(const b of buckets.values())html+='<tr>'+[b.row.dataset,b.row.target,b.row.defense,`${b.row.shadow} / ${b.row.attack}`,`${b.stage} / ${groups[b.group]||b.group}`,b.metric,b.seeds.join(', ')||'缺少配对结果',statisticText(stats(b.base)),statisticText(stats(b.value)),statisticText(stats(b.changes))].map(v=>`<td>${escapeHTML(v)}</td>`).join('')+'</tr>';
  $('defense-table').innerHTML=buckets.size?html+'</tbody></table>':'<p class="muted">选择同一目标模型的无防御与 SAL 配对结果以查看防御效果。</p>';
}
function render(){const selected=selectedRows(),allRows=aggregate(selected),rows=$('hide-empty').checked?allRows.filter(row=>row.runs.some(run=>run.results.attack!=null)):allRows,cached=selected.filter(r=>r.status.attack==='cached').length,computed=selected.filter(r=>r.status.attack==='computed').length,missing=selected.filter(r=>!r.results.attack).length;
  $('baseline-note').hidden=!$('baseline').value;
  $('summary').textContent=`${rows.length} 个展示组合 · 隐藏 ${allRows.length-rows.length} 个无记录组合 · ${selected.length} 次种子运行 · 攻击结果：缓存 ${cached} / 本次计算 ${computed} / 缺失 ${missing}`;
  renderTable(rows);renderCharts(rows);renderRecommendation(allRows);renderDefense(selected);
}
function download(name,content,type){const url=URL.createObjectURL(new Blob(['\ufeff',content],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('csv').onclick=()=>{const quote=v=>'"'+String(v).replace(/"/g,'""')+'"';download('sralt-comparison.csv',[currentTable.headers,...currentTable.rows].map(row=>row.map(quote).join(',')).join('\r\n'),'text/csv;charset=utf-8');};
$('json').onclick=()=>download('sralt-comparison.json',JSON.stringify(report,null,2),'application/json');
$('reset').onclick=()=>{for(const [key,values] of Object.entries(initial))choices[key]=new Set(values);$('filters').querySelectorAll('input').forEach(i=>i.checked=true);$('hide-empty').checked=false;$('baseline').value=defaultBaseline;render();};
for(const id of ['aggregate','recommendation','hide-empty','baseline','chart-metric','chart-group'])$(id).onchange=render;
render();
</script></html>'''


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.chdir(ROOT)  # Existing pipeline paths/caches are relative to repository root.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    payload = collect(args)
    if not args.dry_run:
        output = args.output.resolve()
        write_report(payload, output)
        logging.info("HTML report: %s", output)


if __name__ == "__main__":
    main()

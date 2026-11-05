"""Score a cached audit and construct reproducible source-grouped evaluation folds."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from extract import BLOCKS, write_json

SEED = 20261102


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                             else v for k, v in row.items()})


class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, key):
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def annotation_signature(annotations):
    def rounded(value):
        if isinstance(value, float):
            return round(value, 3)
        if isinstance(value, list):
            return [rounded(v) for v in value]
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        return value
    return sorted(json.dumps(rounded({k: a.get(k) for k in
                                      ("category_id", "bbox", "segmentation", "iscrowd")}),
                             sort_keys=True) for a in annotations)


def resolve_groups(audit, original):
    images = {i["key"]: i for i in audit["images"]}
    union = UnionFind(images)
    buckets = {field: collections.defaultdict(list) for field in
               ("source_group", "canonical_object", "pixel_sha256")}
    for key, im in images.items():
        for field in buckets:
            if im.get(field):
                buckets[field][im[field]].append(key)
    for by_value in buckets.values():
        for keys in by_value.values():
            for key in keys[1:]:
                union.union(keys[0], key)
    components = collections.defaultdict(list)
    for key in images:
        components[union.find(key)].append(key)
    ordered = sorted(components.values(), key=lambda keys: min(images[k]["source_group"] for k in keys))
    group_of = {key: "G%03d" % (n+1) for n, keys in enumerate(ordered) for key in keys}
    annotations = collections.defaultdict(list)
    for split, data in original.items():
        for a in data["annotations"]:
            annotations[split+":"+str(a["image_id"])].append(a)
    duplicate_rows, conflicts = [], set()
    for digest, keys in buckets["pixel_sha256"].items():
        if len(keys) < 2:
            continue
        signatures = [annotation_signature(annotations[k]) for k in keys]
        conflict = any(s != signatures[0] for s in signatures[1:])
        if conflict:
            conflicts.update(keys)
        duplicate_rows.append({
            "pixel_sha256": digest, "count": len(keys),
            "cross_original_split": len({images[k]["split"] for k in keys}) > 1,
            "annotation_conflict": conflict, "image_keys": keys,
            "files": [images[k]["split"]+"/"+images[k]["file_name"] for k in keys],
            "component": group_of[keys[0]],
        })
    return group_of, duplicate_rows, conflicts, annotations


def grouped_folds(images, eligible, groups, labels, categories, trials=400):
    """Multilabel group balancing. Uses labels/counts, never holdout model results."""
    names = sorted({groups[k] for k in eligible})
    group_index = {g: n for n, g in enumerate(names)}
    category_index = {c: n for n, c in enumerate(categories)}
    counts = np.zeros((len(names), len(categories)), dtype=float)
    sizes = np.zeros(len(names), dtype=float)
    for key in sorted(eligible):
        g = group_index[groups[key]]
        sizes[g] += 1
        for c in labels[key]:
            if c in category_index:
                counts[g, category_index[c]] += 1
    target = np.maximum(counts.sum(axis=0)/5, 1)
    target_size = sizes.sum()/5
    rng = np.random.RandomState(SEED)
    best = None
    priority = (counts/np.maximum(counts.sum(axis=0), 1)).sum(axis=1)
    for trial in range(trials):
        order = np.argsort(-(priority * rng.uniform(.5, 1.5, len(names))))
        fc = np.zeros((5, len(categories)))
        fs = np.zeros(5)
        fg = np.zeros((5, len(categories)))
        assignment = np.zeros(len(names), dtype=int)
        for g in order:
            delta = (((fc+counts[g])/target)**2-(fc/target)**2).sum(axis=1)
            delta += .5 * (((fs+sizes[g])/target_size)**2-(fs/target_size)**2)
            # Seed ties without systematically favoring fold zero.
            choice = int(np.argmin(delta+rng.uniform(0, 1e-8, 5)))
            assignment[g] = choice
            fc[choice] += counts[g]
            fs[choice] += sizes[g]
            fg[choice] += counts[g] > 0
        loss = float(np.mean((fc/target-1)**2) + .5*np.mean((fs/target_size-1)**2)
                     + 10*np.mean(fg < 2))
        if best is None or loss < best[0]:
            best = (loss, assignment.copy(), fc.copy(), fs.copy(), fg.copy())
    fold_of = {key: int(best[1][group_index[groups[key]]]) for key in eligible}
    support = []
    for f in range(5):
        for ci, c in enumerate(categories):
            support.append({"fold": f, "category_id": c, "positive_images": int(best[2][f, ci]),
                            "independent_groups": int(best[4][f, ci]),
                            "total_fold_images": int(best[3][f])})
    return fold_of, support, best[0]


def fit_transform(train, other):
    low, median, high = np.percentile(train, [10, 50, 90], axis=0)
    scale = np.maximum(high-low, 1e-3)
    weights = np.ones(train.shape[1], np.float32)
    for start, end, weight in BLOCKS:
        weights[start:end] = math.sqrt(weight/(end-start))
    def apply(x):
        return (np.clip((x-median)/scale, -5, 5)*weights).astype(np.float32)
    return apply(train), apply(other)


def distinct_image_neighbors(x, image_keys):
    """Leave-one-image-content-out distance, including fallback for dense images."""
    k = min(32, len(x))
    model = NearestNeighbors(n_neighbors=k, n_jobs=2).fit(x)
    distances, neighbors = model.kneighbors(x)
    out, nearest = np.zeros(len(x)), np.zeros(len(x), int)
    for row in range(len(x)):
        valid = np.flatnonzero(image_keys[neighbors[row]] != image_keys[row])
        if len(valid):
            p = valid[0]
            out[row], nearest[row] = distances[row, p], neighbors[row, p]
        else:
            allowed = np.flatnonzero(image_keys != image_keys[row])
            if not len(allowed):
                out[row], nearest[row] = 0, row
            else:
                dist = np.linalg.norm(x[allowed]-x[row], axis=1)
                p = int(np.argmin(dist))
                out[row], nearest[row] = dist[p], allowed[p]
    return out, nearest


def evaluate(name, train_keys, val_keys, records, features, groups, categories,
             eligible, pixel_hashes, cluster_multiplier=1.0, details=True):
    train_groups = {groups[k] for k in train_keys}
    overlap = {k for k in val_keys if groups[k] in train_groups}
    result = {"name": name, "train_images": len(train_keys), "val_images": len(val_keys),
              "group_overlap_val_images": len(overlap),
              "group_isolation": 1-len(overlap)/max(len(val_keys), 1),
              "classes": [], "modes": [], "nearest_pairs": [], "gaps": []}
    for category in categories:
        train_idx = [n for n, r in enumerate(records) if r["category_id"] == category
                     and r["image_key"] in train_keys and r["image_key"] in eligible and not r["issues"]]
        val_idx = [n for n, r in enumerate(records) if r["category_id"] == category
                   and r["image_key"] in val_keys and r["image_key"] in eligible and not r["issues"]]
        ti = np.array(train_idx, dtype=int)
        vi = np.array(val_idx, dtype=int)
        tr = [records[n] for n in ti]
        va = [records[n] for n in vi]
        if len(tr) < 2 or len({r["image_key"] for r in tr}) < 2 or not va:
            raise ValueError("Insufficient eligible data for class %s in %s" % (category, name))
        x, v = fit_transform(features[ti], features[vi])
        k = min(len(tr), 20, max(2, int(round(math.sqrt(len(tr)/10)*cluster_multiplier))))
        km = MiniBatchKMeans(n_clusters=k, random_state=SEED+category, n_init=5,
                            batch_size=256, max_iter=150, reassignment_ratio=.01)
        assignment = km.fit_predict(x)
        val_assignment = km.predict(v)
        train_nn, nearest = distinct_image_neighbors(
            x, np.array([pixel_hashes[r["image_key"]] for r in tr]))
        # Empirical calibration, not a semantic-equivalence or deletion threshold.
        threshold = max(float(np.percentile(train_nn, 90)), 1e-6)
        model = NearestNeighbors(n_neighbors=1, n_jobs=2).fit(v)
        to_val, _ = model.kneighbors(x)
        to_val = to_val[:, 0]
        val_to_train = NearestNeighbors(n_neighbors=1, n_jobs=2).fit(x).kneighbors(v)[0][:, 0]
        eligible_modes, covered_modes, present_modes = 0, 0, 0
        for c in range(k):
            members = np.flatnonzero(assignment == c)
            val_members = np.flatnonzero(val_assignment == c)
            imgs = {tr[n]["image_key"] for n in members}
            gs = {groups[key] for key in imgs}
            vimgs = {va[n]["image_key"] for n in val_members}
            vgs = {groups[key] for key in vimgs}
            supported = len(imgs) >= 5 and len(gs) >= 2
            covered = len(vimgs) >= 3 and len(vgs) >= 2
            eligible_modes += supported
            covered_modes += supported and covered
            present_modes += supported and bool(vimgs)
            if len(members):
                center_dist = np.linalg.norm(x[members]-km.cluster_centers_[c], axis=1)
                representative = int(ti[members[int(np.argmin(center_dist))]])
                mode = {"evaluation": name, "category_id": category, "mode": c,
                        "train_instances": len(members), "train_images": len(imgs), "train_groups": len(gs),
                        "val_instances": len(val_members), "val_images": len(vimgs), "val_groups": len(vgs),
                        "eligible": bool(supported), "covered": bool(covered),
                        "representative_record": representative}
                result["modes"].append(mode)
                if supported and not covered:
                    result["gaps"].append(mode)
        area_edges = np.quantile([r["area_fraction"] for r in tr], [1/3, 2/3])
        contrast_edges = np.quantile([abs(r["contrast"]) for r in tr], [1/3, 2/3])
        def cell(r):
            return (min(2, max(0, int(r["cx"]*3))), min(2, max(0, int(r["cy"]*3))),
                    int(np.searchsorted(area_edges, r["area_fraction"])),
                    int(np.searchsorted(contrast_edges, abs(r["contrast"]))))
        cells, val_cells = collections.defaultdict(set), collections.defaultdict(set)
        for r in tr:
            cells[cell(r)].add(r["image_key"])
        for r in va:
            val_cells[cell(r)].add(r["image_key"])
        eligible_cells = [c for c, imgs in cells.items() if len(imgs) >= 5
                          and len({groups[key] for key in imgs}) >= 2]
        covered_cells = sum(bool(val_cells[c]) for c in eligible_cells)
        vimgs = {r["image_key"] for r in va}
        vg = {groups[key] for key in vimgs}
        # Count one exact image content per category and account for source concentration.
        group_contents = collections.defaultdict(set)
        for key in vimgs:
            group_contents[groups[key]].add(pixel_hashes[key])
        unique_count = sum(len(v) for v in group_contents.values())
        proportions = np.array([len(v)/unique_count for v in group_contents.values()])
        effective_groups = float(1/np.sum(proportions**2))
        adequacy = math.sqrt(min(unique_count/100, 1)*min(effective_groups/5, 1))
        class_result = {
            "category_id": category, "train_instances": len(tr), "val_instances": len(va),
            "train_positive_images": len({r["image_key"] for r in tr}), "val_positive_images": len(vimgs),
            "train_groups": len({groups[r["image_key"]] for r in tr}), "val_groups": len(vg),
            "val_unique_images": unique_count, "val_effective_groups": effective_groups,
            "val_top_group_fraction": float(proportions.max()),
            "adequacy": adequacy, "clusters": k, "eligible_modes": eligible_modes,
            "covered_modes": covered_modes, "present_modes": present_modes,
            "mode_coverage": covered_modes/max(eligible_modes, 1),
            "eligible_joint_cells": len(eligible_cells), "covered_joint_cells": covered_cells,
            "joint_coverage": covered_cells/max(len(eligible_cells), 1),
            "feature_coverage": float(np.mean(to_val <= threshold)),
            "empirical_nn_threshold": threshold,
            "to_val_distance_p50": float(np.percentile(to_val, 50)),
            "to_val_distance_p95": float(np.percentile(to_val, 95)),
            "val_novelty_fraction": float(np.mean(val_to_train > threshold)),
            "short_px_p10": float(np.percentile([r["short_px"] for r in tr], 10)),
            "area_fraction_p50": float(np.median([r["area_fraction"] for r in tr])),
        }
        class_result["coverage"] = (.45*class_result["mode_coverage"]
                                    + .35*class_result["joint_coverage"]
                                    + .20*class_result["feature_coverage"])
        result["classes"].append(class_result)
        if details:
            seen_pairs = set()
            for n in np.argsort(train_nn):
                pair = tuple(sorted((int(ti[n]), int(ti[nearest[n]]))))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                result["nearest_pairs"].append({
                    "evaluation": name, "category_id": category,
                    "record_a": pair[0], "record_b": pair[1], "distance": float(train_nn[n]),
                    "same_group": groups[tr[n]["image_key"]] == groups[tr[nearest[n]]["image_key"]],
                })
                if len(seen_pairs) == 8:
                    break
        print("%s class %s: %d/%d images, C=%.3f" %
              (name, category, class_result["train_positive_images"], len(vimgs),
               class_result["coverage"]), flush=True)
    for key in ("adequacy", "mode_coverage", "joint_coverage", "feature_coverage", "coverage"):
        result[key] = float(np.mean([c[key] for c in result["classes"]]))
    return result


def export_coco(path, keys, original, original_annotations):
    # Relative filenames include the original train/val directory; root is format3.
    result = {"info": {"description": "AUDIT CANDIDATE: source semantics unconfirmed; not an independent final test"},
              "images": [], "annotations": [], "categories": original["train"]["categories"]}
    ann_id = 1
    for split in ("train", "val"):
        for image in original[split]["images"]:
            key = split+":"+str(image["id"])
            if key not in keys:
                continue
            new_id = len(result["images"])+1
            result["images"].append(dict(image, id=new_id, file_name=split+"/"+image["file_name"]))
            for a in original_annotations[key]:
                result["annotations"].append(dict(a, id=ann_id, image_id=new_id))
                ann_id += 1
    write_json(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.cache/"audit.json").read_text(encoding="utf-8"))
    features = np.load(str(args.cache/"features.npz"))["features"]
    assert len(features) == len(audit["records"]) and np.isfinite(features).all()
    original = {}
    for split, p in audit["provenance"].items():
        raw = Path(p["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != p["sha256"]:
            raise ValueError("Annotation changed since extraction; rerun extraction")
        original[split] = json.loads(raw.decode("utf-8-sig"))
    images = {i["key"]: i for i in audit["images"]}
    records = audit["records"]
    categories = sorted(int(c) for c in audit["categories"])
    pixel_hashes = {k: i.get("pixel_sha256", "") for k, i in images.items()}
    groups, duplicates, conflicts, annotations = resolve_groups(audit, original)
    problem_keys = {p["image_key"] for p in audit["problems"] if "image_key" in p}
    excluded = {k for k, i in images.items() if not i["readable"] or i["issues"]}
    excluded |= problem_keys | conflicts
    eligible = set(images)-excluded
    labels = {k: {a["category_id"] for a in annotations[k]} for k in images}
    fold_of, support, split_loss = grouped_folds(images, eligible, groups, labels, categories)
    for f in range(5):
        tr, va = {k for k in eligible if fold_of[k] != f}, {k for k in eligible if fold_of[k] == f}
        assert tr.isdisjoint(va) and tr | va == eligible
        assert {groups[k] for k in tr}.isdisjoint({groups[k] for k in va})
        assert {images[k]["pixel_sha256"] for k in tr}.isdisjoint(
            {images[k]["pixel_sha256"] for k in va})
    current = evaluate("current", {k for k in images if images[k]["split"] == "train"},
                       {k for k in images if images[k]["split"] == "val"},
                       records, features, groups, categories, eligible, pixel_hashes)
    candidate_train = {k for k in eligible if fold_of[k] != 0}
    candidate_val = {k for k in eligible if fold_of[k] == 0}
    candidate = evaluate("grouped_fold0", candidate_train, candidate_val,
                         records, features, groups, categories, eligible, pixel_hashes)
    grouped_evaluations = [candidate]
    for fold in range(1, 5):
        grouped_evaluations.append(evaluate(
            "grouped_fold%d" % fold, {k for k in eligible if fold_of[k] != fold},
            {k for k in eligible if fold_of[k] == fold},
            records, features, groups, categories, eligible, pixel_hashes, details=False))
    integrity = len(eligible)/len(images)
    readable = [i for i in images.values() if i["readable"]]
    unique_fraction = len({i["pixel_sha256"] for i in readable})/len(readable)
    for result in [current]+grouped_evaluations:
        # Fixed global input-quality terms permit a before/after comparison.
        reliability = (.30*integrity+.15*unique_fraction+.35*result["group_isolation"]
                       + .20*result["adequacy"])
        coverage = result["coverage"]
        result.update(integrity=integrity, unique_fraction=unique_fraction, reliability=reliability,
                      audit_index=100*2*reliability*coverage/max(reliability+coverage, 1e-12))
    # Sensitivity check: repeat the original evaluation with fewer/more modes.
    sensitivity = []
    for multiplier in (.7, 1.3):
        check = evaluate("current_k%.1f" % multiplier,
                         {k for k in images if images[k]["split"] == "train"},
                         {k for k in images if images[k]["split"] == "val"},
                         records, features, groups, categories, eligible, pixel_hashes, multiplier, False)
        sensitivity.append({"cluster_multiplier": multiplier, "coverage": check["coverage"],
                            "mode_coverage": check["mode_coverage"]})
    image_rows = []
    for key, im in images.items():
        reasons = list(im["issues"])
        if key in problem_keys:
            reasons.append("invalid_or_outside_bbox")
        if key in conflicts:
            reasons.append("pixel_identical_annotation_conflict")
        image_rows.append(dict(im, component=groups[key], eligible=key in eligible,
                               exclusion_reasons=reasons, fold=fold_of.get(key, ""),
                               category_ids=sorted(labels[key])))
    instance_rows = [dict(r, feature_row=n, component=groups[r["image_key"]],
                          eligible=r["image_key"] in eligible and not r["issues"])
                     for n, r in enumerate(records)]
    overview = {
        "images": len(images), "annotations": sum(len(d["annotations"]) for d in original.values()),
        "readable_images": len(readable), "extracted_defects": len(records),
        "eligible_images": len(eligible), "excluded_images": len(excluded),
        "source_groups": len({i["source_group"] for i in images.values()}),
        "connected_groups": len(set(groups.values())),
        "unknown_source_images": sum(i["source_group"].startswith("UNKNOWN:") for i in images.values()),
        "duplicate_pixel_groups": len(duplicates),
        "redundant_pixel_images": sum(d["count"]-1 for d in duplicates),
        "cross_split_duplicate_groups": sum(d["cross_original_split"] for d in duplicates),
        "annotation_conflict_groups": sum(d["annotation_conflict"] for d in duplicates),
        "annotation_conflict_images": len(conflicts),
        "image_issue_counts": dict(collections.Counter(issue for i in images.values() for issue in i["issues"])),
        "annotation_issue_counts": dict(collections.Counter(p["error"] for p in audit["problems"])),
        "unannotated_images": sum(not annotations[k] for k in images),
        "created_at": datetime.datetime.now().isoformat(),
    }
    grouped_summary = {}
    for field in ("audit_index", "reliability", "coverage", "adequacy",
                  "mode_coverage", "joint_coverage", "feature_coverage"):
        values = [e[field] for e in grouped_evaluations]
        grouped_summary[field] = {"mean": float(np.mean(values)), "min": float(np.min(values)),
                                  "max": float(np.max(values)), "std": float(np.std(values))}
    group_profiles = []
    for category in categories:
        recs = [r for r in records if r["category_id"] == category
                and r["image_key"] in eligible and not r["issues"]]
        for group in sorted({groups[r["image_key"]] for r in recs}):
            rs = [r for r in recs if groups[r["image_key"]] == group]
            group_profiles.append({
                "category_id": category, "category_name": audit["categories"][str(category)],
                "component": group, "source_groups": sorted({r["source_group"] for r in rs}),
                "images": len({r["image_key"] for r in rs}), "instances": len(rs),
                "fold": fold_of[rs[0]["image_key"]],
                **{key+"_median": float(np.median([r[key] for r in rs]))
                   for key in ("area_fraction", "short_px", "cx", "cy", "gray_mean", "contrast")},
            })
    summary = {
        "overview": overview, "provenance": audit["provenance"], "categories": audit["categories"],
        "feature_method": audit["feature_method"], "versions": audit["versions"],
        "current": current, "candidate": candidate, "sensitivity": sensitivity,
        "grouped_summary": grouped_summary, "grouped_evaluations": grouped_evaluations,
        "fold_support": support, "split_objective": split_loss, "seed": SEED,
        "thresholds": {"adequacy_images": 100, "adequacy_effective_groups": 5, "train_mode_images": 5,
                       "train_mode_groups": 2, "val_mode_images": 3, "val_mode_groups": 2,
                       "nearest_distance_train_percentile": 90},
        "limitations": [
            "Source group L/R-number and object identity semantics not confirmed by dataset owner.",
            "No model training, registered product coordinate system, defect masks, or verified normal set.",
            "Handcrafted feature distances are diagnostic; semantic equivalence not human calibrated.",
            "Coverage references observed training data, not all possible production defects.",
            "All supplied labels informed balancing; candidate folds are not an untouched final test set.",
        ],
    }
    write_json(args.output/"metrics.json", summary)
    write_csv(args.output/"images.csv", image_rows)
    write_csv(args.output/"defects.csv", instance_rows)
    write_csv(args.output/"issues.csv", [
        {"image_key": r["key"], "file_name": r["file_name"], "split": r["split"],
         "reasons": r["exclusion_reasons"]} for r in image_rows if not r["eligible"]])
    write_csv(args.output/"annotation_issues.csv", audit["problems"])
    write_csv(args.output/"duplicates.csv", duplicates)
    write_csv(args.output/"fold_support.csv", support)
    write_csv(args.output/"group_profiles.csv", group_profiles)
    write_csv(args.output/"modes.csv", current["modes"]+[m for e in grouped_evaluations for m in e["modes"]])
    write_csv(args.output/"coverage_gaps.csv", current["gaps"]+[g for e in grouped_evaluations for g in e["gaps"]])
    pairs = current["nearest_pairs"]
    for p in pairs:
        for suffix in ("a", "b"):
            r = records[p["record_"+suffix]]
            p["image_"+suffix] = r["image_key"]
            p["annotation_"+suffix] = r["annotation_id"]
            p["file_"+suffix] = r["split"]+"/"+r["file_name"]
    write_csv(args.output/"similarity_review.csv", pairs)
    write_csv(args.output/"grouped_folds.csv", [
        {"image_key": key, "file_name": images[key]["split"]+"/"+images[key]["file_name"],
         "component": groups[key], "source_group": images[key]["source_group"], "fold": fold_of[key],
         "category_ids": sorted(labels[key])} for key in sorted(eligible)])
    export_coco(args.output/"candidate_fold0_train.coco.json", candidate_train, original, annotations)
    export_coco(args.output/"candidate_fold0_val.coco.json", candidate_val, original, annotations)
    write_json(args.cache/"analysis_details.json", {"images": image_rows, "records": instance_rows})
    print(json.dumps(overview, ensure_ascii=False, indent=2), flush=True)
    print("Metrics and candidate splits: %s" % args.output, flush=True)


if __name__ == "__main__":
    main()

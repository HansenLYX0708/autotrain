"""Verify generated evidence, source isolation, and annotation-preserving exports."""
import argparse
import collections
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from analyze import UnionFind, annotation_signature, distinct_image_neighbors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.cache/"audit.json").read_text(encoding="utf-8"))
    metrics = json.loads((args.output/"metrics.json").read_text(encoding="utf-8"))
    details = json.loads((args.cache/"analysis_details.json").read_text(encoding="utf-8"))
    images = {i["key"]: i for i in details["images"]}
    eligible = {k for k, i in images.items() if i["eligible"]}
    assert len(eligible) == metrics["overview"]["eligible_images"]
    assert len(images) == metrics["overview"]["images"]
    assert len(details["records"]) == metrics["overview"]["extracted_defects"]
    features = np.load(str(args.cache/"features.npz"))["features"]
    assert features.shape == (len(details["records"]), 174) and np.isfinite(features).all()
    with (args.output/"grouped_folds.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(eligible)
    assert {r["image_key"] for r in rows} == eligible
    fold_of = {r["image_key"]: int(r["fold"]) for r in rows}
    for fold in range(5):
        tr = {k for k in eligible if fold_of[k] != fold}
        va = eligible-tr
        assert tr and va and tr.isdisjoint(va)
        for key in ("component", "source_group", "canonical_object", "pixel_sha256"):
            assert {images[k][key] for k in tr if images[k].get(key)}.isdisjoint(
                {images[k][key] for k in va if images[k].get(key)})
    old_train_groups = {i["component"] for i in images.values() if i["split"] == "train"}
    old_val = [i for i in images.values() if i["split"] == "val"]
    expected_isolation = 1-sum(i["component"] in old_train_groups for i in old_val)/len(old_val)
    assert metrics["current"]["group_isolation"] == expected_isolation
    assert metrics["candidate"]["group_isolation"] == 1
    for result in [metrics["current"]]+metrics["grouped_evaluations"]:
        for cls in result["classes"]:
            coverage = .45*cls["mode_coverage"]+.35*cls["joint_coverage"]+.20*cls["feature_coverage"]
            assert math.isclose(cls["coverage"], coverage)
            adequacy = math.sqrt(min(cls["val_unique_images"]/100, 1)*min(cls["val_effective_groups"]/5, 1))
            assert math.isclose(cls["adequacy"], adequacy)
        r = (.30*result["integrity"]+.15*result["unique_fraction"]
             + .35*result["group_isolation"]+.20*result["adequacy"])
        c = float(np.mean([v["coverage"] for v in result["classes"]]))
        assert math.isclose(result["audit_index"], 100*2*r*c/(r+c))
        assert 0 <= result["audit_index"] <= 100
        if result["name"].startswith("grouped_"):
            assert result["group_isolation"] == 1
    original_annotations = collections.defaultdict(list)
    original_images = {}
    for split, meta in audit["provenance"].items():
        original = json.loads(Path(meta["path"]).read_text(encoding="utf-8-sig"))
        for im in original["images"]:
            original_images[split+"/"+im["file_name"]] = (split+":"+str(im["id"]), im)
        for a in original["annotations"]:
            original_annotations[split+":"+str(a["image_id"])].append(a)
    for split in ("train", "val"):
        result = json.loads((args.output/("candidate_fold0_%s.coco.json" % split)).read_text(encoding="utf-8"))
        anns = collections.defaultdict(list)
        assert len({i["id"] for i in result["images"]}) == len(result["images"])
        assert len({a["id"] for a in result["annotations"]}) == len(result["annotations"])
        for a in result["annotations"]:
            anns[a["image_id"]].append(a)
        expected = {k for k in eligible if (fold_of[k] == 0) == (split == "val")}
        seen = set()
        for im in result["images"]:
            key, original_im = original_images[im["file_name"]]
            seen.add(key)
            assert (im["width"], im["height"]) == (original_im["width"], original_im["height"])
            assert annotation_signature(anns[im["id"]]) == annotation_signature(original_annotations[key])
            assert (Path(audit["root"])/im["file_name"]).is_file()
        assert seen == expected and len(seen) == len(result["images"])
        assert set(anns).issubset({i["id"] for i in result["images"]})
    doc = (args.output/"report.html").read_text(encoding="utf-8")
    assert 'lang="zh-CN"' in doc and "�" not in doc
    assert doc.count("data:image/") >= 5
    for target in re.findall(r'href="([^"]+)"', doc):
        if target.startswith("#"):
            assert 'id="%s"' % target[1:] in doc
        elif "://" not in target:
            assert (args.output/target).is_file(), target
    # Transitive grouping must not split A--B--C when the direct A--C edge is absent.
    union = UnionFind(["A", "B", "C", "D"])
    union.union("A", "B")
    union.union("C", "B")
    assert union.find("A") == union.find("C") != union.find("D")
    # Identical image content must not act as a zero-distance distinct neighbor.
    x = np.array([[0], [0], [.5], [2]], dtype=np.float32)
    distances, neighbors = distinct_image_neighbors(x, np.array(["same", "same", "other", "third"]))
    assert np.allclose(distances[:2], [.5, .5]) and np.all(neighbors[:2] == 2)
    # Exercise the dense-image fallback where the first 32 neighbors share content.
    x = np.concatenate([np.zeros((40, 1)), [[1.]]]).astype(np.float32)
    distances, _ = distinct_image_neighbors(x, np.array(["same"]*40+["other"]))
    assert np.allclose(distances, 1)
    a = [{"category_id": 1, "bbox": [0., 0., 10., 10.], "segmentation": [], "iscrowd": 0}]
    b = [dict(a[0], category_id=2)]
    assert annotation_signature(a) != annotation_signature(b)
    print("PASS: counts, finite features, scoring formulas, all 5 fold partitions,")
    print("      source/object/pixel isolation, complete COCO annotations and paths,")
    print("      report links/images, transitive grouping and neighbor regression cases.")


if __name__ == "__main__":
    main()

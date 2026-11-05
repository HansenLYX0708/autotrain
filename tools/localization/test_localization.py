"""Contract checks without torch/GPU: python -m unittest discover -s tools/localization."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from prepare import corners, label_line, prepare, validate
from run import long_edge_pose


def fixture():
    obj = {
        "id": "one", "label": "slider", "kind": "obb",
        "cx": 50, "cy": 50, "width": 40, "height": 20, "angle": 30,
        "keypoints": {"hole": {"x": 40, "y": 45, "visibility": 2}, "notch": {"x": 0, "y": 0, "visibility": 0}},
    }
    return {
        "format": "autotrain-localization", "version": 1,
        "labels": ["slider"], "keypointNames": ["hole", "notch"],
        "images": [
            {"name": split + "/same.png", "width": 100, "height": 100, "objects": [copy.deepcopy(obj)],
             "reviewed": True, "split": split, "group": "batch-" + split}
            for split in ("train", "val", "test")
        ],
    }


class LocalizationTests(unittest.TestCase):
    def test_formats_and_direction(self):
        project = fixture()
        image, obj = project["images"][0], project["images"][0]["objects"][0]
        obb = list(map(float, label_line(obj, image, project["labels"], project["keypointNames"], "obb").split()))
        self.assertEqual(len(obb), 9)
        self.assertAlmostEqual(obb[1], (50 - 20 * math.cos(math.pi / 6) + 5) / 100)
        pose = list(map(float, label_line(obj, image, project["labels"], project["keypointNames"], "pose").split()))
        self.assertEqual(len(pose), 11)
        self.assertAlmostEqual(pose[3], (40 * math.cos(math.pi / 6) + 10) / 100)
        self.assertEqual(pose[-6:], [0.4, 0.45, 2, 0, 0, 0])
        result = long_edge_pose(corners(obj))
        self.assertAlmostEqual(result["angle_deg_mod180"], 30)
        self.assertAlmostEqual(result["long_side_px"], 40)
        obj["angle"] = -150
        self.assertAlmostEqual(long_edge_pose(corners(obj))["angle_deg_mod180"], 30)

    def test_reject_bad_annotations(self):
        cases = [
            lambda p: p["images"][0].update(reviewed=False),
            lambda p: p["images"][0].update(name="../escape.png"),
            lambda p: p["images"][1].update(group="batch-train"),
            lambda p: p["images"][0]["objects"][0].update(cx=-30),
            lambda p: p["images"][0]["objects"][0].update(angle=float("nan")),
            lambda p: p["images"][0]["objects"][0]["keypoints"].pop("hole"),
            lambda p: p["images"][0]["objects"][0]["keypoints"]["hole"].update(visibility=3),
            lambda p: p["images"][0]["objects"][0]["keypoints"]["hole"].update(visibility=0),
        ]
        for mutate in cases:
            with self.subTest(mutation=mutate):
                project = fixture()
                mutate(project)
                with self.assertRaises(ValueError):
                    validate(project)
        project = fixture()
        for image in project["images"]:
            image["objects"][0]["keypoints"] = {}
        validate(project, pose=False)

    def test_prepare_round_trip_and_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = fixture()
            for i, image in enumerate(project["images"]):
                source = root / "images" / image["name"]
                source.parent.mkdir(parents=True)
                Image.new("RGB", (100, 100), (i * 50, 20, 30)).save(source)
            # Explicit negative images must survive as empty label files.
            negative = copy.deepcopy(project["images"][0])
            negative.update(name="train/negative.png", objects=[])
            project["images"].append(negative)
            Image.new("RGB", (100, 100), (200, 200, 200)).save(root / "images/train/negative.png")
            source_project = root / "project.json"
            source_project.write_text(json.dumps(project), encoding="utf-8")
            out = prepare(source_project, root / "images", root / "output")
            for task in ("obb", "pose"):
                config = json.loads((out / task / "data.yaml").read_text(encoding="utf-8"))
                self.assertEqual(config["names"], {"0": "slider"})
                self.assertEqual(len(list((out / task / "images/train").glob("*"))), 2)
                self.assertEqual(sum(p.read_text() == "" for p in (out / task / "labels/train").glob("*.txt")), 1)
            self.assertEqual(json.loads((out / "pose/data.yaml").read_text())["kpt_shape"], [2, 3])
            self.assertNotIn("flip_idx", json.loads((out / "pose/data.yaml").read_text()))
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(source_project, root / "images", out)
            # A copied image renamed into another batch must not leak into validation.
            (root / "images/val/same.png").write_bytes((root / "images/train/same.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                prepare(source_project, root / "images", root / "leaked")
            self.assertFalse((root / "leaked").exists())
            Image.new("RGB", (101, 100)).save(root / "images/val/same.png")
            with self.assertRaisesRegex(ValueError, "dimensions"):
                prepare(source_project, root / "images", root / "wrong-size")


if __name__ == "__main__":
    unittest.main()

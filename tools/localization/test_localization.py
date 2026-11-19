"""Contract checks without torch/GPU: python -m unittest discover -s tools/localization."""
import copy
import json
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from prepare import corners, label_line, prepare, validate
from run import long_edge_pose
from torchtrain.localization import angle_diagnostic, angle_selection_score, axis_angle_penalty, align_rotated_targets, rotated_target_distances, long_edge


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
    def test_angle_selection_preserves_detection_coverage(self):
        geometry = {"targets": 65, "matched": 59, "falsePositives": 2,
                    "angleErrorDegMod180": {"count": 55, "mean": 1.5}}
        self.assertEqual(angle_selection_score(geometry, .9, 2), 1.5)
        self.assertIsNone(angle_selection_score(dict(geometry, matched=58), .9, 2))
        self.assertIsNone(angle_selection_score(dict(geometry, falsePositives=3), .9, 2))
        self.assertIsNone(angle_selection_score(dict(geometry, angleErrorDegMod180={"count": 0, "mean": None}), .9, 2))

    def test_angle_diagnostics_do_not_hide_square_predictions(self):
        target = {"corners_px": corners(fixture()["images"][0]["objects"][0])}
        target.update(long_edge(target["corners_px"]))
        predicted = dict(target, confidence=0.95, angle_deg_mod180=75, long_side_px=30, short_side_px=30)
        record = angle_diagnostic(target, predicted, "part.png", 0, 100, 100)
        self.assertTrue(record["predictionNearSquare"])
        self.assertTrue(record["includedInAngleMetric"])
        self.assertAlmostEqual(record["angleErrorDeg"], 45)
        target["angle_deg_mod180"], predicted["angle_deg_mod180"] = 179, 1
        self.assertAlmostEqual(angle_diagnostic(target, predicted, "part.png", 0, 100, 100)["angleErrorDeg"], 2)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("ultralytics"), "torch and ultralytics required for gradient checks")
    def test_axis_angle_penalty_periodicity_and_gradient(self):
        import torch
        predicted = torch.tensor([math.radians(10)], requires_grad=True)
        target = torch.zeros(1)
        penalty = axis_angle_penalty(predicted, target, torch.ones(1), torch.tensor(1.))
        penalty.backward()
        self.assertGreater(predicted.grad.item(), 0)
        for offset in (math.pi/2, math.pi):
            self.assertAlmostEqual(axis_angle_penalty(predicted.detach()+offset, target, torch.ones(1), 1).item(), penalty.item(), places=5)
        self.assertAlmostEqual(axis_angle_penalty(predicted.detach()+.7, target+.7, torch.ones(1), 1).item(), penalty.item(), places=5)
        # A square has no ProbIoU angle gradient, but the added term does.
        from ultralytics.utils.metrics import probiou
        angle = torch.tensor(.2, requires_grad=True)
        box = torch.stack([angle*0, angle*0, angle*0+20, angle*0+20, angle]).unsqueeze(0)
        ground_truth = torch.tensor([[0., 0., 20., 20., 0.]])
        (1-probiou(box, ground_truth)).sum().backward(retain_graph=True)
        self.assertAlmostEqual(angle.grad.item(), 0, places=5)
        angle.grad.zero_()
        axis_angle_penalty(angle, torch.tensor(0.), torch.tensor(1.), 1).backward()
        self.assertGreater(abs(angle.grad.item()), .5)
        diagonal = torch.tensor(math.pi/4, requires_grad=True)
        axis_angle_penalty(diagonal, torch.tensor(0.), torch.tensor(1.), 1).backward()
        self.assertGreater(abs(diagonal.grad.item()), 1.9)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("ultralytics"), "torch and ultralytics required")
    def test_rotated_representation_and_dfl_round_trip(self):
        import torch
        from ultralytics.utils.ops import xywhr2xyxyxyxy
        from ultralytics.utils.tal import dist2rbox, bbox2dist
        targets = torch.tensor([[10., 8., 6., 4., math.pi/3], [20., 20., 4., 8., -.01]])
        predictions = targets.clone()
        predictions[0, 4] -= math.pi/2
        predictions[1, 4] += 3*math.pi
        aligned = align_rotated_targets(predictions, targets)
        self.assertLess((aligned[:, 4]-predictions[:, 4]).abs().max().item(), math.pi/4)
        before, after = xywhr2xyxyxyxy(targets), xywhr2xyxyxyxy(aligned)
        errors = torch.stack([(before-after.roll(k, dims=1)).abs().amax(dim=(1,2)) for k in range(4)])
        self.assertLess(errors.amin(dim=0).max().item(), 1e-5)
        anchors = torch.tensor([[9., 8.], [20.5, 20.3]])
        for encoded in (targets, aligned):
            distances = rotated_target_distances(anchors, encoded)
            decoded = dist2rbox(distances, encoded[:, 4:], anchors)
            torch.testing.assert_close(decoded, encoded[:, :4], atol=1e-5, rtol=1e-5)
        # Axis-aligned distances are not the inverse of rotated decoding.
        from ultralytics.utils.ops import xywh2xyxy
        old = bbox2dist(anchors, xywh2xyxy(targets[:, :4]), 16)
        self.assertGreater((dist2rbox(old, targets[:, 4:], anchors)[:, :2]-targets[:, :2]).norm().item(), .5)

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

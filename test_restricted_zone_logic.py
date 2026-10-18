import unittest
import numpy as np
import importlib

# Import pipeline class
pipeline_mod = importlib.import_module("05_live_pipeline_run")
UniversalRestrictedZoneSentinel = pipeline_mod.UniversalRestrictedZoneSentinel

class TestRestrictedZoneLogic(unittest.TestCase):
    def setUp(self):
        # Hazard zone: [600, 250] to [1150, 650]
        self.hazard_zone = np.array([[600, 250], [1150, 250], [1150, 650], [600, 650]], np.int32)
        # Dummy sentinel instance (local only, no mqtt)
        self.sentinel = UniversalRestrictedZoneSentinel.__new__(UniversalRestrictedZoneSentinel)
        self.sentinel.hazard_zone = self.hazard_zone
        self.sentinel.target_states = {}

    def test_case_01_worker_completely_outside(self):
        """Case 1: Worker completely outside the zone -> SAFE, no zone-breach alert."""
        # Worker at x=150..210, y=350..490 (zone starts at x=600)
        bbox = (150, 350, 210, 490)
        state, dist = self.sentinel.evaluate_bbox_zone(bbox, self.hazard_zone)
        self.assertEqual(state, "SAFE")
        self.assertGreater(dist, 0.0)
        self.assertFalse(self.sentinel.does_bbox_intersect_zone(bbox, self.hazard_zone))

    def test_case_02_cargo_completely_outside(self):
        """Case 2: Cargo completely outside the zone -> SAFE, no zone-breach alert."""
        # Cargo at x=340..405, y=440..495
        bbox = (340, 440, 405, 495)
        state, dist = self.sentinel.evaluate_bbox_zone(bbox, self.hazard_zone)
        self.assertEqual(state, "SAFE")
        self.assertGreater(dist, 0.0)

    def test_case_03_worker_near_boundary_outside(self):
        """Case 3: Worker near the boundary but outside -> SAFE, no zone-breach alert."""
        # Worker right edge at x=598 (zone starts at x=600)
        bbox = (540, 300, 598, 440)
        state, dist = self.sentinel.evaluate_bbox_zone(bbox, self.hazard_zone)
        self.assertEqual(state, "SAFE")
        self.assertAlmostEqual(dist, 2.0, places=1)

    def test_case_04_worker_partially_entering_zone(self):
        """Case 4: Worker partially entering the zone -> BREACH."""
        # Worker crosses boundary: x=580..620 (zone starts at 600)
        bbox = (580, 300, 620, 440)
        state, dist = self.sentinel.evaluate_bbox_zone(bbox, self.hazard_zone)
        self.assertEqual(state, "BREACH")
        self.assertEqual(dist, 0.0)
        self.assertTrue(self.sentinel.does_bbox_intersect_zone(bbox, self.hazard_zone))

    def test_case_05_worker_fully_inside_zone(self):
        """Case 5: Worker fully inside the zone -> BREACH."""
        bbox = (700, 300, 760, 440)
        state, dist = self.sentinel.evaluate_bbox_zone(bbox, self.hazard_zone)
        self.assertEqual(state, "BREACH")
        self.assertEqual(dist, 0.0)

    def test_case_06_cargo_entering_zone(self):
        """Case 6: Cargo entering or overlapping the zone -> BREACH."""
        bbox = (800, 400, 880, 470)
        state, dist = self.sentinel.evaluate_bbox_zone(bbox, self.hazard_zone)
        self.assertEqual(state, "BREACH")
        self.assertEqual(dist, 0.0)

    def test_case_07_object_leaving_zone_restores_safe(self):
        """Case 7: Object leaving the zone -> Clear active breach and restore SAFE after confirmation."""
        target_id = "test_obj_1"
        # Enter zone for 2 frames
        self.sentinel.update_target_state(target_id, "BREACH", min_confirm_frames=2)
        confirmed = self.sentinel.update_target_state(target_id, "BREACH", min_confirm_frames=2)
        self.assertEqual(confirmed, "BREACH")

        # Leave zone for 2 frames
        self.sentinel.update_target_state(target_id, "SAFE", min_confirm_frames=2)
        confirmed_cleared = self.sentinel.update_target_state(target_id, "SAFE", min_confirm_frames=2)
        self.assertEqual(confirmed_cleared, "SAFE")

    def test_case_08_moving_across_boundary_realtime(self):
        """Case 8: Object moving across the boundary updates in real time."""
        # Step 1: Outside (x=550)
        s1, _ = self.sentinel.evaluate_bbox_zone((500, 300, 560, 440), self.hazard_zone)
        self.assertEqual(s1, "SAFE")

        # Step 2: Overlapping (x=610)
        s2, _ = self.sentinel.evaluate_bbox_zone((550, 300, 610, 440), self.hazard_zone)
        self.assertEqual(s2, "BREACH")

        # Step 3: Outside again (x=540)
        s3, _ = self.sentinel.evaluate_bbox_zone((480, 300, 540, 440), self.hazard_zone)
        self.assertEqual(s3, "SAFE")

    def test_case_09_invalid_or_missing_coordinates(self):
        """Case 9: Object with invalid or missing coordinates -> UNKNOWN."""
        s1, d1 = self.sentinel.evaluate_bbox_zone(None, self.hazard_zone)
        self.assertEqual(s1, "UNKNOWN")
        self.assertIsNone(d1)

        s2, d2 = self.sentinel.evaluate_bbox_zone((np.nan, 100, 200, 300), self.hazard_zone)
        self.assertEqual(s2, "UNKNOWN")
        self.assertIsNone(d2)

        # Inverted box (x2 <= x1)
        s3, d3 = self.sentinel.evaluate_bbox_zone((200, 100, 150, 300), self.hazard_zone)
        self.assertEqual(s3, "UNKNOWN")
        self.assertIsNone(d3)

    def test_case_10_multiple_objects_evaluated_independently(self):
        """Case 10: Multiple objects in different positions evaluated independently."""
        objects = [
            ("worker_1", (150, 350, 210, 490)), # Outside (Safe)
            ("worker_2", (700, 300, 760, 440)), # Inside (Breach)
            ("cargo_1",  (300, 400, 360, 460)), # Outside (Safe)
            ("vehicle_1",(650, 500, 750, 580)), # Inside (Breach)
        ]
        results = [self.sentinel.evaluate_bbox_zone(bbox, self.hazard_zone)[0] for _, bbox in objects]
        self.assertEqual(results, ["SAFE", "BREACH", "SAFE", "BREACH"])

if __name__ == "__main__":
    unittest.main()

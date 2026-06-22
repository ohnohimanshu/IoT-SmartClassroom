"""Unit tests for behavior classification heuristics (synthetic keypoints)."""

import numpy as np
from django.test import SimpleTestCase

from classroom_monitor.behavior_detection import TemporalBehaviorEngine, TrackedPerson


def _blank_kp(n=17):
    """COCO pose keypoints: (x, y, conf) per joint."""
    return np.zeros((n, 3), dtype=np.float32)


def _person(track_id, bbox, keypoints=None):
    return TrackedPerson(
        track_id=track_id,
        bbox=bbox,
        keypoints=keypoints,
        last_seen=0.0,
    )


class PhoneDetectionTests(SimpleTestCase):
    def setUp(self):
        self.engine = TemporalBehaviorEngine()

    def test_yolo_phone_requires_wrist_proximity_not_bbox_only(self):
        """Phone bbox inside person without wrist near phone must NOT alert."""
        bbox = (100, 50, 200, 350)
        kp = _blank_kp()
        kp[9] = [120, 300, 0.9]
        kp[10] = [180, 300, 0.9]
        kp[0] = [150, 80, 0.9]
        person = _person(1, bbox, kp)
        phone_dets = [(130, 200, 170, 240, 0.85)]  # phone mid-torso, not near wrists
        is_phone, conf = self.engine._detect_phone_usage(
            person, phone_dets, 'focused', [])
        self.assertFalse(is_phone)
        self.assertEqual(conf, 0.0)

    def test_single_hand_at_notebook_not_phone(self):
        """One hand at lap writing in notebook — must NOT trigger phone."""
        bbox = (0, 0, 100, 200)
        kp = _blank_kp()
        kp[0] = [50, 75, 0.9]    # nose pitched down
        kp[1] = [42, 65, 0.9]
        kp[2] = [58, 65, 0.9]
        kp[9] = [45, 140, 0.9]   # one wrist on notebook (desk level)
        person = _person(1, bbox, kp)
        is_phone, _ = self.engine._detect_phone_usage(person, [], 'head_down', [])
        self.assertFalse(is_phone)

    def test_single_hand_mid_torso_phone_detected(self):
        """One hand centered at mid-torso + head down — typical phone use."""
        bbox = (0, 0, 100, 200)
        kp = _blank_kp()
        kp[0] = [50, 75, 0.9]
        kp[1] = [42, 65, 0.9]
        kp[2] = [58, 65, 0.9]
        kp[9] = [50, 110, 0.9]   # mid-torso, centered
        person = _person(1, bbox, kp)
        is_phone, conf = self.engine._detect_phone_usage(person, [], 'head_down', [])
        self.assertTrue(is_phone)
        self.assertGreater(conf, 0.5)

    def test_two_close_hands_at_desk_not_phone(self):
        """Two close hands on desk writing — must NOT trigger cupped-phone heuristic."""
        bbox = (0, 0, 100, 200)
        kp = _blank_kp()
        kp[0] = [50, 75, 0.9]
        kp[1] = [42, 65, 0.9]
        kp[2] = [58, 65, 0.9]
        kp[9] = [46, 145, 0.9]
        kp[10] = [54, 147, 0.9]  # spread 0.04 but on desk
        person = _person(1, bbox, kp)
        is_phone, _ = self.engine._detect_phone_usage(person, [], 'head_down', [])
        self.assertFalse(is_phone)

    def test_cupped_hands_at_lap_with_head_down_is_phone(self):
        bbox = (0, 0, 100, 200)
        kp = _blank_kp()
        kp[9] = [45, 128, 0.9]   # mid-lap cupped phone (not desk level)
        kp[10] = [55, 130, 0.9]
        kp[0] = [50, 70, 0.9]
        person = _person(1, bbox, kp)
        is_phone, conf = self.engine._detect_phone_usage(person, [], 'head_down', [])
        self.assertTrue(is_phone)
        self.assertGreater(conf, 0.5)


class FightDetectionTests(SimpleTestCase):
    def setUp(self):
        self.engine = TemporalBehaviorEngine()

    def test_passing_paper_reach_does_not_confirm_fight(self):
        """Wrist enters neighbor bbox with moderate motion — below threshold."""
        a_bbox = (0, 0, 80, 200)
        b_bbox = (70, 0, 150, 200)
        kp_a = _blank_kp()
        kp_b = _blank_kp()
        # A's right wrist reaches into B's bbox
        kp_a[10] = [100, 100, 0.9]
        kp_b[9] = [110, 120, 0.9]
        kp_b[10] = [130, 120, 0.9]

        pa = _person(1, a_bbox, kp_a)
        pb = _person(2, b_bbox, kp_b)
        scale = 100.0

        # Simulate moderate motion history
        import time
        t0 = time.time()
        for i, wx in enumerate([85, 90, 95, 100]):
            kpa = kp_a.copy()
            kpa[10] = [wx, 100, 0.9]
            pa.keypoint_history.append((t0 + i * 0.1, kpa))

        score = self.engine._skeleton_fight_score(pa, pb)
        self.assertLess(score, self.engine.FIGHT_SCORE_THRESHOLD)

    def test_mutual_wrist_proximity_and_fast_motion_scores_high(self):
        a_bbox = (0, 0, 80, 200)
        b_bbox = (60, 0, 140, 200)
        kp_a = _blank_kp()
        kp_b = _blank_kp()
        kp_a[9] = [95, 100, 0.9]
        kp_a[10] = [98, 105, 0.9]
        kp_b[9] = [100, 102, 0.9]
        kp_b[10] = [103, 108, 0.9]

        pa = _person(1, a_bbox, kp_a)
        pb = _person(2, b_bbox, kp_b)

        import time
        t0 = time.time()
        for i in range(5):
            kpa = kp_a.copy()
            kpb = kp_b.copy()
            kpa[9] = [95 + i * 25, 100 + i * 10, 0.9]
            kpa[10] = [98 + i * 25, 105 + i * 10, 0.9]
            kpb[9] = [100 + i * 25, 102 + i * 10, 0.9]
            kpb[10] = [103 + i * 25, 108 + i * 10, 0.9]
            pa.keypoint_history.append((t0 + i * 0.1, kpa))
            pb.keypoint_history.append((t0 + i * 0.1, kpb))

        score = self.engine._skeleton_fight_score(pa, pb)
        self.assertGreaterEqual(score, self.engine.FIGHT_SCORE_THRESHOLD)

    def test_grappling_elbow_contact_scores_high(self):
        """Interlocked arms: wrist-to-elbow contacts + overlapping bboxes."""
        a_bbox = (0, 0, 90, 200)
        b_bbox = (50, 0, 140, 200)
        kp_a = _blank_kp()
        kp_b = _blank_kp()
        kp_a[9] = [80, 90, 0.9]
        kp_a[10] = [85, 95, 0.9]
        kp_a[7] = [75, 80, 0.9]
        kp_b[9] = [82, 92, 0.9]
        kp_b[8] = [78, 85, 0.9]

        pa = _person(1, a_bbox, kp_a)
        pb = _person(2, b_bbox, kp_b)

        import time
        t0 = time.time()
        for i in range(4):
            kpa = kp_a.copy()
            kpb = kp_b.copy()
            kpa[9] = [80 + i * 3, 90 + i * 2, 0.9]
            kpb[9] = [82 + i * 3, 92 + i * 2, 0.9]
            pa.keypoint_history.append((t0 + i * 0.12, kpa))
            pb.keypoint_history.append((t0 + i * 0.12, kpb))

        score = self.engine._skeleton_fight_score(pa, pb)
        self.assertGreaterEqual(score, self.engine.FIGHT_SCORE_THRESHOLD)


class HeadPoseTests(SimpleTestCase):
    def setUp(self):
        self.engine = TemporalBehaviorEngine()

    def test_single_visible_eye_profile_not_auto_distracted(self):
        bbox = (0, 0, 100, 200)
        kp = _blank_kp()
        kp[0] = [50, 60, 0.9]
        kp[1] = [40, 50, 0.9]
        kp[2] = [0, 0, 0.1]
        person = _person(1, bbox, kp)
        pose = self.engine._calculate_head_pose(person)
        self.assertEqual(pose, 'focused')


class IncidentSeverityTests(SimpleTestCase):
    def test_fight_is_critical(self):
        from classroom_monitor.views import _incident_severity
        self.assertEqual(_incident_severity('fighting'), 'critical')

    def test_phone_is_high(self):
        from classroom_monitor.views import _incident_severity
        self.assertEqual(_incident_severity('using_phone'), 'high')

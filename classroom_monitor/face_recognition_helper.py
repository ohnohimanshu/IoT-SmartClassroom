"""
Face Recognition Helper
========================
Thread-safety note:
  dlib (which face_recognition uses internally) is NOT thread-safe on Windows.
  Always acquire DLIB_LOCK before calling any method on this class.

  from classroom_monitor.face_recognition_helper import StudentFaceRecognizer, DLIB_LOCK
  with DLIB_LOCK:
      sid, name, roll, dist = recognizer.match(crop)
"""

import json
import threading
import numpy as np

# ── Single process-wide lock for ALL dlib / face_recognition calls ────────────
# This prevents the 0xC0000005 access violation crash on Windows.
DLIB_LOCK = threading.Lock()


class StudentFaceRecognizer:
    """
    Match a face crop (BGR numpy array) to registered students.
    All public methods are protected by DLIB_LOCK internally.
    """

    def __init__(self, tolerance=0.52):
        self.tolerance        = tolerance
        self._known_encodings = []
        self._known_students  = []
        self._fr              = None
        self._loaded          = False

    def load_from_db(self):
        """Load student face encodings from DB. Call once at startup."""
        # Import face_recognition inside the lock so dlib initialises safely
        with DLIB_LOCK:
            try:
                import face_recognition as fr
                self._fr = fr
            except ImportError:
                print('[WARN] face_recognition not installed')
                self._loaded = True
                return

        try:
            from entrance_cam.models import Student
        except ImportError:
            print('[WARN] entrance_cam.models not available')
            self._loaded = True
            return

        students = (Student.objects
                    .filter(is_active=True, face_encoding__isnull=False)
                    .exclude(face_encoding=''))

        encs, studs = [], []
        for s in students:
            try:
                enc = np.array(json.loads(s.face_encoding))
                if enc.shape == (128,):
                    encs.append(enc)
                    studs.append({
                        'id':       s.id,
                        'name':     s.name,
                        'roll_no':  s.roll_no,
                        'whatsapp': (getattr(s, 'parent_whatsapp', '')
                                     or getattr(s, 'parent_phone', '') or ''),
                    })
            except Exception as e:
                print(f'[WARN] Bad encoding for {s.name}: {e}')

        self._known_encodings = encs
        self._known_students  = studs
        self._loaded = True
        print(f'[OK] Loaded {len(encs)} student face encodings')

    def match(self, face_crop_bgr):
        """
        Match crop to a known student.
        Acquires DLIB_LOCK internally — safe to call from any single thread,
        but do NOT call from multiple threads concurrently (use a queue instead).

        Returns (student_id, name, roll_no, distance)
        """
        if not self._loaded:
            self.load_from_db()

        if not self._known_encodings or self._fr is None:
            return None, 'Unknown', '', 1.0

        with DLIB_LOCK:
            try:
                import cv2
                rgb      = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                encodings = self._fr.face_encodings(rgb, num_jitters=1, model='small')
                if not encodings:
                    return None, 'Unknown', '', 1.0

                detected  = encodings[0]
                distances = self._fr.face_distance(self._known_encodings, detected)
                best_idx  = int(np.argmin(distances))
                best_dist = float(distances[best_idx])

                if best_dist < self.tolerance:
                    s = self._known_students[best_idx]
                    return s['id'], s['name'], s['roll_no'], best_dist

                return None, 'Unknown', '', best_dist

            except Exception as e:
                print(f'[ERROR] face match: {e}')
                return None, 'Unknown', '', 1.0

    def get_whatsapp(self, student_id):
        for s in self._known_students:
            if s['id'] == student_id:
                return s.get('whatsapp', '')
        return ''
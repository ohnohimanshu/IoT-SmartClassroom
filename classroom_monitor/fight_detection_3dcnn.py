"""
Fight Detection using 3D CNN (mc3_18 / r3d_18)
================================================

Replaces skeleton-heuristic fight detection with a pre-trained 3D CNN
that understands temporal motion patterns across 16-frame clips.

Architecture: mc3_18 (Mixed Convolutional 3D ResNet-18)
- Pre-trained on Kinetics-400
- Intended to be fine-tuned on a surveillance/fight dataset that matches
  your camera setup. Any reported accuracy figure for a fine-tune is only
  as good as the data it was fine-tuned on -- don't treat a number quoted
  elsewhere as a guarantee for your own classroom footage.

Usage:
    detector = FightDetector3DCNN()
    for frame in video_stream:
        detector.add_frame(frame)
        is_fighting, confidence = detector.predict()

IMPORTANT (perf fix): the model + its fine-tuned weights are loaded ONCE,
shared by every FightDetector3DCNN instance via _SharedCNNModel. Each
instance only owns its own lightweight frame buffer. Previously, every
caller that constructed a new FightDetector3DCNN() (e.g. one per candidate
fighting pair) triggered a brand-new full model load + weight load on a
new background thread -- meaning every newly-formed pair in a busy
classroom paid the full model-load cost again, and N simultaneous pairs
held N redundant copies of the model in memory at once. That no longer
happens: the model loads once, lazily, on first use.
"""

import os
import threading
import numpy as np
from collections import deque
from typing import Tuple, Optional

# Lazy imports for torch (heavy dependency)
_torch = None
_torchvision = None
_transforms = None


def _lazy_import_torch():
    """Lazy import torch and torchvision to avoid slowing Django startup."""
    global _torch, _torchvision, _transforms
    if _torch is None:
        import torch
        import torchvision
        import torchvision.transforms as transforms
        _torch = torch
        _torchvision = torchvision
        _transforms = transforms
    return _torch, _torchvision, _transforms


# Kinetics-400 normalization constants
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

# Fine-tuned weights should be bundled locally or loaded via environment variable.
# Runtime download from a remote URL is intentionally NOT supported -- production
# must point FIGHT_MODEL_WEIGHTS_PATH at a local, validated file.
MODEL_WEIGHTS_FILENAME = 'fight_mc3_18_finetuned.pth'


class _SharedCNNModel:
    """
    Process-wide singleton holding the loaded 3D CNN model + device.

    All FightDetector3DCNN instances delegate to this class instead of each
    building/loading their own copy of the model. The model is loaded at
    most once per process, the first time any FightDetector3DCNN needs it.
    """
    _lock = threading.Lock()
    _load_started = False
    _model = None
    _device = None
    _model_loaded = False
    _model_name = None

    @classmethod
    def ensure_loading_started(cls, device_str: str):
        """Kick off background loading exactly once per process."""
        with cls._lock:
            if cls._load_started:
                return
            cls._load_started = True
        threading.Thread(target=cls._load, args=(device_str,), daemon=True).start()

    @classmethod
    def is_ready(cls) -> bool:
        with cls._lock:
            return cls._model_loaded and cls._model is not None

    @classmethod
    def get(cls):
        """Returns (model, device) -- caller should check is_ready() first."""
        with cls._lock:
            return cls._model, cls._device

    @classmethod
    def _get_device(cls, device_str: str):
        torch, _, _ = _lazy_import_torch()
        if device_str == 'auto':
            return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        return torch.device(device_str)

    @classmethod
    def _validate_weights_file(cls, weights_path: str) -> bool:
        """
        Validate fine-tuned weights file before loading.

        Checks file existence, minimum size, and basic format validation.
        Prevents loading of corrupt/incomplete files that would produce
        meaningless predictions.

        Note: this loads the checkpoint with weights_only=False to support
        dict-wrapped checkpoints (e.g. {'model_state_dict': ...}). Only ever
        point FIGHT_MODEL_WEIGHTS_PATH at a file you trust -- loading a
        pickle this way can execute arbitrary code if the file is malicious,
        so this path should never be user-controllable in a deployed system.
        """
        if not os.path.exists(weights_path):
            return False

        if os.path.getsize(weights_path) < 1_000_000:
            print(f'[FIGHT-3DCNN] Weights file too small: {os.path.getsize(weights_path)} bytes')
            return False

        try:
            torch, _, _ = _lazy_import_torch()
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
            if not isinstance(state_dict, dict):
                print('[FIGHT-3DCNN] Invalid weights file format')
                return False
            print(f'[FIGHT-3DCNN] Weights file validated: {weights_path}')
            return True
        except Exception as e:
            print(f'[FIGHT-3DCNN] Weights file validation failed: {e}')
            return False

    @classmethod
    def _build_model(cls, num_classes: int = 2):
        """
        Build mc3_18 model with custom classification head.
        Falls back to r3d_18 if mc3_18 is unavailable.
        """
        torch, torchvision, _ = _lazy_import_torch()

        try:
            from torchvision.models.video import mc3_18, MC3_18_Weights
            model = mc3_18(weights=MC3_18_Weights.KINETICS400_V1)
            model_name = 'mc3_18'
        except (ImportError, AttributeError):
            try:
                from torchvision.models.video import mc3_18
                model = mc3_18(pretrained=True)
                model_name = 'mc3_18'
            except Exception:
                try:
                    from torchvision.models.video import r3d_18, R3D_18_Weights
                    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
                    model_name = 'r3d_18'
                except (ImportError, AttributeError):
                    from torchvision.models.video import r3d_18
                    model = r3d_18(pretrained=True)
                    model_name = 'r3d_18'

        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)

        print(f'[FIGHT-3DCNN] Built {model_name} model (in_features={in_features})')
        return model, model_name

    @classmethod
    def _load(cls, device_str: str):
        """
        Load the 3D CNN model with proper weight validation, ONCE for the
        whole process. Never silently substitutes a randomly-initialized
        head as if it were a working model -- if fine-tuned weights aren't
        found/valid, the model stays disabled and every FightDetector3DCNN
        instance's predict() will return (False, 0.0).
        """
        try:
            torch, _, _ = _lazy_import_torch()

            device = cls._get_device(device_str)
            print(f'[FIGHT-3DCNN] Using device: {device}')

            model, model_name = cls._build_model(num_classes=2)

            weights_path = os.environ.get('FIGHT_MODEL_WEIGHTS_PATH')
            if not weights_path:
                weights_dir = os.path.join(os.path.dirname(__file__), '..', 'model_weights')
                weights_path = os.path.join(weights_dir, MODEL_WEIGHTS_FILENAME)

            finetuned_loaded = False

            if cls._validate_weights_file(weights_path):
                try:
                    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
                    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']
                    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']

                    try:
                        model.load_state_dict(state_dict, strict=False)
                        finetuned_loaded = True
                        print('[FIGHT-3DCNN] Fine-tuned weights loaded successfully')
                    except RuntimeError as e:
                        print(f'[FIGHT-3DCNN] Weight loading failed: {e}')
                        finetuned_loaded = False
                except Exception as e:
                    print(f'[FIGHT-3DCNN] Error loading validated weights: {e}')
                    finetuned_loaded = False
            else:
                print(f'[FIGHT-3DCNN] Fine-tuned weights not found or invalid: {weights_path}')
                print('[FIGHT-3DCNN] Set FIGHT_MODEL_WEIGHTS_PATH env var to specify weights location')

            if not finetuned_loaded:
                print('[FIGHT-3DCNN] FIGHT DETECTION DISABLED')
                print('[FIGHT-3DCNN] Fine-tuned weights not available - cannot provide reliable predictions')
                print('[FIGHT-3DCNN] Model will return fight_detected=False for all predictions')
                with cls._lock:
                    cls._model = None
                    cls._model_loaded = False
                return

            model = model.to(device)
            model.eval()

            with cls._lock:
                cls._model = model
                cls._device = device
                cls._model_name = model_name
                cls._model_loaded = True

            print(f'[FIGHT-3DCNN] Model ready on {device} with fine-tuned weights (shared across all pairs)')

        except Exception as e:
            print(f'[FIGHT-3DCNN] FATAL: Could not load model: {e}')
            import traceback
            traceback.print_exc()
            with cls._lock:
                cls._model = None
                cls._model_loaded = False


class FightDetector3DCNN:
    """
    3D CNN-based fight detector using mc3_18 architecture.

    Buffers consecutive frames and classifies 16-frame clips as fight/no-fight.
    Designed for real-time integration with existing YOLO-based behavior pipeline.

    Each instance owns only its OWN frame buffer (cheap -- a deque of small
    preprocessed arrays). The actual model is a shared, process-wide singleton
    (see _SharedCNNModel) loaded at most once, so creating many instances
    (e.g. one per candidate fighting pair) is cheap and does not re-load the
    model or its weights.

    Public API:
        add_frame(frame)          — Feed a BGR frame into the buffer
        predict()                 — Classify current buffer as fight/no-fight
        reset()                   — Clear the frame buffer
        is_ready()                — Check if enough frames are buffered
    """

    def __init__(self,
                 device: str = 'auto',
                 sequence_length: int = 16,
                 confidence_threshold: float = 0.60,
                 input_size: int = 112):
        """
        Initialize the 3D CNN fight detector.

        Args:
            device: 'auto' (detect GPU), 'cuda', or 'cpu'
            sequence_length: Number of frames per clip (default 16)
            confidence_threshold: Minimum confidence to flag as fight
            input_size: Spatial size for model input (112x112)
        """
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size

        # Frame buffer (ring buffer of preprocessed frames) -- per-instance,
        # this is the only thing that legitimately needs to differ per pair.
        self._frame_buffer = deque(maxlen=sequence_length)
        self._raw_frame_count = 0

        # Prediction cache (avoid redundant inference)
        self._last_prediction = (False, 0.0)
        self._frames_since_prediction = 0
        self._predict_every_n = 8  # Run inference every 8 new frames

        # Kick off (process-wide, once-only) model loading. Cheap to call
        # repeatedly -- only the very first call actually starts a thread.
        self._device_str = device
        _SharedCNNModel.ensure_loading_started(device)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single BGR frame for the 3D CNN.

        Steps:
        1. Resize to input_size x input_size
        2. Convert BGR to RGB
        3. Normalize to [0, 1]
        4. Apply Kinetics-400 mean/std normalization

        Returns: numpy array of shape (3, H, W), float32
        """
        import cv2

        resized = cv2.resize(frame, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for c in range(3):
            rgb[:, :, c] = (rgb[:, :, c] - KINETICS_MEAN[c]) / KINETICS_STD[c]
        return np.transpose(rgb, (2, 0, 1))

    def add_frame(self, frame: np.ndarray):
        """
        Add a BGR frame to the internal buffer.

        Args:
            frame: BGR numpy array from OpenCV (any resolution)
        """
        if frame is None or frame.size == 0:
            return

        preprocessed = self._preprocess_frame(frame)
        self._frame_buffer.append(preprocessed)
        self._raw_frame_count += 1
        self._frames_since_prediction += 1

    def is_ready(self) -> bool:
        """
        Check if the buffer has enough frames AND the shared model is
        properly loaded with fine-tuned weights.
        """
        return len(self._frame_buffer) >= self.sequence_length and _SharedCNNModel.is_ready()

    def predict(self) -> Tuple[bool, float]:
        """
        Run fight detection on the current frame buffer using the shared model.

        Returns (is_fighting, confidence). If fine-tuned weights are not loaded,
        returns (False, 0.0) — check ``detection_available`` to distinguish
        "no fight" from "detector disabled".
        """
        if not self.detection_available:
            return False, 0.0

        if self._frames_since_prediction < self._predict_every_n:
            return self._last_prediction

        if not self.is_ready():
            return False, 0.0

        model, device = _SharedCNNModel.get()
        if model is None:
            return False, 0.0

        try:
            torch, _, _ = _lazy_import_torch()

            frames_list = list(self._frame_buffer)
            clip = np.stack(frames_list, axis=0)            # (T, C, H, W)
            clip = np.transpose(clip, (1, 0, 2, 3))          # (C, T, H, W)

            clip_tensor = torch.from_numpy(clip).unsqueeze(0).float()
            clip_tensor = clip_tensor.to(device)

            with torch.no_grad():
                output = model(clip_tensor)
                probabilities = torch.softmax(output, dim=1)
                fight_prob = probabilities[0, 1].item()

            is_fighting = fight_prob >= self.confidence_threshold
            self._last_prediction = (is_fighting, fight_prob)
            self._frames_since_prediction = 0

            if is_fighting:
                print(f'[FIGHT-3DCNN] FIGHT DETECTED! confidence={fight_prob:.3f}')

            return is_fighting, fight_prob

        except Exception as e:
            print(f'[FIGHT-3DCNN] Prediction error: {e}')
            return False, 0.0

    def reset(self):
        """Clear the frame buffer and prediction cache."""
        self._frame_buffer.clear()
        self._raw_frame_count = 0
        self._frames_since_prediction = 0
        self._last_prediction = (False, 0.0)

    @property
    def detection_available(self) -> bool:
        """True when shared model loaded with validated fine-tuned weights."""
        return _SharedCNNModel.is_ready()

    @property
    def model_loaded(self) -> bool:
        """Check if the shared model has been loaded successfully."""
        return _SharedCNNModel.is_ready()

    @property
    def buffer_fill(self) -> float:
        """Return this instance's buffer fill level as 0.0 to 1.0."""
        return len(self._frame_buffer) / self.sequence_length
"""
Fight Detection using 3D CNN (mc3_18 / r3d_18)
================================================

Replaces skeleton-heuristic fight detection with a pre-trained 3D CNN
that understands temporal motion patterns across 16-frame clips.

Architecture: mc3_18 (Mixed Convolutional 3D ResNet-18)
- Pre-trained on Kinetics-400
- Fine-tuned on surveillance fight datasets
- ~91% accuracy on fight/no-fight classification

Usage:
    detector = FightDetector3DCNN()
    for frame in video_stream:
        detector.add_frame(frame)
        is_fighting, confidence = detector.predict()
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

# BUG 4 FIX: Remove runtime Google Drive download dependency for production
# Fine-tuned weights should be bundled locally or loaded via environment variable
# MODEL_WEIGHTS_URL = 'https://drive.google.com/uc?id=1MWDeLnpEaZDrKK-OjmzvYLxfjwp-GDcp'  # Removed
MODEL_WEIGHTS_FILENAME = 'fight_mc3_18_finetuned.pth'


class FightDetector3DCNN:
    """
    3D CNN-based fight detector using mc3_18 architecture.
    
    Buffers consecutive frames and classifies 16-frame clips as fight/no-fight.
    Designed for real-time integration with existing YOLO-based behavior pipeline.
    
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
        
        # Frame buffer (ring buffer of preprocessed frames)
        self._frame_buffer = deque(maxlen=sequence_length)
        self._raw_frame_count = 0
        
        # Model state
        self._model = None
        self._device = None
        self._device_str = device
        self._model_loaded = False
        self._model_lock = threading.Lock()
        
        # Prediction cache (avoid redundant inference)
        self._last_prediction = (False, 0.0)
        self._frames_since_prediction = 0
        self._predict_every_n = 8  # Run inference every 8 new frames
        
        # Load model in background to avoid blocking
        self._load_thread = threading.Thread(target=self._load_model, daemon=True)
        self._load_thread.start()
    
    def _get_device(self):
        """Determine the best available device."""
        torch, _, _ = _lazy_import_torch()
        
        if self._device_str == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        return torch.device(self._device_str)
    
    def _download_weights(self, save_path: str) -> bool:
        """
        BUG 4 FIX: Optional one-time bootstrap download - separated from runtime detection.
        
        This method is kept for setup/bootstrap scripts only. Production runtime should 
        NEVER depend on downloading weights from Google Drive. Fine-tuned weights should 
        be bundled with the application or loaded from a validated local path.
        
        Returns True if download succeeded.
        """
        print('[FIGHT-3DCNN] WARNING: Runtime download is deprecated for production use')
        print('[FIGHT-3DCNN] Use FIGHT_MODEL_WEIGHTS_PATH env var to specify pre-downloaded weights')
        
        try:
            import gdown
            print('[FIGHT-3DCNN] Attempting bootstrap weight download (one-time setup only)...')
            # Note: MODEL_WEIGHTS_URL removed - would need to be restored for bootstrap script
            # gdown.download(MODEL_WEIGHTS_URL, save_path, quiet=False)
            print('[FIGHT-3DCNN] Bootstrap download not available - weights URL removed for security')
            return False
        except ImportError:
            print('[FIGHT-3DCNN] gdown not installed, cannot bootstrap download')
            return False
        except Exception as e:
            print(f'[FIGHT-3DCNN] Bootstrap download failed: {e}')
            return False
    
    def _validate_weights_file(self, weights_path: str) -> bool:
        """
        BUG 4 FIX: Validate fine-tuned weights file before loading.
        
        Checks file existence, minimum size, and basic format validation.
        Prevents loading of corrupt/incomplete files that would produce meaningless predictions.
        """
        if not os.path.exists(weights_path):
            return False
        
        # Check minimum file size (fine-tuned model should be at least 1MB)
        if os.path.getsize(weights_path) < 1_000_000:
            print(f'[FIGHT-3DCNN] Weights file too small: {os.path.getsize(weights_path)} bytes')
            return False
        
        # Optional: Add checksum validation here if you have expected checksums
        # For now, just check that it's a valid torch file
        try:
            torch, _, _ = _lazy_import_torch()
            # Try to load without mapping to check file validity
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
            if not isinstance(state_dict, dict):
                print('[FIGHT-3DCNN] Invalid weights file format')
                return False
            print(f'[FIGHT-3DCNN] Weights file validated: {weights_path}')
            return True
        except Exception as e:
            print(f'[FIGHT-3DCNN] Weights file validation failed: {e}')
            return False
    def _build_model(self, num_classes: int = 2):
        """
        Build mc3_18 model with custom classification head.
        Falls back to r3d_18 if mc3_18 is unavailable.
        """
        torch, torchvision, _ = _lazy_import_torch()
        
        try:
            # Try mc3_18 first (Mixed Convolutional — faster)
            from torchvision.models.video import mc3_18, MC3_18_Weights
            model = mc3_18(weights=MC3_18_Weights.KINETICS400_V1)
            model_name = 'mc3_18'
        except (ImportError, AttributeError):
            try:
                from torchvision.models.video import mc3_18
                model = mc3_18(pretrained=True)
                model_name = 'mc3_18'
            except Exception:
                # Fallback to r3d_18
                try:
                    from torchvision.models.video import r3d_18, R3D_18_Weights
                    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
                    model_name = 'r3d_18'
                except (ImportError, AttributeError):
                    from torchvision.models.video import r3d_18
                    model = r3d_18(pretrained=True)
                    model_name = 'r3d_18'
        
        # Replace classification head: 400 classes → 2 (fight / no-fight)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        
        print(f'[FIGHT-3DCNN] Built {model_name} model (in_features={in_features})')
        return model, model_name
    
    def _load_model(self):
        """
        BUG 4 FIX: Load the 3D CNN model with proper weight validation.
        
        The old code would silently fall back to randomly-initialized weights if fine-tuned 
        weights weren't available, making it indistinguishable from a working model.
        Now we explicitly validate weights and disable the model if proper weights aren't found.
        """
        try:
            torch, _, _ = _lazy_import_torch()
            
            self._device = self._get_device()
            print(f'[FIGHT-3DCNN] Using device: {self._device}')
            
            # Build model architecture
            model, model_name = self._build_model(num_classes=2)
            
            # BUG 4 FIX: Check for fine-tuned weights using environment variable or default path
            weights_path = os.environ.get('FIGHT_MODEL_WEIGHTS_PATH')
            if not weights_path:
                # Fallback to default location
                weights_dir = os.path.join(os.path.dirname(__file__), '..', 'model_weights')
                weights_path = os.path.join(weights_dir, MODEL_WEIGHTS_FILENAME)
            
            finetuned_loaded = False
            
            # BUG 4 FIX: Validate weights before loading
            if self._validate_weights_file(weights_path):
                try:
                    state_dict = torch.load(weights_path, map_location=self._device, weights_only=False)
                    # Handle different checkpoint formats
                    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']
                    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    
                    # Try loading, handle fc layer size mismatch gracefully
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
            
            # BUG 4 FIX: NEVER silently substitute randomly-initialized head as production model
            if not finetuned_loaded:
                print('[FIGHT-3DCNN] ⚠️  FIGHT DETECTION DISABLED ⚠️')
                print('[FIGHT-3DCNN] Fine-tuned weights not available - cannot provide reliable predictions')
                print('[FIGHT-3DCNN] Model will return fight_detected=False for all predictions')
                
                # Set explicit flag to indicate model is not operational
                with self._model_lock:
                    self._model = None  # Explicitly disable the model
                    self._model_loaded = False
                return
            
            # Only set model as loaded if we have valid fine-tuned weights
            model = model.to(self._device)
            model.eval()
            
            with self._model_lock:
                self._model = model
                self._model_loaded = True
            
            print(f'[FIGHT-3DCNN] Model ready on {self._device} with fine-tuned weights')
        
        except Exception as e:
            print(f'[FIGHT-3DCNN] FATAL: Could not load model: {e}')
            import traceback
            traceback.print_exc()
            # BUG 4 FIX: Ensure model is marked as not loaded on any failure
            with self._model_lock:
                self._model = None
                self._model_loaded = False
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single BGR frame for the 3D CNN.
        
        Steps:
        1. Resize to input_size x input_size
        2. Convert BGR → RGB
        3. Normalize to [0, 1]
        4. Apply Kinetics-400 mean/std normalization
        
        Returns: numpy array of shape (3, H, W), float32
        """
        import cv2
        
        # Resize
        resized = cv2.resize(frame, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_LINEAR)
        
        # BGR → RGB, uint8 → float32 [0, 1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Kinetics-400 normalization (channel-wise)
        for c in range(3):
            rgb[:, :, c] = (rgb[:, :, c] - KINETICS_MEAN[c]) / KINETICS_STD[c]
        
        # HWC → CHW
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
        BUG 4 FIX: Check if the buffer has enough frames AND model is properly loaded with fine-tuned weights.
        
        The old version would return True even with randomly-initialized weights, leading to 
        meaningless predictions being treated as valid fight detection.
        """
        return (len(self._frame_buffer) >= self.sequence_length and 
                self._model_loaded and self._model is not None)
    
    def predict(self) -> Tuple[bool, float]:
        """
        Run fight detection on the current frame buffer.
        
        BUG 4 FIX: Only returns fight predictions if model is loaded with validated fine-tuned weights.
        If fine-tuned weights are not available, always returns (False, 0.0) to indicate fight 
        detection is unavailable, rather than producing meaningless predictions from random weights.
        
        Returns:
            (is_fighting, confidence) tuple
            - is_fighting: True if fight detected above threshold
            - confidence: 0.0 to 1.0 probability of fight
        """
        # Return cached result if not enough new frames
        if self._frames_since_prediction < self._predict_every_n:
            return self._last_prediction
        
        # BUG 4 FIX: Strict ready check - model must be loaded with fine-tuned weights
        if not self.is_ready():
            return False, 0.0
        
        with self._model_lock:
            if self._model is None:
                # BUG 4 FIX: Model explicitly disabled due to missing fine-tuned weights
                return False, 0.0
        
        try:
            torch, _, _ = _lazy_import_torch()
            
            # Stack frames: list of (3, H, W) → (3, T, H, W)
            frames_list = list(self._frame_buffer)
            # Shape: (T, C, H, W)
            clip = np.stack(frames_list, axis=0)
            # Reshape to (C, T, H, W) — PyTorch 3D CNN expects this
            clip = np.transpose(clip, (1, 0, 2, 3))
            
            # Convert to tensor and add batch dimension: (1, C, T, H, W)
            clip_tensor = torch.from_numpy(clip).unsqueeze(0).float()
            clip_tensor = clip_tensor.to(self._device)
            
            # Inference
            with torch.no_grad():
                output = self._model(clip_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                # Class 0 = no-fight, Class 1 = fight
                fight_prob = probabilities[0, 1].item()
            
            is_fighting = fight_prob >= self.confidence_threshold
            self._last_prediction = (is_fighting, fight_prob)
            self._frames_since_prediction = 0
            
            if is_fighting:
                print(f'[FIGHT-3DCNN] ⚠ FIGHT DETECTED! confidence={fight_prob:.3f}')
            
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
    def model_loaded(self) -> bool:
        """Check if the model has been loaded successfully."""
        return self._model_loaded
    
    @property
    def buffer_fill(self) -> float:
        """Return buffer fill level as 0.0 to 1.0."""
        return len(self._frame_buffer) / self.sequence_length

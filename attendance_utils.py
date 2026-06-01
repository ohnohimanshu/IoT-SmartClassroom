"""
Shared utilities for attendance systems.
Used by both entrance_cam and camera_attendance apps.
"""
import base64
import logging
from django.core.files.base import ContentFile
from django.utils import timezone

logger = logging.getLogger(__name__)

# Emotion choices - shared across all apps
EMOTION_CHOICES = [
    ('happy', 'Happy'),
    ('sad', 'Sad'),
    ('angry', 'Angry'),
    ('neutral', 'Neutral'),
    ('surprise', 'Surprise'),
    ('fear', 'Fear'),
    ('disgust', 'Disgust'),
    ('unknown', 'Unknown'),
]

# Mood comparison choices
MOOD_COMPARISON_CHOICES = [
    ('improved', 'Improved'),
    ('declined', 'Declined'),
    ('stable', 'Stable'),
    ('unknown', 'Unknown'),
]

# Emotion categorization
POSITIVE_EMOTIONS = {'happy', 'surprise'}
NEGATIVE_EMOTIONS = {'sad', 'angry', 'fear', 'disgust'}


def mood_comparison(entry_emotion, exit_emotion):
    """
    Compare entry and exit emotions to determine mood change.
    
    Args:
        entry_emotion (str): Emotion at entry
        exit_emotion (str): Emotion at exit
    
    Returns:
        str: One of 'improved', 'declined', 'stable', 'unknown'
    """
    e_in = (entry_emotion or '').lower().strip()
    e_out = (exit_emotion or '').lower().strip()

    if not e_in or not e_out or e_in == 'unknown' or e_out == 'unknown':
        return 'unknown'
    
    if e_in == e_out:
        return 'stable'

    in_pos = e_in in POSITIVE_EMOTIONS
    in_neg = e_in in NEGATIVE_EMOTIONS
    out_pos = e_out in POSITIVE_EMOTIONS
    out_neg = e_out in NEGATIVE_EMOTIONS

    if in_neg and out_pos:
        return 'improved'
    if in_pos and out_neg:
        return 'declined'
    
    return 'stable'


def decode_snapshot(snapshot_b64, roll_no, event_type, max_size_mb=5):
    """
    Decode base64 snapshot and save to file.
    
    Args:
        snapshot_b64 (str): Base64 encoded image
        roll_no (str): Student roll number
        event_type (str): 'entry' or 'exit'
        max_size_mb (int): Maximum file size in MB
    
    Returns:
        ContentFile or None: Decoded image file or None if invalid
    """
    if not snapshot_b64:
        return None
    
    try:
        # Strip data-URI prefix if present
        if ',' in snapshot_b64:
            snapshot_b64 = snapshot_b64.split(',', 1)[1]
        
        # Decode base64
        image_data = base64.b64decode(snapshot_b64)
        
        # Validate file size
        max_bytes = max_size_mb * 1024 * 1024
        if len(image_data) > max_bytes:
            logger.warning(f"Snapshot too large: {len(image_data)} bytes (max {max_bytes})")
            return None
        
        # Validate JPEG header (FFD8FF)
        if not image_data.startswith(b'\xff\xd8\xff'):
            logger.warning("Invalid JPEG header in snapshot")
            return None
        
        # Create filename
        filename = (
            f"snapshot_{roll_no}_{event_type}_"
            f"{timezone.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        
        return ContentFile(image_data, name=filename)
    
    except Exception as e:
        logger.error(f"Failed to decode snapshot: {e}")
        return None


def validate_emotion(emotion):
    """
    Validate emotion value.
    
    Args:
        emotion (str): Emotion to validate
    
    Returns:
        str: Validated emotion or 'unknown'
    """
    emotion = (emotion or '').lower().strip()
    valid_emotions = {choice[0] for choice in EMOTION_CHOICES}
    return emotion if emotion in valid_emotions else 'unknown'


def clamp_score(score, min_val=0.0, max_val=1.0):
    """
    Clamp score between min and max values.
    
    Args:
        score (float): Score to clamp
        min_val (float): Minimum value
        max_val (float): Maximum value
    
    Returns:
        float: Clamped score
    """
    try:
        score = float(score)
        return max(min_val, min(max_val, score))
    except (ValueError, TypeError):
        return min_val


def setup_logging():
    """Setup logging for attendance systems."""
    import logging.config
    
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'verbose': {
                'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
                'style': '{',
            },
            'simple': {
                'format': '{levelname} {asctime} {message}',
                'style': '{',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'logs/attendance.log',
                'maxBytes': 1024 * 1024 * 10,  # 10MB
                'backupCount': 5,
                'formatter': 'verbose',
            },
        },
        'loggers': {
            'attendance': {
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': False,
            },
        },
    }
    
    logging.config.dictConfig(LOGGING_CONFIG)

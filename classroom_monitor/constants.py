"""Shared constants for classroom behavior detection (no heavy imports)."""

FACE_MATCH_TOLERANCE = 0.52

COLOR_MAP = {
    'focused':     (0,  200,  60),
    'distracted':  (0,  165, 255),
    'using_phone': (0,    0, 220),
    'eating_food': (0,    0, 220),
    'fighting':    (255,  0,  0),
    'hand_raised': (255,  255, 0),
    'not_visible': (120, 120, 120),
}

LABEL_MAP = {
    'focused':     'Focused',
    'distracted':  'Distracted',
    'using_phone': 'Using Phone',
    'eating_food': 'Eating Food',
    'fighting':    'Fighting!',
    'hand_raised': 'Hand Raised',
    'not_visible': 'Not Visible',
}

ALERT_POSES = {'using_phone', 'eating_food', 'fighting'}
DISTRACTED_POSES = {'distracted'}

# Incident types that trigger email alerts
EMAIL_ALERT_TYPES = {'using_phone', 'eating_food', 'fighting'}

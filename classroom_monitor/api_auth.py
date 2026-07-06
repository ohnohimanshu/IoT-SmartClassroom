"""Optional API-key auth for detection/reporting endpoints."""

import os
from django.http import JsonResponse


def detection_api_key_configured() -> bool:
    return bool(os.environ.get('DETECTION_API_KEY', '').strip())


def check_detection_api_key(request):
    """
    When DETECTION_API_KEY is set, require X-Detection-API-Key header to match.
    Returns None if authorized, or a JsonResponse error.
    """
    expected = os.environ.get('DETECTION_API_KEY', '').strip()
    if not expected:
        return None
    provided = (request.headers.get('X-Detection-API-Key')
                or request.META.get('HTTP_X_DETECTION_API_KEY', '')).strip()
    if provided != expected:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    return None

"""
Suppress noisy SSLEOFError / BrokenPipeError tracebacks from the dev server.
These are caused by browsers closing connections mid-response and are harmless.
Also suppress mediapipe FaceLandmarker __del__ TypeError on shutdown.
"""
import ssl
import sys
import warnings
from django.core.servers.basehttp import WSGIRequestHandler

# Suppress mediapipe shutdown noise: "TypeError: 'NoneType' object is not callable"
# This happens when the interpreter is shutting down and mediapipe's C++ callbacks
# are already deallocated before Python's __del__ runs.
_original_excepthook = sys.excepthook

def _quiet_excepthook(exc_type, exc_value, exc_tb):
    if exc_type is TypeError and 'NoneType' in str(exc_value):
        return  # swallow mediapipe shutdown noise
    _original_excepthook(exc_type, exc_value, exc_tb)

sys.excepthook = _quiet_excepthook


_original_log_exception = WSGIRequestHandler.log_message.__func__ if hasattr(WSGIRequestHandler.log_message, '__func__') else None


def _handle_request_noisy(self):
    try:
        self.handle_one_request()
    except (BrokenPipeError, ConnectionResetError, ssl.SSLEOFError):
        pass
    except Exception:
        raise


# Patch the low-level handle() method from BaseHTTPRequestHandler
import http.server as _http

_orig_handle = _http.BaseHTTPRequestHandler.handle


def _quiet_handle(self):
    try:
        _orig_handle(self)
    except (BrokenPipeError, ConnectionResetError, ssl.SSLEOFError):
        pass


WSGIRequestHandler.handle = _quiet_handle

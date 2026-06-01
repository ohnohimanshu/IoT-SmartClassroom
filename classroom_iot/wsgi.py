"""
WSGI config for classroom_iot project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.0/howto/deployment/wsgi/
"""

import os
import logging

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'classroom_iot.settings')

# Suppress SSL warnings in development
if os.environ.get('DEBUG', 'True') == 'True':
    logging.disable(logging.CRITICAL)
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except (ImportError, AttributeError):
        pass

application = get_wsgi_application()

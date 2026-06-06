
"""
Entrance Cam App Configuration
"""

from django.apps import AppConfig
from django.db.backends.signals import connection_created


class EntranceCamConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'entrance_cam'

    def ready(self):
        """Called when Django app registry is fully populated."""
        # Register post_save signals
        from . import signals  # noqa: F401

        # WAL mode for better SQLite concurrency
        def _set_wal_mode(sender, connection, **kwargs):
            if connection.vendor == 'sqlite':
                with connection.cursor() as cursor:
                    cursor.execute('PRAGMA journal_mode=WAL;')
                    cursor.execute('PRAGMA synchronous=NORMAL;')
        
        connection_created.connect(_set_wal_mode)

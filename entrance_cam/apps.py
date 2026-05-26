from django.apps import AppConfig
import atexit
import os
import sys


class EntranceCamConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'entrance_cam'

    def ready(self):
        """Initialise app — register signals and optionally start detection service."""
        # Always register signals
        from . import signals  # noqa: F401

        # Enable WAL mode on every new SQLite connection for better concurrency.
        # Use a single cursor per call — creating two cursors for two pragmas was
        # leaving the first one open unnecessarily.
        from django.db.backends.signals import connection_created

        def _set_wal_mode(sender, connection, **kwargs):
            if connection.vendor == 'sqlite':
                with connection.cursor() as cursor:
                    cursor.execute('PRAGMA journal_mode=WAL;')
                    cursor.execute('PRAGMA synchronous=NORMAL;')

        connection_created.connect(_set_wal_mode)

        # Skip starting detection service during server runs to avoid startup hangs
        # and camera connection failures. The detection service should be started
        # manually when needed using: python entrance_cam/detection_script.py
        if len(sys.argv) > 1:
            cmd = sys.argv[1]
            # Skip for all server commands and management commands that don't need detection
            if cmd in ('runsslserver', 'runserver', 'runsslserver_plus', 'check', 'migrate', 'makemigrations', 'shell', 'dbshell', 'collectstatic'):
                return

        # Start detection service only for explicit manual execution
        # This prevents it from starting during server startup or other management commands
        if len(sys.argv) > 1 and sys.argv[1] not in ('runsslserver', 'runserver'):
            from .detection_service import detection_service
            detection_service.start()
            atexit.register(detection_service.stop)
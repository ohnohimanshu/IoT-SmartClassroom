from django.apps import AppConfig
import atexit
import os
import sys


class EntranceCamConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'entrance_cam'

    def ready(self):
        """Initialize app — register signals and optionally start detection service."""
        # Always register signals
        from . import signals  # noqa: F401

        # Enable WAL mode on every new SQLite connection for better concurrency
        from django.db.backends.signals import connection_created

        def _set_wal_mode(sender, connection, **kwargs):
            if connection.vendor == 'sqlite':
                connection.cursor().execute('PRAGMA journal_mode=WAL;')
                connection.cursor().execute('PRAGMA synchronous=NORMAL;')

        connection_created.connect(_set_wal_mode)

        # Only start the detection service in the main server process.
        # Skip during management commands (migrate, collectstatic, etc.) and
        # during the Reloader's parent-spawning phase (RUN_MAIN is not set yet).
        is_management_command = len(sys.argv) > 1 and sys.argv[1] not in (
            'runsslserver', 'runserver',
        )
        # With StatReloader, Django spawns a child with RUN_MAIN=true.
        # We only want to start the service in that child, not the watcher parent.
        is_reloader_parent = os.environ.get('RUN_MAIN') != 'true'

        if is_management_command or is_reloader_parent:
            return

        from .detection_service import detection_service
        detection_service.start()
        atexit.register(detection_service.stop)

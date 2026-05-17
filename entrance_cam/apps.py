from django.apps import AppConfig
import atexit


class EntranceCamConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'entrance_cam'

    def ready(self):
        """Initialize app - register signals and start detection service."""
        # Import signals to register them
        from . import signals
        
        # Import here to avoid circular imports
        from .detection_service import detection_service
        
        # Start detection in background
        detection_service.start()
        
        # Stop detection when Django shuts down
        atexit.register(detection_service.stop)

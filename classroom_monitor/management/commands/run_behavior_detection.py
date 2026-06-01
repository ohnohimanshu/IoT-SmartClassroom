"""
Django management command to run classroom behavior detection.

Usage:
    python manage.py run_behavior_detection --camera-url "rtsp://192.168.1.100/stream" --camera-id 1
"""

from django.core.management.base import BaseCommand
from classroom_monitor.behavior_detection import ClassroomBehaviorDetector
import signal
import sys


class Command(BaseCommand):
    help = 'Run classroom behavior detection (phone, eating, fighting, distracted)'
    
    def add_arguments(self, parser):
        parser.add_argument('--camera-url', type=str, default='0',
                          help='Camera URL or webcam index')
        parser.add_argument('--camera-id', type=int, required=True,
                          help='Camera database ID')
        parser.add_argument('--server', type=str, default='http://localhost:8000',
                          help='Django server URL')
        parser.add_argument('--cooldown', type=int, default=120,
                          help='Seconds between alerts for same behavior')
    
    def handle(self, *args, **options):
        detector = ClassroomBehaviorDetector(
            camera_url=options['camera_url'],
            camera_id=options['camera_id'],
            server_url=options['server'],
            alert_cooldown=options['cooldown']
        )
        
        def signal_handler(sig, frame):
            self.stdout.write(self.style.WARNING('\nStopping detection...'))
            detector.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        self.stdout.write(self.style.SUCCESS('Starting behavior detection...'))
        detector.start()
        
        # Keep running
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            detector.stop()

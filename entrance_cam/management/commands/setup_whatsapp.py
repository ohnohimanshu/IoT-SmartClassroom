"""
Management command to configure WhatsApp alerts
Usage: python manage.py setup_whatsapp
"""

from django.core.management.base import BaseCommand
from entrance_cam.whatsapp_service import get_whatsapp_service
import os
from django.conf import settings


class Command(BaseCommand):
    help = 'Configure and test WhatsApp integration'

    def add_arguments(self, parser):
        parser.add_argument(
            '--test',
            action='store_true',
            help='Test WhatsApp connection',
        )
        parser.add_argument(
            '--status',
            action='store_true',
            help='Show WhatsApp configuration status',
        )

    def handle(self, *args, **options):
        service = get_whatsapp_service()

        if options['status']:
            self.show_status(service)
        elif options['test']:
            self.test_connection(service)
        else:
            self.setup_interactive(service)

    def show_status(self, service):
        """Show WhatsApp configuration status."""
        self.stdout.write(self.style.SUCCESS('\n=== WhatsApp Configuration Status ===\n'))

        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        twilio_num = os.getenv('TWILIO_WHATSAPP_NUMBER')
        alert_num = os.getenv('WHATSAPP_ALERT_NUMBER')

        checks = {
            'TWILIO_ACCOUNT_SID': account_sid,
            'TWILIO_AUTH_TOKEN': auth_token,
            'TWILIO_WHATSAPP_NUMBER': twilio_num,
            'WHATSAPP_ALERT_NUMBER': alert_num,
        }

        for key, value in checks.items():
            if value:
                masked = value[:10] + '...' if len(str(value)) > 10 else value
                self.stdout.write(
                    self.style.SUCCESS(f'✓ {key}: {masked}')
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f'✗ {key}: NOT SET')
                )

        self.stdout.write('')

    def test_connection(self, service):
        """Test WhatsApp connection."""
        self.stdout.write(self.style.SUCCESS('\n=== Testing WhatsApp Connection ===\n'))

        if service.test_connection():
            self.stdout.write(
                self.style.SUCCESS('✓ WhatsApp connection successful!')
            )
            self.stdout.write(
                self.style.WARNING('\nNote: Send a message to the WhatsApp number to activate\n')
            )
        else:
            self.stdout.write(
                self.style.ERROR('✗ WhatsApp connection failed!')
            )
            self.show_status(service)

    def setup_interactive(self, service):
        """Interactive setup wizard."""
        self.stdout.write(self.style.SUCCESS(
            '\n=== WhatsApp Integration Setup ===\n'
        ))

        self.stdout.write('''
This setup will help you configure WhatsApp alerts for the behavior monitoring system.

Requirements:
1. Twilio Account (free tier available at twilio.com)
2. WhatsApp Business Account
3. Twilio Account SID & Auth Token
4. Admin WhatsApp number (for receiving alerts)

Steps:
1. Go to https://www.twilio.com
2. Sign up or log in
3. Set up WhatsApp Sandbox
4. Get your credentials from the Twilio Console
5. Create a .env file in the project root with the credentials
''')

        self.stdout.write(self.style.WARNING('\nExample .env file:\n'))
        self.stdout.write('''
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155552671
WHATSAPP_ALERT_NUMBER=whatsapp:+919876543210
''')

        self.stdout.write(self.style.SUCCESS('\nAfter creating .env file, run:\n'))
        self.stdout.write('python manage.py setup_whatsapp --test\n')

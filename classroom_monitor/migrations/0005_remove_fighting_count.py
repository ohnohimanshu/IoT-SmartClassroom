# Generated migration to remove fighting_count field

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('classroom_monitor', '0004_incidentreport'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='engagementsnapshot',
            name='fighting_count',
        ),
        migrations.RemoveField(
            model_name='videoanalysisframe',
            name='fighting_count',
        ),
    ]

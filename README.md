# ClassroomIoT — Entrance Camera Module

## Setup

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Open http://localhost:8000 → login as your superuser.

## Running the detection script

```bash
python entrance_cam/detection_script.py \
  --camera-url http://192.168.1.100:8080/video \
  --camera-id 1 \
  --server http://localhost:8000 \
  --cooldown 30
```

- `--camera-id` = the ID from the Cameras page in admin
- `--cooldown` = seconds before the same student is logged again (30 = 30s between entry/exit events)

## What it tracks
- Entry time, exit time per student per day
- Emotion at entry and exit (happy/sad/angry/neutral/etc.)
- Total time spent (duration_minutes)
- Snapshot images at entry and exit

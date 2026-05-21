"""
Django signals to auto-generate face encodings when photos are uploaded
and auto-create a User account when a Student is first created.
"""
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
import os
import json

from .models import Student


@receiver(post_save, sender=Student)
def generate_face_encoding_on_photo_upload(sender, instance, created, **kwargs):
    """
    Auto-generate (or regenerate) a face encoding whenever a student's photo
    changes.

    FIX 1: The old guard was:
        if instance.face_encoding and instance.face_encoding.strip(): return
    That meant if a student's photo was *replaced*, the encoding was never
    updated — the system kept matching against the old face.

    New logic:
    - On CREATE: always generate if photo present.
    - On UPDATE: regenerate only if the photo field has actually changed
      (compare the current DB value via a fresh .only() query to avoid a
      full re-fetch).  This prevents unnecessary re-encoding on every save
      while still catching real photo replacements.
    """
    if not instance.photo:
        return

    # Decide whether encoding needs (re)generation
    if not created:
        try:
            db_photo = Student.objects.filter(pk=instance.pk).values_list('photo', flat=True).first()
            photo_unchanged = (db_photo == instance.photo.name)
            encoding_exists = bool(instance.face_encoding and instance.face_encoding.strip())
            if photo_unchanged and encoding_exists:
                # Photo didn't change and we already have an encoding — skip
                return
        except Exception:
            pass  # If the check fails, fall through and regenerate to be safe

    try:
        import face_recognition
    except ImportError:
        print(f"[WARN] face_recognition not installed — cannot encode {instance.name}")
        return

    try:
        image_path = instance.photo.path
        if not os.path.exists(image_path):
            print(f"[ERROR] Photo not found for {instance.name}: {image_path}")
            return

        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            print(f"[WARN] No face detected in photo for {instance.name}")
            return

        # Use .update() to avoid re-triggering post_save signals
        encoding_json = json.dumps(encodings[0].tolist())
        Student.objects.filter(pk=instance.pk).update(face_encoding=encoding_json)
        print(f"[OK] Face encoding {'generated' if created else 'updated'} for {instance.name}")

    except Exception as e:
        print(f"[ERROR] Failed to encode face for {instance.name}: {e}")


@receiver(post_save, sender=Student)
def create_user_for_student(sender, instance, created, **kwargs):
    """
    Auto-create a Django User account the first time a Student is saved.

    FIX 2: The old code called instance.save(update_fields=['user']) to link
    the user back to the student.  That call re-fired post_save, which
    re-triggered the encoding signal (wasting time) and could cause subtle
    double-save races.  Use Student.objects.update() instead — same result,
    no signal re-fire.

    FIX 3: The old password was name.lower() + roll_no with spaces removed.
    Names like "Ravi Kumar Sharma" produced "ravikumarsharma2021CS001" — still
    weak, but consistent.  More importantly, set_password was only called on
    *newly created* users; if get_or_create found an existing user the
    password was left untouched (correct behaviour, but was previously
    ambiguous).
    """
    if not created or instance.user_id:
        # Only run on first-time creation and only if no user is linked yet
        return

    try:
        username = instance.roll_no
        # Simple default password: lowercase name (no spaces) + roll number
        password = instance.name.lower().replace(' ', '') + instance.roll_no

        name_parts = instance.name.split(' ', 1)
        first_name = name_parts[0]
        last_name  = name_parts[1] if len(name_parts) > 1 else ''

        user, user_created = User.objects.get_or_create(
            username=username,
            defaults={
                'first_name': first_name,
                'last_name':  last_name,
                'email':      instance.email,
            },
        )

        if user_created:
            user.set_password(password)
            user.save()
            print(f"[OK] User created for {instance.name} (username: {username})")
        else:
            print(f"[INFO] Username '{username}' already exists — linking existing user to {instance.name}")

        # FIX 2: Use queryset .update() so post_save is NOT re-triggered
        Student.objects.filter(pk=instance.pk).update(user=user)

    except Exception as e:
        print(f"[ERROR] Failed to create user for {instance.name}: {e}")
"""
Django signals:
  1. Auto-generate / refresh face encoding when a student photo is saved.
  2. Auto-create a Django User account on first Student creation.
"""
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
import os
import json
import threading

from .models import Student


def process_face_encoding(student_pk):
    """Process face encoding in a background thread to avoid blocking requests."""
    try:
        instance = Student.objects.get(pk=student_pk)
    except Student.DoesNotExist:
        return

    if not instance.photo:
        # No photo — mark as not enrolled
        Student.objects.filter(pk=instance.pk).update(is_enrolled=False)
        return

    # ── Load face_recognition ONLY WHEN NEEDED ─────────────────────────────────
    try:
        import face_recognition
    except ImportError:
        error_msg = (
            f"[ERROR] face_recognition not installed on server — "
            f"cannot encode photo for {instance.name}. "
            f"Run:  pip install face_recognition"
        )
        print(error_msg)
        Student.objects.filter(pk=instance.pk).update(is_enrolled=False)
        return

    # ── Generate encoding ─────────────────────────────────────────────────────
    try:
        image_path = instance.photo.path
        if not os.path.exists(image_path):
            error_msg = f"[ERROR] Photo file missing for {instance.name}: {image_path}"
            print(error_msg)
            Student.objects.filter(pk=instance.pk).update(is_enrolled=False)
            return

        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            error_msg = (
                f"[WARN] No face detected in photo for {instance.name}. "
                f"Please upload a clear, well-lit front-facing photo with the face centered."
            )
            print(error_msg)
            Student.objects.filter(pk=instance.pk).update(
                face_encoding='',
                is_enrolled=False
            )
            return

        if len(encodings) > 1:
            warn_msg = (
                f"[WARN] Multiple faces detected in photo for {instance.name}. "
                f"Using the first face. Please upload a photo with only one face."
            )
            print(warn_msg)

        # Use first face found
        encoding_json = json.dumps(encodings[0].tolist())

        # Use .update() — does NOT re-trigger post_save
        Student.objects.filter(pk=instance.pk).update(
            face_encoding=encoding_json,
            is_enrolled=True  # ← Mark as successfully enrolled
        )

        success_msg = f"[OK] Face encoding generated for {instance.name} (pk={instance.pk}) — ENROLLED ✓"
        print(success_msg)

    except Exception as e:
        error_msg = f"[ERROR] Failed to encode face for {instance.name}: {e}"
        print(error_msg)
        Student.objects.filter(pk=instance.pk).update(is_enrolled=False)


# ── Signal 1: Face encoding ───────────────────────────────────────────────────

@receiver(post_save, sender=Student)
def generate_face_encoding_on_photo_upload(sender, instance, created, **kwargs):
    """
    Regenerate the face encoding whenever a student record is saved with a photo.
    Runs in a background thread to avoid blocking the HTTP request.
    """
    if not instance.photo:
        # No photo — mark as not enrolled immediately
        Student.objects.filter(pk=instance.pk).update(is_enrolled=False)
        return

    # Check if we need to regenerate
    if not created:
        try:
            db_photo = (Student.objects
                        .filter(pk=instance.pk)
                        .values_list('photo', flat=True)
                        .first())
            photo_unchanged = (db_photo == instance.photo.name)
            encoding_exists = bool(instance.face_encoding and instance.face_encoding.strip())
            if photo_unchanged and encoding_exists:
                return  # Nothing to do
        except Exception:
            pass

    # Launch face encoding in a background thread
    thread = threading.Thread(
        target=process_face_encoding,
        args=(instance.pk,),
        daemon=True
    )
    thread.start()


# ── Signal 2: Auto-create User account ───────────────────────────────────────

@receiver(post_save, sender=Student)
def create_user_for_student(sender, instance, created, **kwargs):
    """
    Create a Django User the first time a Student is saved.

    - Username  : student's roll number
    - Password  : lowercase name (no spaces) + roll number
                  e.g.  "ravikumar2021CS001"
    - Uses Student.objects.update() to link user back — avoids re-firing
      post_save and the face encoding signal.
    """
    # Only run on brand-new students with no user yet
    if not created or instance.user_id:
        return

    try:
        username = instance.email
        password = instance.branch.lower() + instance.roll_no

        name_parts = instance.name.strip().split(' ', 1)
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
            print(
                f"[OK] User created for {instance.name} "
                f"(username: {username}  |  default password set)"
            )
        else:
            print(
                f"[INFO] Username '{username}' already exists — "
                f"linking existing user to {instance.name}"
            )

        # Link back without re-triggering signals
        Student.objects.filter(pk=instance.pk).update(user=user)

    except Exception as e:
        print(f"[ERROR] Failed to create user for {instance.name}: {e}")
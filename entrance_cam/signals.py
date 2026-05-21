"""
Django signals to auto-generate face encodings when photos are uploaded
and auto-create User when Student is created
"""
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
import os
import json

from .models import Student


@receiver(post_save, sender=Student)
def generate_face_encoding_on_photo_upload(sender, instance, created, **kwargs):
    """Auto-generate face encoding when student photo is uploaded."""
    if not instance.photo:
        return
    if instance.face_encoding and instance.face_encoding.strip():
        return

    try:
        # Lazy import — dlib/face_recognition takes ~30s to load; don't block startup
        import face_recognition
    except ImportError:
        print(f"[WARN] face_recognition not installed. Cannot generate encoding for {instance.name}")
        return

    try:
        image_path = instance.photo.path
        if not os.path.exists(image_path):
            print(f"[ERROR] Photo file not found for {instance.name}: {image_path}")
            return

        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            print(f"[WARN] No face detected in photo for {instance.name}")
            return

        encoding_json = json.dumps(face_encodings[0].tolist())
        Student.objects.filter(pk=instance.pk).update(face_encoding=encoding_json)
        print(f"Face encoding generated for {instance.name}")

    except Exception as e:
        print(f"[ERROR] Failed to generate encoding for {instance.name}: {e}")


@receiver(post_save, sender=Student)
def create_user_for_student(sender, instance, created, **kwargs):
    """Auto-create a Django User when a Student is created for the first time."""
    if created and not instance.user:
        try:
            username = instance.roll_no
            password = instance.name.lower().replace(" ", "") + instance.roll_no
            
            # Check if user already exists with this username
            user, created_user = User.objects.get_or_create(
                username=username,
                defaults={
                    'first_name': instance.name.split(' ')[0] if ' ' in instance.name else instance.name,
                    'last_name': instance.name.split(' ')[1] if ' ' in instance.name else '',
                    'email': instance.email,
                }
            )
            
            if created_user:
                user.set_password(password)
                user.save()
                instance.user = user
                instance.save(update_fields=['user'])
                print(f"User created for {instance.name} (username: {username})")
            else:
                print(f"[INFO] User already exists for {instance.name}")
                
        except Exception as e:
            print(f"[ERROR] Failed to create user for {instance.name}: {e}")

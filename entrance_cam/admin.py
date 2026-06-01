from django.contrib import admin
from django.utils.html import format_html
from .models import Student, Camera, AttendanceLog, ESP32Device, FingerprintAttendance

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ['name', 'roll_no', 'course', 'enrollment_status_display', 'fingerprint_status_display', 'is_active']
    list_filter = ['course', 'branch', 'year', 'is_enrolled', 'is_active']
    search_fields = ['name', 'roll_no', 'email']
    readonly_fields = ['face_encoding', 'enrollment_info', 'created_at']
    fieldsets = (
        ('Student Information', {
            'fields': ('name', 'roll_no', 'email', 'course', 'branch', 'year')
        }),
        ('Photo & Face Recognition', {
            'fields': ('photo', 'enrollment_info', 'face_encoding'),
            'description': 'Upload a clear, front-facing photo. Face encoding will be generated automatically.'
        }),
        ('Fingerprint', {
            'fields': ('fingerprint_id', 'fp_confidence', 'fp_scan_count', 'fp_last_seen', 'fp_image'),
            'classes': ('collapse',),
        }),
        ('Account', {
            'fields': ('user', 'is_active', 'created_at'),
            'classes': ('collapse',),
        }),
    )
    actions = ['regenerate_face_encodings_action']

    def enrollment_status_display(self, obj):
        """Display enrollment status with color-coded badge."""
        if obj.is_enrolled and obj.face_encoding:
            return format_html(
                '<span style="color: green; font-weight: bold;">✓ Face Ready</span>'
            )
        elif obj.photo and not obj.face_encoding:
            return format_html(
                '<span style="color: orange; font-weight: bold;">⚠ No Face</span>'
            )
        elif not obj.photo:
            return format_html(
                '<span style="color: red; font-weight: bold;">✗ No Photo</span>'
            )
        else:
            return format_html(
                '<span style="color: gray;">⊘ Unknown</span>'
            )
    enrollment_status_display.short_description = 'Camera Detection Status'

    def fingerprint_status_display(self, obj):
        """Display fingerprint enrollment status."""
        # Show as enrolled if fingerprint_id is set
        if obj.fingerprint_id:
            return format_html(
                '<span style="color: green; font-weight: bold;">✓ Enrolled (ID: {})</span>',
                obj.fingerprint_id
            )
        else:
            return format_html(
                '<span style="color: gray;">⊘ Not Enrolled</span>'
            )
    fingerprint_status_display.short_description = 'Fingerprint Status'

    def enrollment_info(self, obj):
        """Show detailed enrollment information."""
        status = obj.get_enrollment_status()
        encoding_len = len(obj.face_encoding or '') // 100  # Very rough size estimate
        
        html = f'<strong>Status:</strong> {status}<br/>'
        if obj.face_encoding:
            html += f'<strong>Encoding:</strong> Present (~{encoding_len} KB)<br/>'
        html += f'<strong>Photo:</strong> {"✓ Uploaded" if obj.photo else "✗ Missing"}<br/>'
        return format_html(html)
    enrollment_info.short_description = 'Enrollment Information'

    def regenerate_face_encodings_action(self, request, queryset):
        """Admin action to regenerate face encodings for selected students."""
        import json
        import os
        try:
            import face_recognition
        except ImportError:
            self.message_user(request, 'ERROR: face_recognition not installed', level='ERROR')
            return

        updated_count = 0
        failed_students = []

        for student in queryset:
            if not student.photo:
                failed_students.append(f'{student.name} (no photo)')
                continue

            try:
                image_path = student.photo.path
                if not os.path.exists(image_path):
                    failed_students.append(f'{student.name} (file missing)')
                    continue

                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if not encodings:
                    failed_students.append(f'{student.name} (no face detected)')
                    Student.objects.filter(pk=student.pk).update(is_enrolled=False)
                    continue

                encoding_json = json.dumps(encodings[0].tolist())
                Student.objects.filter(pk=student.pk).update(
                    face_encoding=encoding_json,
                    is_enrolled=True
                )
                updated_count += 1

            except Exception as e:
                failed_students.append(f'{student.name} ({str(e)[:30]})')

        if updated_count > 0:
            self.message_user(
                request,
                f'✓ Successfully regenerated encodings for {updated_count} student(s)',
                level='SUCCESS'
            )

        if failed_students:
            msg = f'⚠ Failed for {len(failed_students)} student(s):\n' + '\n'.join(failed_students)
            self.message_user(request, msg, level='WARNING')

    regenerate_face_encodings_action.short_description = '🔄 Regenerate Face Encodings for selected students'

@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ['name', 'url', 'location', 'is_active']
    list_filter = ['is_active', 'location']
    search_fields = ['name', 'url']

@admin.register(AttendanceLog)
class AttendanceLogAdmin(admin.ModelAdmin):
    list_display = ['student', 'camera', 'date', 'entry_time', 'exit_time', 'mood_comparison']
    list_filter = ['date', 'mood_comparison']
    search_fields = ['student__name', 'student__roll_no']
    date_hierarchy = 'date'

@admin.register(ESP32Device)
class ESP32DeviceAdmin(admin.ModelAdmin):
    list_display = ['name', 'ip_address', 'location', 'is_active', 'last_seen']
    list_filter = ['is_active', 'location']
    search_fields = ['name', 'ip_address']

@admin.register(FingerprintAttendance)
class FingerprintAttendanceAdmin(admin.ModelAdmin):
    list_display = ['student', 'device', 'date', 'attendance_type', 'timestamp', 'confidence']
    list_filter = ['date', 'attendance_type']
    search_fields = ['student__name', 'student__roll_no']
    date_hierarchy = 'date'

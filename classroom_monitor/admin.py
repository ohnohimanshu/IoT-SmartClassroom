from django.contrib import admin
from django.utils.html import format_html
from django.db.models import Q
from .models import (
    ClassroomCamera, ClassSession, EngagementSnapshot, StudentZoneLog,
    ClassroomVideo, VideoAnalysisFrame, VideoStudentZone, IncidentReport
)


@admin.register(ClassroomCamera)
class ClassroomCameraAdmin(admin.ModelAdmin):
    list_display = ['name', 'location', 'stream_type', 'is_active', 'created_at']
    list_filter = ['is_active', 'stream_type', 'created_at']
    search_fields = ['name', 'location']
    readonly_fields = ['created_at']


@admin.register(ClassSession)
class ClassSessionAdmin(admin.ModelAdmin):
    list_display = ['subject', 'camera', 'teacher', 'start_time', 'is_active', 'total_students_detected']
    list_filter = ['is_active', 'camera', 'start_time']
    search_fields = ['subject', 'teacher']
    readonly_fields = ['start_time']


@admin.register(EngagementSnapshot)
class EngagementSnapshotAdmin(admin.ModelAdmin):
    list_display = ['session', 'timestamp', 'total_detected', 'engagement_score_display']
    list_filter = ['session', 'timestamp']
    readonly_fields = ['timestamp']
    
    def engagement_score_display(self, obj):
        return f"{obj.engagement_score:.1%}"
    engagement_score_display.short_description = 'Engagement Score'


@admin.register(StudentZoneLog)
class StudentZoneLogAdmin(admin.ModelAdmin):
    list_display = ['snapshot', 'zone_id', 'pose', 'possibly_talking', 'confidence']
    list_filter = ['pose', 'possibly_talking']


@admin.register(ClassroomVideo)
class ClassroomVideoAdmin(admin.ModelAdmin):
    list_display = ['title', 'status', 'uploaded_at', 'duration_seconds', 'total_frames_analyzed']
    list_filter = ['status', 'uploaded_at']
    search_fields = ['title', 'notes']
    readonly_fields = ['uploaded_at', 'processed_at']


@admin.register(VideoAnalysisFrame)
class VideoAnalysisFrameAdmin(admin.ModelAdmin):
    list_display = ['video', 'frame_number', 'timestamp', 'total_detected', 'engagement_score']
    list_filter = ['video', 'timestamp']


@admin.register(VideoStudentZone)
class VideoStudentZoneAdmin(admin.ModelAdmin):
    list_display = ['frame', 'zone_id', 'pose', 'confidence']
    list_filter = ['pose']


@admin.register(IncidentReport)
class IncidentReportAdmin(admin.ModelAdmin):
    list_display = [
        'incident_emoji',
        'incident_type_display',
        'severity_badge',
        'student_display',
        'confidence_bar',
        'detected_at',
        'is_reviewed_display',
    ]
    list_filter = [
        'incident_type',
        'severity',
        'detected_at',
        'is_reviewed',
        ('severity', admin.RelatedOnlyFieldListFilter),
    ]
    search_fields = ['student__name', 'student__roll_no', 'description']
    readonly_fields = [
        'detected_at',
        'whatsapp_sent_at',
        'reviewed_at',
        'snapshot_display',
    ]
    
    fieldsets = (
        ('Incident Details', {
            'fields': ('incident_type', 'severity', 'confidence', 'detected_at', 'description'),
        }),
        ('People & Location', {
            'fields': ('student', 'camera'),
        }),
        ('Evidence', {
            'fields': ('snapshot', 'snapshot_display'),
        }),
        ('Alert Status', {
            'fields': ('whatsapp_sent', 'whatsapp_sent_at'),
        }),
        ('Admin Review', {
            'fields': ('is_reviewed', 'reviewed_by', 'reviewed_at', 'admin_notes'),
        }),
    )
    
    def incident_emoji(self, obj):
        emoji_map = {
            'fighting': '🥊',
            'using_phone': '📱',
            'eating_food': '🍔',
            'looking_away': '👀',
            'head_down': '📝',
            'distracted': '😴',
            'other': '⚠️',
        }
        return emoji_map.get(obj.incident_type, '⚠️')
    incident_emoji.short_description = ''
    
    def incident_type_display(self, obj):
        return obj.get_incident_type_display()
    incident_type_display.short_description = 'Type'
    
    def severity_badge(self, obj):
        colors = {
            'critical': '#dc3545',
            'high': '#fd7e14',
            'medium': '#ffc107',
            'low': '#28a745',
        }
        color = colors.get(obj.severity, '#6c757d')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; '
            'border-radius: 3px; font-weight: bold;">{}</span>',
            color,
            obj.get_severity_display(),
        )
    severity_badge.short_description = 'Severity'
    
    def student_display(self, obj):
        if obj.student:
            return format_html(
                '<strong>{}</strong><br/><small>{}</small>',
                obj.student.name,
                obj.student.roll_no,
            )
        return '—'
    student_display.short_description = 'Student'
    
    def confidence_bar(self, obj):
        width = int(obj.confidence * 100)
        color = '#28a745' if width > 70 else '#ffc107' if width > 40 else '#dc3545'
        return format_html(
            '<div style="width: 100px; height: 20px; background-color: #e9ecef; '
            'border-radius: 3px; overflow: hidden;">'
            '<div style="width: {}px; height: 100%; background-color: {}; '
            'display: flex; align-items: center; justify-content: center; '
            'color: white; font-size: 12px; font-weight: bold;">{:.0%}</div></div>',
            width,
            color,
            obj.confidence,
        )
    confidence_bar.short_description = 'Confidence'
    
    def is_reviewed_display(self, obj):
        if obj.is_reviewed:
            return format_html(
                '<span style="color: green; font-weight: bold;">✓ Reviewed</span>'
            )
        return format_html(
            '<span style="color: red; font-weight: bold;">✗ Pending</span>'
        )
    is_reviewed_display.short_description = 'Review Status'
    
    def snapshot_display(self, obj):
        if obj.snapshot:
            return format_html(
                '<img src="{}" style="max-width: 400px; max-height: 400px;" />',
                obj.snapshot.url,
            )
        return '—'
    snapshot_display.short_description = 'Snapshot Preview'
    
    def get_queryset(self, request):
        """Filter incidents by user permissions (optional)."""
        qs = super().get_queryset(request)
        # Fighting incidents are always visible at the top
        return qs.order_by('-severity', '-detected_at')
    
    actions = ['mark_as_reviewed', 'mark_as_unreviewed']
    
    def mark_as_reviewed(self, request, queryset):
        from django.utils import timezone
        count = queryset.update(is_reviewed=True, reviewed_by=request.user, reviewed_at=timezone.now())
        self.message_user(request, f'{count} incident(s) marked as reviewed.')
    mark_as_reviewed.short_description = 'Mark selected as reviewed'
    
    def mark_as_unreviewed(self, request, queryset):
        count = queryset.update(is_reviewed=False, reviewed_by=None, reviewed_at=None)
        self.message_user(request, f'{count} incident(s) marked as unreviewed.')
    mark_as_unreviewed.short_description = 'Mark selected as unreviewed'

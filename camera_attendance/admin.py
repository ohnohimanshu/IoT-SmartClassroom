from django.contrib import admin
from django.utils.html import format_html
from .models import CameraAttendanceLog


@admin.register(CameraAttendanceLog)
class CameraAttendanceLogAdmin(admin.ModelAdmin):
    list_display = [
        'student',
        'camera',
        'date',
        'entry_time',
        'exit_time',
        'entry_emotion_display',
        'exit_emotion_display',
        'duration_display',
        'mood_display'
    ]
    list_filter = ['date', 'camera', 'entry_emotion', 'exit_emotion', 'mood_comparison']
    search_fields = ['student__name', 'student__roll_no']
    date_hierarchy = 'date'
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('Student & Camera', {
            'fields': ('student', 'camera', 'date')
        }),
        ('Entry', {
            'fields': ('entry_time', 'entry_emotion', 'entry_emotion_score', 'entry_snapshot')
        }),
        ('Exit', {
            'fields': ('exit_time', 'exit_emotion', 'exit_emotion_score', 'exit_snapshot')
        }),
        ('Analysis', {
            'fields': ('mood_comparison', 'duration_minutes', 'is_present')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def entry_emotion_display(self, obj):
        """Display entry emotion with color."""
        colors = {
            'happy': '#10b981',
            'sad': '#3b82f6',
            'angry': '#ef4444',
            'neutral': '#6b7280',
            'surprise': '#f59e0b',
            'fear': '#8b5cf6',
            'disgust': '#ec4899',
            'unknown': '#9ca3af',
        }
        color = colors.get(obj.entry_emotion, '#9ca3af')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 4px;">{}</span>',
            color,
            obj.entry_emotion.title()
        )
    entry_emotion_display.short_description = 'Entry Emotion'
    
    def exit_emotion_display(self, obj):
        """Display exit emotion with color."""
        if not obj.exit_emotion:
            return '—'
        colors = {
            'happy': '#10b981',
            'sad': '#3b82f6',
            'angry': '#ef4444',
            'neutral': '#6b7280',
            'surprise': '#f59e0b',
            'fear': '#8b5cf6',
            'disgust': '#ec4899',
            'unknown': '#9ca3af',
        }
        color = colors.get(obj.exit_emotion, '#9ca3af')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 4px;">{}</span>',
            color,
            obj.exit_emotion.title()
        )
    exit_emotion_display.short_description = 'Exit Emotion'
    
    def duration_display(self, obj):
        """Display duration in minutes."""
        if obj.duration_minutes is None:
            return '—'
        return f'{obj.duration_minutes} min'
    duration_display.short_description = 'Duration'
    
    def mood_display(self, obj):
        """Display mood comparison."""
        if not obj.mood_comparison or obj.mood_comparison == 'unknown':
            return '—'
        colors = {
            'improved': '#10b981',
            'declined': '#ef4444',
            'stable': '#6b7280',
        }
        color = colors.get(obj.mood_comparison, '#9ca3af')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 4px;">{}</span>',
            color,
            obj.mood_comparison.title()
        )
    mood_display.short_description = 'Mood Comparison'

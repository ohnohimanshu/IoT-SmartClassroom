from django import template
from entrance_cam.models import Student

register = template.Library()

@register.simple_tag(takes_context=True)
def is_student(context):
    user = context['user']
    if not user.is_authenticated:
        return False
    try:
        student = user.student_profile
        return True
    except Student.DoesNotExist:
        return False

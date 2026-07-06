#!/usr/bin/env python3
"""
Diagnostic script to identify behavior detection issues
Run this inside the Docker container to check system status
"""

import os
import sys
import requests
import cv2
import traceback

def check_yolo_models():
    """Check if YOLO models are available"""
    print("=" * 50)
    print("CHECKING YOLO MODELS")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        
        # Test pose model
        try:
            pose_model = YOLO('yolo11s-pose.pt')
            print("✅ yolo11s-pose.pt loaded successfully")
        except Exception as e:
            print(f"❌ yolo11s-pose.pt failed: {e}")
        
        # Test object detection model  
        try:
            obj_model = YOLO('yolo11s.pt')
            print("✅ yolo11s.pt loaded successfully")
        except Exception as e:
            print(f"❌ yolo11s.pt failed: {e}")
            
    except Exception as e:
        print(f"❌ Ultralytics import failed: {e}")

def check_camera_urls():
    """Check if camera URLs are accessible"""
    print("=" * 50)
    print("CHECKING CAMERA CONNECTIVITY")
    print("=" * 50)
    
    cameras = [
        ("Camera 2 C Lab", "https://10.17.5.46:8080/video"),
        ("Camera 3 ESP Cam", "http://10.17.6.157/stream")
    ]
    
    for name, url in cameras:
        try:
            # Try HTTP HEAD request first (lightweight)
            response = requests.head(url, timeout=5, verify=False)
            print(f"✅ {name}: HTTP {response.status_code}")
        except requests.exceptions.ConnectTimeout:
            print(f"❌ {name}: Connection timeout")
        except requests.exceptions.ConnectionError:
            print(f"❌ {name}: Connection refused/failed")
        except Exception as e:
            print(f"⚠️  {name}: {e}")

def check_opencv_camera():
    """Test OpenCV camera capture"""
    print("=" * 50)
    print("CHECKING OPENCV CAMERA CAPTURE")
    print("=" * 50)
    
    test_urls = [
        "https://10.17.5.46:8080/video",
        "http://10.17.6.157/stream"
    ]
    
    for url in test_urls:
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ {url}: Frame captured {frame.shape}")
                else:
                    print(f"❌ {url}: No frames available")
                cap.release()
            else:
                print(f"❌ {url}: Cannot open with OpenCV")
        except Exception as e:
            print(f"❌ {url}: OpenCV error - {e}")

def check_fight_detector():
    """Check fight detection weights"""
    print("=" * 50)
    print("CHECKING FIGHT DETECTION")
    print("=" * 50)
    
    weights_path = os.environ.get('FIGHT_MODEL_WEIGHTS_PATH')
    if weights_path:
        if os.path.exists(weights_path):
            size = os.path.getsize(weights_path)
            print(f"✅ Fight weights found: {weights_path} ({size} bytes)")
        else:
            print(f"❌ Fight weights not found: {weights_path}")
    else:
        # Check default location
        default_path = "/app/model_weights/fight_mc3_18_finetuned.pth"
        if os.path.exists(default_path):
            size = os.path.getsize(default_path)
            print(f"✅ Default fight weights found: {default_path} ({size} bytes)")
        else:
            print(f"⚠️  No fight weights (detection will be disabled): {default_path}")

def check_behavior_detection_import():
    """Test importing behavior detection module"""
    print("=" * 50)
    print("CHECKING BEHAVIOR DETECTION MODULE")
    print("=" * 50)
    
    try:
        from classroom_monitor.behavior_detection import ClassroomBehaviorDetector
        print("✅ ClassroomBehaviorDetector import successful")
        
        # Try creating instance
        detector = ClassroomBehaviorDetector(
            camera_url='', 
            camera_id=0, 
            server_url=''
        )
        print("✅ ClassroomBehaviorDetector instance created")
        
    except Exception as e:
        print(f"❌ Behavior detection failed: {e}")
        traceback.print_exc()

def main():
    print("🔍 BEHAVIOR DETECTION DIAGNOSTIC TOOL")
    print("🐛 Identifying issues with blank camera feed")
    print()
    
    check_yolo_models()
    check_camera_urls()  
    check_opencv_camera()
    check_fight_detector()
    check_behavior_detection_import()
    
    print()
    print("=" * 50)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
# Arvyax-Assignment

## Objective
Build a prototype that uses a camera feed to track the position of the user’s hand in real time and detect when the hand approaches a virtual object on the screen.
When the hand reaches this boundary, the system should trigger a clear on-screen warning: DANGER DANGER

## The POC should demonstrate
Real-time hand or fingertip tracking
(without using MediaPipe, OpenPose, or any pose-detection APIs)
You may use classical computer-vision techniques (e.g., color segmentation, Canny edges, contours, background subtraction, convex hull, etc.) or a small custom ML model.
A virtual object or virtual boundary drawn on the screen
(line, box, or any object you choose)
dynamic Distance-based state logic

## The system should classify interaction as:
SAFE – hand comfortably far from the virtual object
WARNING – hand approaching the virtual object
DANGER – hand extremely close / touching the virtual boundary
Visual state feedback overlay

## The live camera view should clearly show:
Current state (SAFE / WARNING / DANGER)
"DANGER DANGER" during the danger state

### Real-time performance
Target ≥ 8 FPS on CPU-only execution
Allowed: OpenCV, NumPy, PyTorch / TensorFlow as required
Not allowed: MediaPipe, OpenPose, cloud AI APIs

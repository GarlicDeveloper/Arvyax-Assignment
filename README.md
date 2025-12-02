# Arvyax-Assignment

## Objective
Build a prototype that uses a camera feed to track the position of the user’s hand in real time and detect when the hand approaches a virtual object on the screen.
When the hand reaches this boundary, the system should trigger a clear on-screen warning: DANGER DANGER

## The POC should demonstrate
Real-time hand or fingertip tracking
(without using MediaPipe, OpenPose, or any pose-detection APIs) <br>
You may use classical computer-vision techniques (e.g., color segmentation, Canny edges, contours, background subtraction, convex hull, etc.) or a small custom ML model. <br>
A virtual object or virtual boundary drawn on the screen
(line, box, or any object you choose)
dynamic Distance-based state logic

## The system should classify interaction as:
SAFE – hand comfortably far from the virtual object <br>
WARNING – hand approaching the virtual object <br>
DANGER – hand extremely close / touching the virtual boundary <br>
Visual state feedback overlay

## The live camera view should clearly show:
Current state (SAFE / WARNING / DANGER) <br>
"DANGER DANGER" during the danger state

### Real-time performance
Target ≥ 8 FPS on CPU-only execution <br>
Allowed: OpenCV, NumPy, PyTorch / TensorFlow as required <br>
Not allowed: MediaPipe, OpenPose, cloud AI APIs <br>

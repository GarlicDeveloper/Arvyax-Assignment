import cv2
import numpy as np
import time
import math
CAM_ID = 0
FRAME_WIDTH = 640   
FRAME_HEIGHT = 480
VIRTUAL_RECT = (0.55, 0.25, 0.95, 0.75)
SAFE_RATIO = 0.25   
WARNING_RATIO = 0.12
DANGER_RATIO = 0.04
BLUR = (7, 7)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
SHOW_DEBUG = False
def create_trackbars(window_name='Trackbars'):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('H_lo', window_name, 0, 179, lambda x: None)
    cv2.createTrackbar('H_hi', window_name, 30, 179, lambda x: None)
    cv2.createTrackbar('S_lo', window_name, 30, 255, lambda x: None)
    cv2.createTrackbar('S_hi', window_name, 200, 255, lambda x: None)
    cv2.createTrackbar('V_lo', window_name, 60, 255, lambda x: None)
    cv2.createTrackbar('V_hi', window_name, 255, 255, lambda x: None)
    cv2.createTrackbar('Cr_lo', window_name, 133, 255, lambda x: None)
    cv2.createTrackbar('Cr_hi', window_name, 173, 255, lambda x: None)
    cv2.createTrackbar('Y_lo', window_name, 0, 255, lambda x: None)
    cv2.createTrackbar('Y_hi', window_name, 255, 255, lambda x: None)
def read_trackbars(window_name='Trackbars'):
    vals = {}
    for name in ('H_lo','H_hi','S_lo','S_hi','V_lo','V_hi','Cr_lo','Cr_hi','Y_lo','Y_hi'):
        vals[name] = cv2.getTrackbarPos(name, window_name)
    return vals
def skin_mask(frame, track_vals=None):
    img = cv2.GaussianBlur(frame, BLUR, 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if track_vals is None:
        h_lo, h_hi = 0, 30
        s_lo, s_hi = 30, 200
        v_lo, v_hi = 60, 255
        cr_lo, cr_hi = 133, 173
        y_lo, y_hi = 0, 255
    else:
        h_lo, h_hi = track_vals['H_lo'], track_vals['H_hi']
        s_lo, s_hi = track_vals['S_lo'], track_vals['S_hi']
        v_lo, v_hi = track_vals['V_lo'], track_vals['V_hi']
        cr_lo, cr_hi = track_vals['Cr_lo'], track_vals['Cr_hi']
        y_lo, y_hi = track_vals['Y_lo'], track_vals['Y_hi']
    lower_hsv = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper_hsv = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    lower_ycrcb = np.array([y_lo, cr_lo, 0], dtype=np.uint8)
    upper_ycrcb = np.array([y_hi, cr_hi, 255], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask
def largest_contour_and_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 2000:  
        return None, None
    M = cv2.moments(largest)
    if M['m00'] == 0:
        cx, cy = None, None
    else:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    return largest, (cx, cy)
def fingertip_point(contour, center):
    if contour is None or center is None:
        return None
    cx, cy = center
    pts = contour.reshape(-1, 2)
    dists = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)
    idx = np.argmax(dists)
    return tuple(pts[idx])
def point_rect_distance(px, py, rect):
    x1,y1,x2,y2 = rect
    if px < x1:
        dx = x1 - px
    elif px > x2:
        dx = px - x2
    else:
        dx = 0
    if py < y1:
        dy = y1 - py
    elif py > y2:
        dy = py - y2
    else:
        dy = 0
    return math.hypot(dx, dy)
def main():
    global SHOW_DEBUG
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("Cannot open camera. Check CAM_ID.")
        return
    create_trackbars('Trackbars')
    prev = time.time()
    fps = 0
    frame_count = 0
    paused = False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diag = math.hypot(w, h)
    safe_px = SAFE_RATIO * diag
    warning_px = WARNING_RATIO * diag
    danger_px = DANGER_RATIO * diag
    print(f"Frame: {w}x{h}, diag={diag:.1f}, thresholds(px): SAFE={safe_px:.1f}, WARN={warning_px:.1f}, DANGER={danger_px:.1f}")
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = cv2.flip(frame, 1)  
            display = frame.copy()
            x1 = int(VIRTUAL_RECT[0] * w)
            y1 = int(VIRTUAL_RECT[1] * h)
            x2 = int(VIRTUAL_RECT[2] * w)
            y2 = int(VIRTUAL_RECT[3] * h)
            rect = (x1, y1, x2, y2)
            track_vals = read_trackbars('Trackbars')
            mask = skin_mask(frame, track_vals)
            cnt, center = largest_contour_and_center(mask)
            fingertip = fingertip_point(cnt, center) if cnt is not None and center[0] is not None else None
            if fingertip is not None:
                dist = point_rect_distance(fingertip[0], fingertip[1], rect)
            else:
                dist = None
            state = "NO HAND"
            color = (200,200,200)
            danger_text = ""
            if dist is None:
                state = "NO HAND"
                color = (200,200,200)
            else:
                if dist <= danger_px:
                    state = "DANGER"
                    color = (0,0,255)  
                    danger_text = "DANGER DANGER"
                elif dist <= warning_px:
                    state = "WARNING"
                    color = (0,165,255)  
                elif dist <= safe_px:
                    state = "SAFE"
                    color = (0,255,0)    
                else:
                    state = "FAR SAFE"
                    color = (0,255,0)
            thickness = 3 if state in ("SAFE","FAR SAFE") else 6 if state=="WARNING" else 10 if state=="DANGER" else 2
            cv2.rectangle(display, (x1,y1), (x2,y2), color, thickness)
            if cnt is not None:
                cv2.drawContours(display, [cnt], -1, (100, 255, 100), 2)
                hull = cv2.convexHull(cnt)
                cv2.drawContours(display, [hull], -1, (255, 100, 100), 1)
            if center is not None and center[0] is not None:
                cv2.circle(display, center, 4, (255,255,255), -1)
            if fingertip is not None:
                cv2.circle(display, fingertip, 8, (255,255,255), -1)
                cx = min(max(fingertip[0], x1), x2)
                cy = min(max(fingertip[1], y1), y2)
                cv2.line(display, fingertip, (cx,cy), (255,255,255), 1)
                cv2.putText(display, f"d={dist:.0f}px", (fingertip[0]+10, fingertip[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(display, f"STATE: {state}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            if danger_text:
                cv2.putText(display, danger_text, (w//6, h//6), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,0,255), 6, cv2.LINE_AA)
                cv2.putText(display, danger_text, (w//6, h//6 + 80), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0,0,255), 4, cv2.LINE_AA)
            frame_count += 1
            now = time.time()
            if now - prev >= 1.0:
                fps = frame_count / (now - prev)
                prev = now
                frame_count = 0
            cv2.putText(display, f"FPS: {fps:.1f}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
            if SHOW_DEBUG:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                debug_combined = np.hstack((cv2.resize(frame, (320,240)), cv2.resize(mask_bgr, (320,240))))
                cv2.imshow('DEBUG: frame | mask', debug_combined)
            cv2.imshow('Hand Boundary POC', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            SHOW_DEBUG = not SHOW_DEBUG
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):
            ts = int(time.time())
            filename = f"screenshot_{ts}.png"
            cv2.imwrite(filename, display)
            print(f"Saved {filename}")
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
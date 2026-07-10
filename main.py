"""
GestureCanvas - Touchless Virtual Drawing System
--------------------------------------------------
A gesture-controlled digital canvas using webcam + MediaPipe hand tracking.

Features
    - 21-landmark real-time hand tracking (MediaPipe)
    - Gesture-based draw / erase / select / clear / undo / save
    - Pinch-to-resize brush (thumb + index distance)
    - Hover-to-click toolbar with colors and tools
    - AI-assisted shape recognition (circle, triangle, rectangle, square,
      pentagon, hexagon, polygon) using contour approximation, circularity
      analysis, convex hulls, and vertex counting
    - Smart flood-fill with contour validation
    - Undo history stack (up to 20 states)

Gestures
    Index finger up ONLY                 -> Draw
    Index + Middle up                    -> Select / hover toolbar
                                            (bring index+middle close = click)
    Thumb + Index up                     -> Adjust brush size (pinch distance)
    Fist (all fingers down)              -> Idle
    Open palm (all 5 up)                 -> Cursor only

Keyboard shortcuts
    q -> quit    s -> save    c -> clear    z -> undo    r -> recognize shape

Author: Kushal Ghimire
Stack: Python, OpenCV, MediaPipe, NumPy
"""

import os
import time
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
TOOLBAR_HEIGHT = 112

# BGR colors
COLORS = {
    "red":    (0, 0, 255),
    "green":  (0, 255, 0),
    "blue":   (255, 0, 0),
    "yellow": (0, 255, 255),
    "purple": (255, 0, 255),
    "cyan":   (255, 255, 0),
    "white":  (255, 255, 255),
}

CARD_BG = (255, 255, 255)
CARD_BORDER = (205, 205, 205)
LABEL_COLOR = (35, 35, 35)
ACTIVE_BORDER = (50, 50, 235)


def rounded_rect(img, pt1, pt2, color, radius, thickness=-1):
    """Draw a filled (thickness<0) or outlined rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
    else:
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Flat vector icons for the toolbar (hand-drawn with cv2 primitives so the
# app stays a single self-contained file with no external icon assets)
# ----------------------------------------------------------------------------
def icon_brush(img, cx, cy):
    cv2.line(img, (cx - 14, cy - 16), (cx + 4, cy + 2), (120, 80, 40), 7, cv2.LINE_AA)
    cv2.circle(img, (cx + 4, cy + 2), 4, (180, 180, 180), -1, cv2.LINE_AA)
    pts = np.array([[cx + 2, cy - 1], [cx + 15, cy + 5], [cx + 6, cy + 17], [cx - 3, cy + 10]])
    cv2.fillPoly(img, [pts], (30, 30, 30))


def icon_eraser(img, cx, cy):
    rounded_rect(img, (cx - 17, cy - 11), (cx + 17, cy + 11), (150, 130, 255), 6, -1)
    cv2.line(img, (cx - 5, cy - 11), (cx - 5, cy + 11), (255, 255, 255), 2, cv2.LINE_AA)
    rounded_rect(img, (cx - 17, cy - 11), (cx + 17, cy + 11), (100, 80, 190), 6, 2)


def icon_fill(img, cx, cy):
    pts = np.array([[cx - 15, cy - 6], [cx + 15, cy - 6], [cx + 9, cy + 15], [cx - 9, cy + 15]])
    cv2.fillPoly(img, [pts], (95, 95, 95))
    cv2.ellipse(img, (cx, cy - 6), (15, 4), 0, 0, 360, (60, 60, 60), -1, cv2.LINE_AA)
    cv2.circle(img, (cx + 12, cy - 15), 4, (0, 140, 255), -1, cv2.LINE_AA)


def icon_undo(img, cx, cy):
    cv2.ellipse(img, (cx, cy), (15, 15), 0, 50, 320, (70, 70, 70), 4, cv2.LINE_AA)
    ang = np.deg2rad(50)
    tx, ty = cx + int(15 * np.cos(ang)), cy + int(15 * np.sin(ang))
    pts = np.array([[tx - 9, ty - 2], [tx + 3, ty - 10], [tx + 5, ty + 5]])
    cv2.fillPoly(img, [pts], (70, 70, 70))


def icon_clear(img, cx, cy):
    cv2.rectangle(img, (cx - 10, cy - 6), (cx + 10, cy + 16), (95, 95, 95), -1)
    cv2.rectangle(img, (cx - 14, cy - 11), (cx + 14, cy - 6), (60, 60, 60), -1)
    cv2.rectangle(img, (cx - 4, cy - 17), (cx + 4, cy - 11), (60, 60, 60), -1)
    for lx in (-5, 0, 5):
        cv2.line(img, (cx + lx, cy - 1), (cx + lx, cy + 12), (230, 230, 230), 1, cv2.LINE_AA)


def icon_save(img, cx, cy):
    rounded_rect(img, (cx - 15, cy - 16), (cx + 15, cy + 16), (70, 70, 70), 3, -1)
    cv2.rectangle(img, (cx - 8, cy - 16), (cx + 8, cy - 6), (215, 215, 215), -1)
    cv2.rectangle(img, (cx - 9, cy - 1), (cx + 9, cy + 13), (255, 255, 255), -1)
    cv2.rectangle(img, (cx - 9, cy - 1), (cx + 9, cy + 13), (150, 150, 150), 1)


def _sparkle(img, cx, cy, s, color):
    pts = np.array([
        [cx, cy - s], [cx + s // 3, cy - s // 3], [cx + s, cy], [cx + s // 3, cy + s // 3],
        [cx, cy + s], [cx - s // 3, cy + s // 3], [cx - s, cy], [cx - s // 3, cy - s // 3],
    ])
    cv2.fillPoly(img, [pts], color)


def icon_ai(img, cx, cy):
    _sparkle(img, cx, cy, 14, (0, 165, 255))
    _sparkle(img, cx + 13, cy - 13, 6, (0, 200, 255))
    _sparkle(img, cx - 13, cy + 11, 5, (0, 200, 255))


ICON_DRAW_FN = {
    "draw": icon_brush,
    "eraser": icon_eraser,
    "fill": icon_fill,
    "undo": icon_undo,
    "clear": icon_clear,
    "save": icon_save,
    "shape": icon_ai,
}
ACTION_LABELS = {
    "draw": "Brush", "eraser": "Eraser", "fill": "Fill", "undo": "Undo",
    "clear": "Clear", "save": "Save", "shape": "A.I",
}


# ----------------------------------------------------------------------------
# Threaded camera capture
# ----------------------------------------------------------------------------
class VideoStream:
    """Reads frames on a background thread so the main loop never blocks on I/O.

    On macOS/AVFoundation, cap.read() inside the render loop stalls the whole
    app for a frame or two whenever the camera driver hiccups; decoupling the
    grab from the render/inference loop keeps the UI responsive even if a
    frame gets dropped.
    """

    def __init__(self, src=0, width=CANVAS_WIDTH, height=CANVAS_HEIGHT):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Could not read from camera.")
        self.frame = frame
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


# ----------------------------------------------------------------------------
# Hand Tracker
# ----------------------------------------------------------------------------
class HandTracker:
    """Wraps MediaPipe Hands with helpers for landmark extraction + finger state."""

    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.6,
                 model_complexity=1, smoothing=0.4):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
            model_complexity=model_complexity,
        )
        self.mp_draw = mp.solutions.drawing_utils
        # Exponential moving average smoothing of landmark positions to kill
        # per-frame jitter from the raw model output (main source of shaky
        # cursor/strokes). Lower `smoothing` = snappier, higher = smoother.
        self.smoothing = smoothing
        self._prev_lms = None
        self._finger_history = deque(maxlen=4)

    def find_hands(self, img, draw=False):
        """Return a list of [ (x,y) x 21 ] landmark lists for detected hands."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        hands_landmarks = []
        if results.multi_hand_landmarks:
            h, w, _ = img.shape
            for hand_landmarks in results.multi_hand_landmarks:
                lms = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
                lms = self._smooth(lms)
                hands_landmarks.append([(int(x), int(y)) for x, y in lms])
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
        else:
            self._prev_lms = None
        return hands_landmarks

    def _smooth(self, lms):
        if self._prev_lms is None or len(self._prev_lms) != len(lms):
            self._prev_lms = lms
            return lms
        a = self.smoothing
        smoothed = [
            (a * px + (1 - a) * x, a * py + (1 - a) * y)
            for (px, py), (x, y) in zip(self._prev_lms, lms)
        ]
        self._prev_lms = smoothed
        return smoothed

    def fingers_up(self, lms):
        """Return [thumb, index, middle, ring, pinky] 0/1 array.

        Distance-from-wrist comparisons (rather than a raw y-coordinate
        comparison) so detection still works when the hand is tilted or
        rotated, not just held perfectly upright. A short majority-vote
        history smooths out single-frame flicker near the up/down boundary.
        """
        wrist = lms[0]

        def dist(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])

        fingers = []
        # Thumb: extended if its tip is farther from the pinky-MCP than its
        # own MCP joint is - robust regardless of hand rotation/mirroring.
        pinky_mcp = lms[17]
        fingers.append(1 if dist(lms[4], pinky_mcp) > dist(lms[2], pinky_mcp) * 1.05 else 0)

        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            fingers.append(1 if dist(lms[tip], wrist) > dist(lms[pip], wrist) * 1.02 else 0)

        self._finger_history.append(fingers)
        votes = np.array(self._finger_history)
        return [1 if v >= (len(votes) / 2) else 0 for v in votes.sum(axis=0)]


# ----------------------------------------------------------------------------
# Shape Recognizer
# ----------------------------------------------------------------------------
class ShapeRecognizer:
    """
    Robust shape recognition for noisy hand-drawn sketches.

    Pipeline:
        1. Approximate the contour with Douglas-Peucker to reduce jitter
        2. Compute circularity = 4*pi*A / P^2  (1.0 = perfect circle)
        3. Compute solidity = A / hullArea  (convex-hull ratio, filters noise)
        4. Count vertices of the approximated polygon
        5. Classify by vertex count + geometric ratios
    """

    MIN_AREA = 800
    CIRCLE_CIRCULARITY = 0.82   # Above square's theoretical 0.785
    MIN_SOLIDITY = 0.80

    @classmethod
    def recognize(cls, contour):
        area = cv2.contourArea(contour)
        if area < cls.MIN_AREA:
            return None, None, 0

        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            return None, None, 0

        # Approximate polygon (Douglas-Peucker)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        vertices = len(approx)

        # Circularity
        circularity = (4.0 * np.pi * area) / (peri * peri)

        # Convex hull solidity - rejects jittery hand-drawn scribbles
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        base_conf = int(min(99, max(55, solidity * 100)))

        if solidity < cls.MIN_SOLIDITY:
            return "polygon", approx, base_conf

        # Vertex-count classification (checked BEFORE circle, since squares
        # have circularity ~0.785 which overlaps with lenient circle thresholds)
        if vertices == 3:
            return "triangle", approx, base_conf

        if vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h) if h else 0
            if 0.9 <= ratio <= 1.1:
                return "square", (x, y, w, h), base_conf
            return "rectangle", (x, y, w, h), base_conf

        if vertices == 5:
            return "pentagon", approx, base_conf
        if vertices == 6:
            return "hexagon", approx, base_conf

        # 7+ vertices + high circularity = circle
        if vertices >= 7 and circularity > cls.CIRCLE_CIRCULARITY:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            circle_conf = int(min(99, max(55, circularity * 100)))
            return "circle", ((int(cx), int(cy)), int(radius)), circle_conf

        return "polygon", approx, base_conf


# ----------------------------------------------------------------------------
# Toolbar
# ----------------------------------------------------------------------------
class Toolbar:
    """Top toolbar rendered as white icon cards, matching a flat app-style UI."""

    COLOR_NAMES = ["red", "green", "blue", "yellow", "purple", "cyan", "white"]
    ACTION_NAMES = ["draw", "eraser", "fill", "undo", "clear", "save", "shape"]

    def __init__(self, width, height=TOOLBAR_HEIGHT):
        self.width = width
        self.height = height
        self.card_y = 6
        self.card_h = height - 12
        self.items = []
        self._build()

    def _build(self):
        pad = 14
        circle_r = 22
        gap = 14

        cx = pad + circle_r
        circles = []
        for name in self.COLOR_NAMES:
            circles.append((name, COLORS[name], cx, self.card_y + self.card_h // 2))
            cx += 2 * circle_r + gap
        colors_right = cx - gap + circle_r + pad

        self.items = [{
            "type": "colorcard", "x": pad - 8, "y": self.card_y,
            "w": colors_right - (pad - 8), "h": self.card_h,
            "circles": circles, "r": circle_r,
        }]

        x = colors_right + 10
        action_w = 96
        for name in self.ACTION_NAMES:
            self.items.append({
                "type": "action", "name": name,
                "x": x, "y": self.card_y, "w": action_w, "h": self.card_h,
            })
            x += action_w + 8

    def draw(self, img, active_color, active_tool):
        cv2.rectangle(img, (0, 0), (self.width, self.height), (238, 238, 238), -1)
        cv2.line(img, (0, self.height), (self.width, self.height), (180, 180, 180), 2)

        for it in self.items:
            x, y, w, h = it["x"], it["y"], it["w"], it["h"]

            if it["type"] == "colorcard":
                rounded_rect(img, (x, y), (x + w, y + h), CARD_BG, 10, -1)
                rounded_rect(img, (x, y), (x + w, y + h), CARD_BORDER, 10, 1)
                for name, color, ccx, ccy in it["circles"]:
                    if color == active_color:
                        cv2.circle(img, (ccx, ccy), it["r"] + 4, ACTIVE_BORDER, 2, cv2.LINE_AA)
                    cv2.circle(img, (ccx, ccy), it["r"], color, -1, cv2.LINE_AA)
                    cv2.circle(img, (ccx, ccy), it["r"], (90, 90, 90), 1, cv2.LINE_AA)

            else:
                name = it["name"]
                is_active = (
                    (name == "draw" and active_tool == "draw") or
                    (name == "eraser" and active_tool == "eraser") or
                    (name == "fill" and active_tool == "fill")
                )
                rounded_rect(img, (x, y), (x + w, y + h), CARD_BG, 10, -1)
                border_color = ACTIVE_BORDER if is_active else CARD_BORDER
                rounded_rect(img, (x, y), (x + w, y + h), border_color, 10, 2 if is_active else 1)

                icon_cx, icon_cy = x + w // 2, y + h // 2 - 14
                ICON_DRAW_FN[name](img, icon_cx, icon_cy)

                label = ACTION_LABELS[name]
                (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.58, 1)
                cv2.putText(img, label, (x + (w - tw) // 2, y + h - 8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.58, LABEL_COLOR, 1, cv2.LINE_AA)

    def hit_test(self, px, py):
        if py > self.height:
            return None
        for it in self.items:
            if it["type"] == "colorcard":
                for name, color, ccx, ccy in it["circles"]:
                    if np.hypot(px - ccx, py - ccy) <= it["r"] + 6:
                        return {"type": "color", "name": name, "color": color,
                                "x": ccx - it["r"], "y": ccy - it["r"],
                                "w": 2 * it["r"], "h": 2 * it["r"]}
                continue
            if it["x"] <= px <= it["x"] + it["w"] and \
               it["y"] <= py <= it["y"] + it["h"]:
                return it
        return None


# ----------------------------------------------------------------------------
# GestureCanvas Application
# ----------------------------------------------------------------------------
class GestureCanvas:
    def __init__(self):
        self.stream = VideoStream(0, CANVAS_WIDTH, CANVAS_HEIGHT)

        self.tracker = HandTracker()
        self._frame_times = deque(maxlen=30)
        self.toolbar = Toolbar(CANVAS_WIDTH)
        self.canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

        # Drawing state
        self.active_color = COLORS["red"]
        self.active_tool = "draw"        # draw | eraser | fill
        self.brush_size = 8
        self.eraser_size = 45

        # Stroke state
        self.prev_point = None
        self.stroke_points = []          # points of current stroke (for shape recog)
        self.last_strokes = deque(maxlen=6)  # recent finished strokes

        # Undo history
        self.history = deque(maxlen=20)
        self._push_history()

        # Cooldown for toolbar clicks
        self.click_cooldown_until = 0.0

        # Toast messages
        self.toast = ""
        self.toast_until = 0.0

        # Shape-recognition readout ("Mapped: X / Confidence: NN%")
        self.shape_readout = None
        self.shape_readout_until = 0.0

    # -------------------- state helpers --------------------
    def _push_history(self):
        self.history.append(self.canvas.copy())

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.canvas = self.history[-1].copy()
            self._toast("Undo")

    def clear(self):
        self.canvas[:] = 0
        self._push_history()
        self._toast("Canvas cleared")

    def save_image(self):
        os.makedirs("saves", exist_ok=True)
        fn = f"saves/gesturecanvas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        # Save a white-background version for nicer output
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        out = np.full_like(self.canvas, 255)
        out[mask > 0] = self.canvas[mask > 0]
        cv2.imwrite(fn, out)
        self._toast(f"Saved -> {fn}")

    def _toast(self, msg, duration=2.0):
        self.toast = msg
        self.toast_until = time.time() + duration

    # -------------------- shape recognition --------------------
    def recognize_last_shape(self):
        """Recognize the most recent stroke and replace it with a clean shape."""
        if not self.last_strokes:
            self._toast("No stroke to recognize")
            return

        stroke = self.last_strokes[-1]
        if len(stroke) < 8:
            self._toast("Stroke too small")
            return

        # Build a mask of just this stroke and find contours
        stroke_mask = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.uint8)
        pts = np.array(stroke, dtype=np.int32)
        cv2.polylines(stroke_mask, [pts], isClosed=False,
                      color=255, thickness=max(self.brush_size, 6))

        # Close small gaps so the sketch reads as one shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(stroke_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            self._toast("No contour found")
            return

        largest = max(contours, key=cv2.contourArea)
        shape_name, data, confidence = ShapeRecognizer.recognize(largest)
        if shape_name is None:
            self._toast("Shape unclear")
            return

        # Erase the raw stroke on the actual canvas, then draw the clean shape
        cv2.polylines(self.canvas, [pts], isClosed=False,
                      color=(0, 0, 0), thickness=max(self.brush_size, 6) + 4)

        color = self.active_color
        thick = max(self.brush_size, 3)

        if shape_name == "circle":
            (cx, cy), r = data
            cv2.circle(self.canvas, (cx, cy), r, color, thick)
        elif shape_name in ("rectangle", "square"):
            x, y, w, h = data
            cv2.rectangle(self.canvas, (x, y), (x + w, y + h), color, thick)
        else:
            cv2.drawContours(self.canvas, [data], -1, color, thick)

        self._push_history()
        self.last_strokes.clear()
        self.shape_readout = (shape_name, confidence)
        self.shape_readout_until = time.time() + 3.0

    # -------------------- smart fill --------------------
    def smart_fill(self, seed):
        """Flood-fill from seed; only fills if inside a valid contour."""
        x, y = seed
        if y <= TOOLBAR_HEIGHT:
            self._toast("Fill point in toolbar")
            return

        # Build a binary mask of the current strokes to validate the seed
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, strokes = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

        # Close outlines and find enclosing contour
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        inside = False
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
            if cv2.pointPolygonTest(c, (x, y), False) >= 0:
                inside = True
                break

        if not inside:
            self._toast("Fill point not inside a shape")
            return

        mask = np.zeros((CANVAS_HEIGHT + 2, CANVAS_WIDTH + 2), dtype=np.uint8)
        cv2.floodFill(self.canvas, mask, (x, y), self.active_color,
                      loDiff=(5, 5, 5), upDiff=(5, 5, 5))
        self._push_history()
        self._toast("Filled")

    # -------------------- toolbar interaction --------------------
    def handle_toolbar_click(self, item):
        if time.time() < self.click_cooldown_until:
            return
        self.click_cooldown_until = time.time() + 0.5

        if item["type"] == "color":
            self.active_color = item["color"]
            if self.active_tool in ("eraser", "fill"):
                self.active_tool = "draw"
            self._toast(f"Color: {item['name']}")
        else:
            name = item["name"]
            if name == "draw":
                self.active_tool = "draw"
                self._toast("Brush")
            elif name == "eraser":
                self.active_tool = "eraser"
                self._toast("Eraser")
            elif name == "fill":
                self.active_tool = "fill"
                self._toast("Fill - point at a closed shape")
            elif name == "undo":
                self.undo()
            elif name == "clear":
                self.clear()
            elif name == "save":
                self.save_image()
            elif name == "shape":
                self.recognize_last_shape()

    # -------------------- gesture dispatch --------------------
    def _finish_stroke(self):
        if self.stroke_points and len(self.stroke_points) > 3:
            self.last_strokes.append(list(self.stroke_points))
            self._push_history()
        self.stroke_points = []
        self.prev_point = None

    def process_hand(self, display, lms):
        fingers = self.tracker.fingers_up(lms)
        index_tip = lms[8]
        middle_tip = lms[12]
        thumb_tip = lms[4]

        # Cursor
        cv2.circle(display, index_tip, 9, self.active_color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(display, index_tip, 11, (255, 255, 255), 1, cv2.LINE_AA)

        # ---- Selection mode: index + middle up ----
        if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0:
            self._finish_stroke()
            cv2.line(display, index_tip, middle_tip, (200, 200, 200), 1)

            d = np.hypot(index_tip[0] - middle_tip[0],
                         index_tip[1] - middle_tip[1])
            mid = ((index_tip[0] + middle_tip[0]) // 2,
                   (index_tip[1] + middle_tip[1]) // 2)

            # Small selection-point box at the fingertip midpoint so it's
            # obvious where the "cursor" actually is, independent of any
            # toolbar item underneath it.
            box_color = (0, 200, 0) if d < 48 else (0, 140, 255)
            cv2.rectangle(display, (mid[0] - 10, mid[1] - 10),
                          (mid[0] + 10, mid[1] + 10), box_color, 2, cv2.LINE_AA)

            item = self.toolbar.hit_test(*mid)
            if item:
                cv2.rectangle(display,
                              (item["x"] - 3, item["y"] - 3),
                              (item["x"] + item["w"] + 3, item["y"] + item["h"] + 3),
                              box_color, 2, cv2.LINE_AA)
                if d < 48:
                    self.handle_toolbar_click(item)
            return

        # ---- Brush resize: thumb + index up, middle down ----
        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0:
            self._finish_stroke()
            d = int(np.hypot(index_tip[0] - thumb_tip[0],
                             index_tip[1] - thumb_tip[1]))
            self.brush_size = max(2, min(60, d // 4))
            mid = ((index_tip[0] + thumb_tip[0]) // 2,
                   (index_tip[1] + thumb_tip[1]) // 2)
            cv2.line(display, thumb_tip, index_tip, (255, 255, 255), 1)
            cv2.circle(display, mid, self.brush_size, self.active_color, 2)
            cv2.putText(display, f"Brush: {self.brush_size}",
                        (mid[0] + 20, mid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return

        # ---- Draw mode: index only up ----
        if fingers == [0, 1, 0, 0, 0]:
            if index_tip[1] <= TOOLBAR_HEIGHT:
                self._finish_stroke()
                return

            if self.active_tool == "fill":
                self.smart_fill(index_tip)
                self.active_tool = "draw"
                return

            if self.active_tool == "eraser":
                cv2.circle(self.canvas, index_tip, self.eraser_size,
                           (0, 0, 0), -1)
                cv2.circle(display, index_tip, self.eraser_size,
                           (200, 200, 200), 2)
                self.prev_point = None
                self.stroke_points = []
                return

            # Draw
            if self.prev_point is None:
                self.prev_point = index_tip
            cv2.line(self.canvas, self.prev_point, index_tip,
                     self.active_color, self.brush_size, cv2.LINE_AA)
            self.prev_point = index_tip
            self.stroke_points.append(index_tip)
            return

        # ---- Fist / anything else: finish stroke, idle ----
        self._finish_stroke()

    # -------------------- main loop --------------------
    def run(self):
        print("GestureCanvas running. Press 'q' to quit.")
        while True:
            frame_start = time.time()
            frame = self.stream.read()
            frame = cv2.flip(frame, 1)
            display = frame.copy()

            hands = self.tracker.find_hands(display, draw=False)
            if hands:
                self.process_hand(display, hands[0])
            else:
                self._finish_stroke()

            # Composite canvas onto webcam feed
            gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask_inv = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
            mask_inv_3 = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
            display = cv2.bitwise_and(display, mask_inv_3)
            display = cv2.bitwise_or(display, self.canvas)

            # Toolbar overlay
            self.toolbar.draw(display, self.active_color, self.active_tool)

            # Status bar
            self._frame_times.append(time.time() - frame_start)
            fps = 1.0 / (sum(self._frame_times) / len(self._frame_times) + 1e-6)
            status = (f"Tool: {self.active_tool.upper()}  |  "
                      f"Brush: {self.brush_size}  |  "
                      f"Strokes: {len(self.last_strokes)}  |  "
                      f"FPS: {fps:.0f}")
            cv2.rectangle(display, (0, CANVAS_HEIGHT - 40),
                          (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0), -1)
            cv2.putText(display, status, (10, CANVAS_HEIGHT - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Toast
            if self.toast and time.time() < self.toast_until:
                (tw, th), _ = cv2.getTextSize(self.toast,
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                x = (CANVAS_WIDTH - tw) // 2
                y = CANVAS_HEIGHT - 60
                cv2.rectangle(display, (x - 10, y - th - 10),
                              (x + tw + 10, y + 10), (0, 0, 0), -1)
                cv2.rectangle(display, (x - 10, y - th - 10),
                              (x + tw + 10, y + 10), (0, 255, 255), 1)
                cv2.putText(display, self.toast, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Shape-recognition readout, bottom-left, green (AI-detection style)
            if self.shape_readout and time.time() < self.shape_readout_until:
                shape_name, confidence = self.shape_readout
                line1 = f"Mapped: {shape_name.capitalize()}"
                line2 = f"Confidence: {confidence}%"
                cv2.putText(display, line1, (14, CANVAS_HEIGHT - 66),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 80), 2, cv2.LINE_AA)
                cv2.putText(display, line2, (14, CANVAS_HEIGHT - 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 1, cv2.LINE_AA)

            cv2.imshow("GestureCanvas", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                self.save_image()
            elif key == ord("c"):
                self.clear()
            elif key == ord("z"):
                self.undo()
            elif key == ord("r"):
                self.recognize_last_shape()

        self.stream.stop()
        cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
    GestureCanvas().run()

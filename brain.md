# GestureCanvas — Project Notes

Touchless virtual drawing app: webcam + MediaPipe hand tracking, single file, Python.

## Stack
Python 3.12 · OpenCV · MediaPipe 0.10.14 · NumPy — deps in `requirements.txt`, venv at `.venv/`.

Run:
```bash
source .venv/bin/activate
python main.py
```
(Running `python3 main.py` without activating the venv first fails with `ModuleNotFoundError: No module named 'cv2'` — system Python doesn't have the deps.)

## Architecture (`main.py`, single file)
- `VideoStream` — threaded webcam capture
- `HandTracker` — MediaPipe wrapper, finger-up/down state helper
- `ShapeRecognizer` — contour → shape classification (Douglas-Peucker approximation, circularity `4πA/P²`, convex-hull solidity filter)
- `Toolbar` — color swatches + action buttons (hover-to-select, pinch-to-click)
- `GestureCanvas` — main loop, state machine, gesture dispatch
- helper `icon_*` functions draw toolbar icons; `rounded_rect` draws UI chrome

## Gestures
| Gesture | Action |
|---|---|
| Index up only | Draw |
| Index + middle up | Hover/select toolbar (bring together = click) |
| Thumb + index up | Adjust brush size (pinch distance) |
| Fist | Idle |
| Open palm | Cursor only |

## Keyboard shortcuts
`q` quit · `s` save · `c` clear · `z` undo · `r` recognize shape

## State / output
- Undo history: last 20 states
- Saves export PNG to `saves/gesturecanvas_<timestamp>.png` with white background

## In progress
- A React frontend (`web/`, Vite + TypeScript) is being scaffolded alongside this to eventually pair with/control the Python app, starting with a login page as the entry screen. Not yet wired to the Python backend.

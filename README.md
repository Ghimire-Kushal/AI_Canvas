# GestureCanvas 🎨

Touchless virtual drawing system controlled entirely through webcam-captured hand gestures.

**Stack:** Python · OpenCV · MediaPipe · NumPy

## Features

- **21-landmark hand tracking** — MediaPipe Hands, single-hand mode
- **Gesture drawing** — index finger acts as pen
- **Pinch-to-resize brush** — thumb + index distance controls brush size (2–60 px)
- **Hover-select toolbar** — index + middle up hovers, bringing them together clicks
- **Smart eraser** — dedicated tool with larger radius
- **Shape recognition** — circle, triangle, rectangle, square, pentagon, hexagon, polygon
  - Contour approximation (Douglas–Peucker)
  - Circularity analysis: `4πA / P²`
  - Convex-hull solidity filter for jitter rejection
- **Smart fill** — flood-fills only inside validated closed contours
- **Undo history** — up to 20 states
- **PNG export** — saved to `saves/` with a clean white background

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Requires a working webcam.

## Gestures

| Gesture | Action |
|---|---|
| ☝️ Index up only | **Draw** with active tool |
| ✌️ Index + Middle up | **Hover / select** on toolbar (bring fingers together = click) |
| 👍 Thumb + Index up | **Adjust brush size** (pinch distance) |
| ✊ Fist / other | **Idle** — ends current stroke |

## Keyboard shortcuts

| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Save canvas |
| `c` | Clear canvas |
| `z` | Undo |
| `r` | Recognize last stroke as a shape |

## Toolbar

Row of color swatches (red, green, blue, yellow, purple, cyan, white) followed by:

- **ERASER** — switch to eraser tool
- **SHAPE** — replace last stroke with a clean recognized geometric shape
- **FILL** — next tap floods a closed region with active color
- **UNDO** — pop last history state
- **CLEAR** — wipe canvas
- **SAVE** — export PNG to `saves/`

## Architecture

```
main.py
├── HandTracker         # MediaPipe wrapper, finger-state helper
├── ShapeRecognizer     # Contour → shape classification
├── Toolbar             # Color swatches + action buttons
└── GestureCanvas       # Main loop, state, gesture dispatch
```

Everything is in a single file for portability.

## Notes

- Runs at ~25–30 FPS on a modern laptop CPU.
- If MediaPipe fails to install, ensure your Python is 3.9–3.11 (3.12 is fine with mediapipe ≥ 0.10.9).
- On Linux you may need `sudo apt install libgl1` for OpenCV.
# AI_Canvas
# AI_Canvas

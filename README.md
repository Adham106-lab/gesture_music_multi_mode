---

#  Gesture-Controlled Music System 

### Multi-Mode Hand-Gesture Synthesizer using MediaPipe + OpenCV

A real-time music system that turns **your hand movements into sound.**
Supports **three different control modes**, live note visualization, and generates tones using NumPy + sounddevice.

> Move your hand ➜ select note
> Gesture, fist, or keyboard ➜ play sound

---

## 🔥 Features

| Feature                       | Description                                                 |
| ----------------------------- | ----------------------------------------------------------- |
| 🎹 **5x5 Musical Grid**       | 25 notes from C3 upward with labeled pitch mapping          |
| 🖐 Multi Interaction Modes    | Keyboard Trigger / Two-Hand / Gesture-Only Control          |
| 🎶 Harmonics + ADSR Envelope  | Rich sound generated per tone for better musical feel       |
| 🎧 Auto Audio Test on Startup | Confirms driver / device readiness                          |
| ⚡ Real-time Hand Tracking     | Powered by MediaPipe Hands + OpenCV                         |
| 🧠 Gesture Recognition        | Peace ✌, Fist ✊, Open Hand ✋ for live actions               |
| 🖥 Visual UI                  | Grid, note names, selection cursor & trigger status overlay |

---

## 📌 Control Modes

| Mode              | Cursor Source                   | Trigger Method                                |
| ----------------- | ------------------------------- | --------------------------------------------- |
| **Keyboard Mode** | Index finger for selecting note | **SPACE** plays tone                          |
| **Two-Hand Mode** | Left hand selects note          | Right-hand **Fist** plays note                |
| **Gesture Mode**  | One hand controls everything    | ✌ Open/Fingers ➜ play<br>✊ Closed fist ➜ stop |

---

## 🛠 Requirements

Install dependencies:

```bash
pip install opencv-python mediapipe numpy sounddevice
```

---

## 🚀 How to Run

```bash
python gesture_music_v2.py
```

or import as a module:

```python
from gesture_music_v2 import GestureMusicSystemV2

system = GestureMusicSystemV2(mode="gesture")  # keyboard / two_hand / gesture
system.run()
```

---

## 📂 File Contents

*(based on your code)*

```
Gesture-Controlled-Music-V2/
├── gesture_music_v2.py   # Full system implementation
└── README.md             # You are here
```

---

## 🧩 To-Do / Future Enhancements

* 🎵 Add chords + multi-note harmonics
* 🔊 Add reverb, distortion, filters, autotune
* 📊 Web leaderboard + scoring game mode
* 🧠 LSTM / AI-improvised playback system
* 🎼 MIDI output → connect to real instruments

---

## 👨‍💻 Author

**ADham Omar**
Github: *Adham106-lab*
Instagram: *@adhamomar1112*
**Date:** November 2025

---

If you like the project — **star ⭐ the repo**
and feel free to open issues or contribute!

💬 *If anyone wants the code or has upgrades in mind — send me ideas!* 🎶🔥


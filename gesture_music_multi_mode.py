"""
Gesture-Controlled Music System - Version 2 (Multi-Mode)
Three control modes: Keyboard, Two-Hand, or Single-Hand Gestures
Uses MediaPipe and OpenCV for hand tracking and NumPy/sounddevice for audio synthesis
Author: ADham Omar
Adham106-lab
insta:adhamomar1112
Date: November 2025
"""

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading
import time
from collections import deque

class GestureMusicSystemV2:
    def __init__(self, grid_size=(5, 5), mode="keyboard"):
        """
        Initialize the music system with specified control mode

        Args:
            grid_size: Tuple of (rows, cols) for the note grid
            mode: Control mode - "keyboard", "two_hand", or "gesture"
        """
        self.mode = mode

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Set max hands based on mode
        max_hands = 2 if mode == "two_hand" else 1

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Audio setup with debugging
        self.SAMPLE_RATE = 44100
        print("\n🔊 Audio System Initialization...")
        print(f"Available audio devices:")
        print(sd.query_devices())

        try:
            sd.default.samplerate = self.SAMPLE_RATE
            sd.default.channels = 1
            print(f"✓ Sample rate: {self.SAMPLE_RATE} Hz")
            print(f"✓ Default device: {sd.default.device}")
        except Exception as e:
            print(f"⚠ Audio setup warning: {e}")

        # Grid setup
        self.grid_rows, self.grid_cols = grid_size
        self.total_notes = self.grid_rows * self.grid_cols

        # Generate musical scale
        self.notes = self.generate_scale(self.total_notes)
        self.note_names = self.generate_note_names(self.total_notes)

        # Grid state
        self.cursor_position = (0, 0)
        self.selected_note_idx = 0
        self.last_played_note = -1

        # Hand tracking state
        self.cursor_hand_position = None

        # Trigger state (varies by mode)
        self.trigger_active = False
        self.space_pressed = False
        self.trigger_history = deque(maxlen=3)
        self.gesture_history = deque(maxlen=5)

        # Visual settings
        self.frame_width = 640
        self.frame_height = 480
        self.grid_margin = 50

        # Thread safety
        self.sound_lock = threading.Lock()

        # Test audio on startup
        self.test_audio()

    def test_audio(self):
        """Test audio playback on startup"""
        print("\n🎵 Testing audio playback...")
        try:
            test_freq = 440.0  # A4 note
            tone = self.generate_tone(test_freq, duration=0.3)
            sd.play(tone, blocking=True)
            sd.wait()
            print("✓ Audio test successful!")
        except Exception as e:
            print(f"✗ Audio test failed: {e}")
            print("  Possible solutions:")
            print("  1. Check if your speakers/headphones are connected")
            print("  2. Adjust system volume")
            print("  3. Try: pip install --upgrade sounddevice")

    def generate_scale(self, num_notes):
        """Generate a musical scale with given number of notes"""
        base_freq = 130.81  # C3
        notes = {}
        for i in range(num_notes):
            freq = base_freq * (2 ** (i / 12))
            notes[i] = freq
        return notes

    def generate_note_names(self, num_notes):
        """Generate note names for the scale"""
        note_names_cycle = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        names = {}
        octave = 3
        for i in range(num_notes):
            note_idx = i % 12
            if i > 0 and note_idx == 0:
                octave += 1
            names[i] = f"{note_names_cycle[note_idx]}{octave}"
        return names

    def generate_tone(self, frequency, duration=0.4):
        """Generate a musical tone with ADSR envelope"""
        samples = int(self.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, False)

        # Generate sine wave with harmonics
        wave = 0.5 * np.sin(2 * np.pi * frequency * t)
        wave += 0.2 * np.sin(2 * np.pi * frequency * 2 * t)
        wave += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)

        # ADSR envelope
        attack = int(0.02 * samples)
        decay = int(0.05 * samples)
        sustain_level = 0.7
        release = int(0.2 * samples)

        envelope = np.ones(samples)

        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)
        if decay > 0:
            envelope[attack:attack+decay] = np.linspace(1, sustain_level, decay)
        if release > 0:
            envelope[-release:] = np.linspace(sustain_level, 0, release)

        return (wave * envelope).astype(np.float32)

    def play_note(self, note_index):
        """Play a musical note"""
        if note_index not in self.notes:
            return

        frequency = self.notes[note_index]

        def play():
            try:
                with self.sound_lock:
                    tone = self.generate_tone(frequency)
                    print(f"🔊 Playing: {self.note_names[note_index]} ({frequency:.2f} Hz)")
                    sd.play(tone, blocking=False)
                    time.sleep(0.05)
            except Exception as e:
                print(f"✗ Playback error: {e}")

        thread = threading.Thread(target=play, daemon=True)
        thread.start()

    def detect_hand_gesture(self, hand_landmarks):
        """Detect single-hand gestures (for gesture mode)"""
        if not hand_landmarks:
            return -1

        landmarks = hand_landmarks.landmark

        # Check finger extensions
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y

        # Gesture: Peace sign (index + middle)
        if index_up and middle_up and not ring_up and not pinky_up:
            return 1  # Peace sign detected

        # Gesture: Closed fist (all fingers down)
        if not index_up and not middle_up and not ring_up and not pinky_up:
            return 0  # Fist detected

        # Gesture: Open hand (all fingers up)
        if index_up and middle_up and ring_up and pinky_up:
            return 2  # Open hand detected

        return -1

    def smooth_gesture(self, gesture):
        """Smooth gesture detection using history"""
        self.gesture_history.append(gesture)

        if len(self.gesture_history) < 3:
            return -1

        if self.gesture_history.count(gesture) >= 3:
            return gesture

        return -1

    def detect_fist_trigger(self, hand_landmarks):
        """Detect closed fist for two-hand mode"""
        if not hand_landmarks:
            return False

        landmarks = hand_landmarks.landmark
        palm_y = landmarks[0].y

        fingertip_indices = [8, 12, 16, 20]
        closed_count = 0
        for tip_idx in fingertip_indices:
            if landmarks[tip_idx].y > palm_y + 0.05:
                closed_count += 1

        return closed_count >= 3

    def get_cursor_position(self, hand_landmarks, frame_shape):
        """Get cursor position from index fingertip"""
        if not hand_landmarks:
            return None

        index_tip = hand_landmarks.landmark[8]
        h, w, _ = frame_shape
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        return (x, y)

    def map_to_grid(self, position, frame_shape):
        """Map cursor position to grid cell"""
        if position is None:
            return self.cursor_position

        x, y = position
        h, w, _ = frame_shape

        grid_width = w - 2 * self.grid_margin
        grid_height = h - 2 * self.grid_margin

        cell_width = grid_width / self.grid_cols
        cell_height = grid_height / self.grid_rows

        col = int((x - self.grid_margin) / cell_width)
        row = int((y - self.grid_margin) / cell_height)

        col = max(0, min(self.grid_cols - 1, col))
        row = max(0, min(self.grid_rows - 1, row))

        return (row, col)

    def draw_grid(self, frame):
        """Draw the musical note grid"""
        h, w, _ = frame.shape

        grid_width = w - 2 * self.grid_margin
        grid_height = h - 2 * self.grid_margin

        cell_width = grid_width / self.grid_cols
        cell_height = grid_height / self.grid_rows

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1 = int(self.grid_margin + col * cell_width)
                y1 = int(self.grid_margin + row * cell_height)
                x2 = int(x1 + cell_width)
                y2 = int(y1 + cell_height)

                note_idx = row * self.grid_cols + col

                if (row, col) == self.cursor_position:
                    color = (0, 255, 0) if not self.trigger_active else (0, 255, 255)
                    thickness = 3
                else:
                    color = (100, 100, 100)
                    thickness = 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                note_name = self.note_names.get(note_idx, "")
                text_size = cv2.getTextSize(note_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = x1 + (x2 - x1 - text_size[0]) // 2
                text_y = y1 + (y2 - y1 + text_size[1]) // 2

                cv2.putText(frame, note_name, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Print mode-specific instructions
        print(f"\n🎵 Gesture-Controlled Music System V2 - {self.mode.upper()} MODE")
        print("=" * 60)

        if self.mode == "keyboard":
            print("Controls:")
            print("  HAND: Move index finger to select notes")
            print("  SPACEBAR: Press to play selected note")
        elif self.mode == "two_hand":
            print("Controls:")
            print("  LEFT HAND: Move index finger to select notes")
            print("  RIGHT HAND: Close fist to play selected note")
        elif self.mode == "gesture":
            print("Controls:")
            print("  HAND: Move index finger to select notes")
            print("  GESTURES:")
            print("    ✌️  Peace Sign: Play note")
            print("    ✊  Closed Fist: Stop")
            print("    ✋  Open Hand: Play note")

        print("  Q: Quit application")
        print(f"\nGrid Layout: {self.grid_rows}x{self.grid_cols} notes")
        print("=" * 60)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            cursor_hand = None
            trigger_hand = None

            # Process hands based on mode
            if results.multi_hand_landmarks:
                if self.mode == "two_hand" and results.multi_handedness:
                    # Two-hand mode: separate cursor and trigger hands
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

                        label = handedness.classification[0].label

                        if label == "Left":
                            cursor_hand = hand_landmarks
                            cursor_pos = self.get_cursor_position(cursor_hand, frame.shape)
                            if cursor_pos:
                                self.cursor_position = self.map_to_grid(cursor_pos, frame.shape)
                                self.selected_note_idx = self.cursor_position[0] * self.grid_cols + self.cursor_position[1]
                                cv2.circle(frame, cursor_pos, 10, (0, 255, 0), -1)

                        elif label == "Right":
                            trigger_hand = hand_landmarks
                            is_triggered = self.detect_fist_trigger(trigger_hand)
                            self.trigger_history.append(is_triggered)

                            if len(self.trigger_history) >= 2 and sum(self.trigger_history) >= 2:
                                self.trigger_active = True
                                if self.selected_note_idx != self.last_played_note:
                                    self.play_note(self.selected_note_idx)
                                    self.last_played_note = self.selected_note_idx
                            else:
                                self.trigger_active = False
                                self.last_played_note = -1

                elif self.mode == "gesture":
                    # Gesture mode: single hand with gesture recognition
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

                        cursor_hand = hand_landmarks
                        cursor_pos = self.get_cursor_position(cursor_hand, frame.shape)
                        if cursor_pos:
                            self.cursor_position = self.map_to_grid(cursor_pos, frame.shape)
                            self.selected_note_idx = self.cursor_position[0] * self.grid_cols + self.cursor_position[1]
                            cv2.circle(frame, cursor_pos, 10, (0, 255, 0), -1)

                        # Detect gestures
                        gesture = self.detect_hand_gesture(hand_landmarks)
                        smoothed_gesture = self.smooth_gesture(gesture)

                        if smoothed_gesture == 1 or smoothed_gesture == 2:  # Peace or open hand
                            self.trigger_active = True
                            if self.selected_note_idx != self.last_played_note:
                                self.play_note(self.selected_note_idx)
                                self.last_played_note = self.selected_note_idx
                        else:
                            self.trigger_active = False
                            if smoothed_gesture == 0:  # Fist
                                self.last_played_note = -1

                elif self.mode == "keyboard":
                    # Keyboard mode: single hand for cursor only
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

                        cursor_hand = hand_landmarks
                        cursor_pos = self.get_cursor_position(cursor_hand, frame.shape)
                        if cursor_pos:
                            self.cursor_position = self.map_to_grid(cursor_pos, frame.shape)
                            self.selected_note_idx = self.cursor_position[0] * self.grid_cols + self.cursor_position[1]
                            cv2.circle(frame, cursor_pos, 10, (0, 255, 0), -1)

            # Draw grid
            self.draw_grid(frame)

            # Draw status
            y_offset = 30
            if self.mode == "two_hand":
                cv2.putText(frame, f"Left Hand: {'Active' if cursor_hand else 'Not detected'}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 0) if cursor_hand else (0, 0, 255), 2)
                y_offset += 25
                cv2.putText(frame, f"Right Hand: {'Active' if trigger_hand else 'Not detected'}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 0) if trigger_hand else (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Hand: {'Active' if cursor_hand else 'Not detected'}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 0) if cursor_hand else (0, 0, 255), 2)

            # Draw note info
            if self.selected_note_idx in self.note_names:
                note_info = f"Selected: {self.note_names[self.selected_note_idx]}"
                cv2.putText(frame, note_info, (10, frame.shape[0] - 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Draw trigger status
            trigger_status = "PLAYING!" if self.trigger_active else "Ready"
            trigger_color = (0, 255, 255) if self.trigger_active else (200, 200, 200)
            cv2.putText(frame, f"Status: {trigger_status}", (10, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, trigger_color, 2)

            # Draw instructions
            instruction = "Press SPACEBAR to play | Q to quit" if self.mode == "keyboard" else "Q to quit"
            cv2.putText(frame, instruction, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Gesture Music System V2', frame)

            # Keyboard handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' ') and self.mode == "keyboard":
                if not self.space_pressed:
                    self.space_pressed = True
                    self.trigger_active = True
                    if self.selected_note_idx != self.last_played_note:
                        self.play_note(self.selected_note_idx)
                        self.last_played_note = self.selected_note_idx
            else:
                if self.space_pressed and self.mode == "keyboard":
                    self.space_pressed = False
                    self.trigger_active = False
                    self.last_played_note = -1

        cap.release()
        cv2.destroyAllWindows()
        sd.stop()
        self.hands.close()


def select_mode():
    """Ask user to select control mode"""
    print("\n" + "="*60)
    print("🎵 GESTURE-CONTROLLED MUSIC SYSTEM V2 🎵")
    print("="*60)
    print("\nSelect your control mode:\n")
    print("  1️⃣  KEYBOARD MODE")
    print("      - Move hand to select notes")
    print("      - Press SPACEBAR to play")
    print("      - Easiest and most precise\n")

    print("  2️⃣  TWO-HAND MODE")
    print("      - Left hand: Move to select notes")
    print("      - Right hand: Close fist to play")
    print("      - Original two-hand interface\n")

    print("  3️⃣  GESTURE MODE")
    print("      - Move hand to select notes")
    print("      - Make gestures to play:")
    print("        ✌️  Peace sign → Play")
    print("        ✋  Open hand → Play")
    print("        ✊  Closed fist → Stop")
    print("      - Most expressive\n")

    print("="*60)

    while True:
        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == "1":
            return "keyboard"
        elif choice == "2":
            return "two_hand"
        elif choice == "3":
            return "gesture"
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        mode = select_mode()
        system = GestureMusicSystemV2(grid_size=(5, 5), mode=mode)
        system.run()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have installed all required packages:")
        print("pip install mediapipe opencv-python numpy sounddevice")

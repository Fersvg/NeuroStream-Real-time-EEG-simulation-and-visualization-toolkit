# NeuroStream: A real-time EEG simulation and visualization toolkit using **LSL**
This repository contains two scripts:

- `Emitter.py`: generates a realistic multi-channel EEG-like signal and streams it over LSL.
- `Receiver.py`: receives the EEG stream, visualizes time-domain channels, applies real-time filters, and computes PSD/bandpower.

---

## Features

### `Emitter.py` (EEG Simulator)

- 8-channel correlated EEG-like output
- 10-20 inspired spatial topography (`Fp1`, `Fp2`, `C3`, `C4`, `P3`, `P4`, `O1`, `O2`)
- Theta/alpha/beta rhythms with:
  - continuous phase
  - slow frequency random-walk
  - stochastic band-limited components
- Stateful 1/f (pink) noise
- Slow baseline drift
- Eye-blink artifacts (frontal dominant)
- EMG burst artifacts
- LSL streaming at `250 Hz`

### `Receiver.py` (Real-time Viewer)

- Real-time multi-channel EEG plot
- Real-time PSD (Welch)
- Bandpower display (Theta/Alpha/Beta) in `µV²`
- Notch filter selector: `Off / 50 Hz / 60 Hz`
- Dynamic bandpass control:
  - `lowcut` (Hz)
  - `highcut` (Hz)
- PSD display selector:
  - `Relative dB`
  - `Absolute µV²/Hz`
- Gain control for amplitude scaling

---

## Requirements

- Python 3.9+ (recommended)
- Dependencies:
  - `numpy`
  - `scipy`
  - `pyqtgraph`
  - `pylsl`
  - `PyQt5` (or a compatible Qt binding supported by pyqtgraph)

Install:

```bash
pip install numpy scipy pyqtgraph pylsl PyQt5
```

---

## How to Run

Open two terminals in the project root.

### 1) Start the EEG stream (Emitter)

```bash
python Emitter.py
```

Expected output (example):

```text
Streaming realistic EEG (8 channels) at 250 Hz...
```

### 2) Start the viewer (Receiver)

```bash
python Receiver.py
```

The receiver will discover the LSL EEG stream and open the UI.

---

## Receiver UI Controls

- **Notch**: remove mains interference (`50 Hz` or `60 Hz`)
- **Bandpass (low/high)**:
  - enable/disable bandpass filter
  - choose lower and upper cutoff frequencies in real time
- **PSD display**:
  - `Relative dB`: useful for peak comparison (dominant band near 0 dB)
  - `Absolute µV²/Hz`: physical PSD units
- **Gain (x)**: scales time-domain traces

---

## Interpreting the PSD

- In `Relative dB` mode, negative values are normal (they are relative to the current maximum).
- In `Absolute µV²/Hz` mode, PSD values represent absolute power density.
- Bandpower text (Theta/Alpha/Beta) is shown as integrated absolute power in `µV²`.

---

## Typical Workflow

1. Start `Emitter.py`
2. Start `Receiver.py`
3. In the receiver:
   - set notch according to local mains frequency
   - tune `lowcut/highcut` depending on analysis target
   - switch PSD mode depending on whether you want relative peak visibility or absolute units

---

## Notes

- This project is a **realistic EEG-like simulator**, not a medical device.
- Parameters are tuned for realistic visual/spectral behavior and can be adjusted in code.

---

## License

MIT.


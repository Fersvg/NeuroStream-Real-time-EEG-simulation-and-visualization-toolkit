"""
EEG Receiver (LSL: Labs Streaming Layer)) real-time viewer with filters and PSD.

This script:
1) Subscribes to an LSL stream of type ``EEG``.
2) Maintains a sliding circular buffer (time window) for multi-channel traces.
3) Optionally applies zero-phase IIR filters (notch, bandpass) and estimates PSD with Welch.

PSD modes (panel selector):
- Relative dB: curve is 10*log10(PSD), then shifted so the current peak is 0 dB.
- Absolute µV²/Hz: raw Welch power spectral density in physical units.

Bandpower overlay: integrated power (Theta / Alpha / Beta) in µV² (absolute).

Author: Fernando Sala-Vivé Gallego
"""

import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import StreamInlet, resolve_stream
from scipy.signal import butter, filtfilt, iirnotch, welch


# =========================
# CONFIG
# =========================
FS = 250
WINDOW_SECONDS = 4
BUFFER_SIZE = FS * WINDOW_SECONDS

DEFAULT_LOW = 1.0
DEFAULT_HIGH = 40.0
DEFAULT_NOTCH = 50.0
Q = 30  # notch quality factor

# Fixed vertical range for EEG traces (µV); gain slider scales the displayed amplitude
Y_RANGE = 250

frame_count = 0
fps_frame_count = 0

# PSD x-axis limit (Hz)
PSD_MAX_FREQ = 80.0

# PSD display mode labels
PSD_MODE_REL_DB = "Relative dB"
PSD_MODE_ABS_UV = "Absolute µV²/Hz"


# =========================
# FILTER DESIGN (dynamic coefficients; recalc when UI changes)
# =========================
def design_bandpass(lowcut, highcut, fs=FS, order=4):
    """
    Butterworth bandpass. Normalized cutoffs are clamped for numerical stability
    when the user moves spin boxes quickly.
    """
    nyq = fs / 2.0
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    low = max(1e-4, min(low, 0.999))
    high = max(1e-4, min(high, 0.999))
    if low >= high:
        low = min(low, 0.9)
        high = max(high, low + 1e-3)
    b, a = butter(order, [low, high], btype="band")
    return b, a


def design_notch(notch_freq, fs=FS):
    """IIR notch at ``notch_freq`` Hz (e.g. 50 or 60)."""
    w0 = float(notch_freq) / (fs / 2.0)
    return iirnotch(w0, Q)


lowcut_curr = DEFAULT_LOW
highcut_curr = DEFAULT_HIGH
notch_freq_curr = DEFAULT_NOTCH

b_bp, a_bp = design_bandpass(lowcut_curr, highcut_curr)
b_notch, a_notch = design_notch(notch_freq_curr)


def bandpass(data):
    """Zero-phase bandpass along the time axis (axis=1)."""
    return filtfilt(b_bp, a_bp, data, axis=1)


def notch_filter(data):
    """Zero-phase notch along the time axis (axis=1)."""
    return filtfilt(b_notch, a_notch, data, axis=1)


# =========================
# PSD (Welch)
# =========================
def compute_psd(signal):
    """
    Welch PSD. ``signal`` can be 1D or 2D (n_channels × n_samples); axis=-1 is time.
    """
    return welch(signal, FS, nperseg=FS * 2, noverlap=FS, axis=-1)


# =========================
# LSL
# =========================
print("Searching for EEG stream...")
streams = resolve_stream("type", "EEG")

inlet = StreamInlet(streams[0])
n_channels = inlet.info().channel_count()
print(f"Connected: {n_channels} channels")


# =========================
# BUFFER
# =========================
buffer = np.zeros((n_channels, BUFFER_SIZE))
ptr = 0
current_time = 0.0

# Fixed relative time axis (seconds); shifted by (current_time - WINDOW_SECONDS)
time_axis_fixed = np.arange(BUFFER_SIZE, dtype=np.float64) / FS

# Cached filtered window; recompute periodically and when filter params change
filtered_data_cache = np.zeros((n_channels, BUFFER_SIZE), dtype=np.float64)
filter_dirty = True


# =========================
# PyQtGraph window
# =========================
pg.setConfigOptions(antialias=True)

app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="NeuroStream: Real-time EEG")
win.resize(1400, 800)
win.show()

channel_names = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
plots = []
curves = []

for ch in range(min(8, n_channels)):
    p = win.addPlot(row=ch, col=0)
    p.setYRange(-Y_RANGE, Y_RANGE)
    p.enableAutoRange(axis="y", enable=False)
    p.showGrid(x=True, y=True, alpha=0.3)

    label = pg.TextItem(channel_names[ch], anchor=(0, 0.5), color="w")
    p.addItem(label)
    label.setPos(0, 0)

    p.setLabel("left", "µV")
    p.addLine(y=0, pen=pg.mkPen((100, 100, 100), width=1))

    curve = p.plot(pen=pg.mkPen("g", width=1))
    plots.append(p)
    curves.append(curve)

for ch in range(1, min(8, n_channels)):
    plots[ch].setXLink(plots[0])


# PSD plot spans all EEG rows so row heights stay even
psd_plot = win.addPlot(row=0, col=1, rowspan=min(8, n_channels), title="PSD (Welch)")
psd_curve = psd_plot.plot(pen="y")

psd_plot.setLabel("left", "Power (dB/Hz, rel.)")
psd_plot.setLabel("bottom", "Hz")
psd_plot.setXRange(0, PSD_MAX_FREQ)
psd_plot.setLogMode(y=False)

psd_plot.addItem(pg.LinearRegionItem([4, 8], brush=(50, 50, 150, 50)))
psd_plot.addItem(pg.LinearRegionItem([8, 12], brush=(50, 150, 50, 50)))
psd_plot.addItem(pg.LinearRegionItem([13, 30], brush=(150, 50, 50, 50)))

bandpower_text = pg.TextItem(color="w")
psd_plot.addItem(bandpower_text)
bandpower_text.setPos(25, 20)


# =========================
# Control panel
# =========================
control_widget = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout()

notch_combo = QtWidgets.QComboBox()
notch_combo.addItems(["Off", "50 Hz", "60 Hz"])
notch_combo.setCurrentIndex(1)

bandpass_checkbox = QtWidgets.QCheckBox("Bandpass (low/high)")
bandpass_checkbox.setChecked(True)

lowcut_spin = QtWidgets.QDoubleSpinBox()
lowcut_spin.setRange(0.1, 20.0)
lowcut_spin.setDecimals(1)
lowcut_spin.setSingleStep(0.5)
lowcut_spin.setValue(DEFAULT_LOW)

highcut_spin = QtWidgets.QDoubleSpinBox()
highcut_spin.setRange(5.0, float(PSD_MAX_FREQ))
highcut_spin.setDecimals(1)
highcut_spin.setSingleStep(0.5)
highcut_spin.setValue(DEFAULT_HIGH)

psd_mode_combo = QtWidgets.QComboBox()
psd_mode_combo.addItems([PSD_MODE_REL_DB, PSD_MODE_ABS_UV])
psd_mode_combo.setCurrentText(PSD_MODE_REL_DB)

scale_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
# Maps [5..35] → gain [0.25 .. 1.75]; center = 1.0 at 20
scale_slider.setMinimum(5)
scale_slider.setMaximum(35)
scale_slider.setValue(20)

layout.addWidget(notch_combo)
layout.addWidget(bandpass_checkbox)
layout.addWidget(QtWidgets.QLabel("PSD display"))
layout.addWidget(psd_mode_combo)
layout.addWidget(QtWidgets.QLabel("Lowcut (Hz)"))
layout.addWidget(lowcut_spin)
layout.addWidget(QtWidgets.QLabel("Highcut (Hz)"))
layout.addWidget(highcut_spin)
layout.addWidget(QtWidgets.QLabel("Gain (x)"))
layout.addWidget(scale_slider)

control_widget.setLayout(layout)

proxy = QtWidgets.QGraphicsProxyWidget()
proxy.setWidget(control_widget)

plot_rows = min(8, n_channels)
control_row = plot_rows
win.addItem(proxy, row=control_row, col=1)


def apply_bandpass_params():
    """Recompute bandpass when low/high cutoffs change."""
    global lowcut_curr, highcut_curr, b_bp, a_bp, filter_dirty
    low = float(lowcut_spin.value())
    high = float(highcut_spin.value())

    if low >= high:
        if high - 0.5 >= 0.1:
            low = high - 0.5
            lowcut_spin.blockSignals(True)
            lowcut_spin.setValue(low)
            lowcut_spin.blockSignals(False)
        else:
            high = low + 0.5
            highcut_spin.blockSignals(True)
            highcut_spin.setValue(high)
            highcut_spin.blockSignals(False)

    lowcut_curr = float(lowcut_spin.value())
    highcut_curr = float(highcut_spin.value())
    b_bp, a_bp = design_bandpass(lowcut_curr, highcut_curr)
    filter_dirty = True


def apply_notch_params():
    """Recompute notch when frequency or Off is selected."""
    global notch_freq_curr, b_notch, a_notch, filter_dirty
    idx = notch_combo.currentIndex()
    if idx == 0:
        filter_dirty = True
        return
    freq = 50.0 if idx == 1 else 60.0
    notch_freq_curr = freq
    b_notch, a_notch = design_notch(notch_freq_curr)
    filter_dirty = True


def apply_psd_mode():
    """Switch Y-axis label and default Y behavior when PSD display mode changes."""
    mode = psd_mode_combo.currentText()
    if mode == PSD_MODE_ABS_UV:
        psd_plot.setLabel("left", "Power (µV²/Hz)")
        psd_plot.enableAutoRange(axis="y", enable=True)
    else:
        psd_plot.setLabel("left", "Power (dB/Hz, rel.)")
        psd_plot.enableAutoRange(axis="y", enable=False)
        psd_plot.setYRange(-80, 5)


lowcut_spin.valueChanged.connect(apply_bandpass_params)
highcut_spin.valueChanged.connect(apply_bandpass_params)
notch_combo.currentIndexChanged.connect(apply_notch_params)
psd_mode_combo.currentIndexChanged.connect(apply_psd_mode)

apply_bandpass_params()
apply_notch_params()
apply_psd_mode()


last_time = time.time()


def update():
    global buffer, ptr, current_time, frame_count, last_time, fps_frame_count
    global filtered_data_cache, filter_dirty

    frame_count += 1

    chunk, _ = inlet.pull_chunk(timeout=0.01)
    if chunk:
        chunk = np.array(chunk).T
        n_samples = chunk.shape[1]
        idx = (ptr + np.arange(n_samples)) % BUFFER_SIZE
        buffer[:, idx] = chunk
        ptr = (ptr + n_samples) % BUFFER_SIZE
        current_time += n_samples / FS

    data = np.hstack((buffer[:, ptr:], buffer[:, :ptr]))
    time_axis = time_axis_fixed + (current_time - WINDOW_SECONDS)

    # Recompute filtered window every ~3 frames, or immediately after filterDirty
    if filter_dirty or (frame_count % 3 == 0):
        temp = data
        if notch_combo.currentIndex() != 0:
            temp = notch_filter(temp)
        if bandpass_checkbox.isChecked():
            temp = bandpass(temp)
        filtered_data_cache = temp
        filter_dirty = False
    else:
        temp = filtered_data_cache

    gain = scale_slider.value() / 20.0

    for ch in range(min(8, n_channels)):
        curves[ch].setData(time_axis, temp[ch] * gain)
    plots[0].setXRange(current_time - WINDOW_SECONDS, current_time)

    if frame_count % 5 == 0:
        freqs, psd = compute_psd(temp)
        psd_mean = np.mean(psd, axis=0)

        mode = psd_mode_combo.currentText()
        if mode == PSD_MODE_ABS_UV:
            psd_curve.setData(freqs, psd_mean)
            pos_vals = psd_mean[psd_mean > 0]
            if pos_vals.size > 0:
                y_top = float(np.percentile(pos_vals, 99.0) * 1.2)
                y_top = max(y_top, 1e-3)
                psd_plot.setYRange(0.0, y_top)
        else:
            eps = 1e-12
            psd_db = 10.0 * np.log10(psd_mean + eps)
            psd_disp = psd_db - float(np.max(psd_db))
            psd_curve.setData(freqs, psd_disp)

        theta_mask = (freqs >= 4) & (freqs <= 8)
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        beta_mask = (freqs >= 13) & (freqs <= 30)

        theta_p = float(np.trapezoid(psd_mean[theta_mask], freqs[theta_mask]))
        alpha_p = float(np.trapezoid(psd_mean[alpha_mask], freqs[alpha_mask]))
        beta_p = float(np.trapezoid(psd_mean[beta_mask], freqs[beta_mask]))

        bandpower_text.setText(
            f"Theta {theta_p:.1f} µV²  |  Alpha {alpha_p:.1f} µV²\nBeta {beta_p:.1f} µV²"
        )

    now = time.time()
    fps = 1 / (now - last_time)
    last_time = now
    fps_frame_count += 1
    if fps_frame_count % 10 == 0:
        win.setWindowTitle(f"NeuroStream | FPS: {fps:.1f}")


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(16)

app.exec()

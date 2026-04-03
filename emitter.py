"""
EEG Emitter (LSL: Labs Streaming Layer) realistic multi-channel EEG simulator.

This script:
1) Synthesizes theta/alpha/beta-like sources with continuous phase and stochastic bands.
2) Mixes them spatially (approximate 10–20 topography) into 8 channels.
3) Adds pink 1/f noise, baseline drift, eye-blink and EMG-like artifacts, and streams
   samples over LSL at ``FS`` Hz for consumption by ``Receiver.py`` (or any LSL client).

Approximate units:
- Samples are in microvolts (µV) per channel, plus modeled amplifier noise.

Signal model (conceptual):

    EEG(t) = M · S(t) + N_pink(t) + N_white(t) + A(t)

where ``S`` are band sources, ``M`` is the mixing matrix, ``N_pink`` / ``N_white`` are
noise terms, and ``A`` groups blink, EMG, and drift.

Author: Fernando Sala-Vivé Gallego
"""

import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock


# =========================
# CONFIG
# =========================
FS = 250  # sampling rate (Hz)
N_CHANNELS = 8
DT = 1.0 / FS  # seconds per sample

# Eye blink (rare frontal-dominant events)
BLINK_RATE_PER_SEC = 0.08  # ~1 blink every ~12.5 s on average
BLINK_DURATION_SEC = 0.25
BLINK_AMPLITUDE_UV = 95.0

# EMG (sporadic high-frequency bursts)
EMG_RATE_PER_SEC = 0.05
EMG_DURATION_SEC = 0.45
EMG_AMPLITUDE_UV = 18.0
EMG_HP_CUTOFF_HZ = 25.0  # shapes EMG toward higher frequencies

# Noise
AMP_NOISE_STD_UV = 0.6  # white amplifier noise (std, µV)
PINK_NOISE_SCALE = 0.9  # scales 1/f background

# Deterministic source amplitudes (µV)
THETA_DET_AMP_UV = 16.0
ALPHA_DET_AMP_UV = 22.0
BETA_DET_AMP_UV = 20.0

# Stochastic band-limited components (µV scale)
THETA_NOISE_AMP = 3.5
ALPHA_NOISE_AMP = 4.2
BETA_NOISE_AMP = 6.0


# =========================
# GLOBAL STATE (RNG + oscillators + noise / artifact state machines)
# =========================
rng = np.random.default_rng()

# Oscillator phases (theta, alpha, beta) continuous evolution
osc_phases = rng.uniform(0, 2 * np.pi, size=3)

# Instantaneous band center frequencies (random-walk, clamped)
theta_freq_curr = 6.0
alpha_freq_curr = 10.0
beta_freq_curr = 20.0

# AR(1) envelope states (smooth amplitude modulation)
alpha_env_state = 0.0
theta_env_state = 0.0

# Pink noise (Paul Kellet-style), per-channel state
pink_b0 = np.zeros(N_CHANNELS)
pink_b1 = np.zeros(N_CHANNELS)
pink_b2 = np.zeros(N_CHANNELS)

# Blink state machine
blink_age_s = 0.0
blink_active = False

# EMG burst + first-order high-pass state
emg_age_s = 0.0
emg_active = False
emg_hp_alpha = float(np.exp(-2 * np.pi * EMG_HP_CUTOFF_HZ * DT))
emg_prev_x = np.zeros(N_CHANNELS)
emg_prev_y = np.zeros(N_CHANNELS)

# Band-limited stochastic resonator (per band)
res_y1 = np.zeros(3)
res_y2 = np.zeros(3)
res_damping = np.array([0.995, 0.993, 0.994], dtype=np.float64)  # theta, alpha, beta


# =========================
# LSL STREAM
# =========================
def create_lsl_stream():
    """
    Create an LSL (Lab Streaming Layer) outlet.

    LSL allows real-time streaming of time-series data between applications.
    Other programs (e.g., Receiver.py) can subscribe to this stream.
    """
    info = StreamInfo(
        name='EEG_Realistic',
        type='EEG',
        channel_count=N_CHANNELS,
        nominal_srate=FS,
        channel_format='float32',
        source_id='neurostream_realistic_v3'
    )
    return StreamOutlet(info)


# =========================
# SOURCES (base band frequencies; extensibility hook)
# =========================
def initialize_sources():
    """
    Base frequencies for EEG bands.

    Note: currently not dynamically used (frequencies evolve independently),
    but kept for extensibility (e.g., external control).
    """
    return {
        "theta_freq": 6,
        "alpha_freq": 10,
        "beta_freq": 20
    }


# =========================
# MIXING MATRIX (topography / 10–20 style)
# =========================
def create_mixing_matrix():
    """
    Spatial mixing matrix (approximate 10–20 system).

    Maps 3 neural sources → 8 EEG channels.
    Introduces spatial correlation between channels.
    """
    return np.array([
        [0.8, 0.3, 0.2],  # Fp1 (frontal → more theta)
        [0.8, 0.3, 0.2],  # Fp2
        [0.5, 0.5, 0.3],  # C3
        [0.5, 0.5, 0.3],  # C4
        [0.3, 0.7, 0.4],  # P3
        [0.3, 0.7, 0.4],  # P4
        [0.2, 1.2, 0.2],  # O1 (occipital → more alpha)
        [0.2, 1.2, 0.2],  # O2
    ])


# =========================
# SIGNAL GENERATION
# =========================
def generate_brain_sources(t, sources):
    """
    Generate neural sources (theta/alpha/beta).

    Each band = deterministic sinusoid + stochastic band-limited component.

    - Continuous phase ensures temporal realism
    - Resonator-based noise broadens the spectrum
    """

    global osc_phases
    global theta_freq_curr, alpha_freq_curr, beta_freq_curr
    global alpha_env_state, theta_env_state
    global res_y1, res_y2

    # AR(1) envelope → smooth amplitude modulation
    alpha_env_state = 0.999 * alpha_env_state + rng.normal(0.0, 0.003)
    theta_env_state = 0.999 * theta_env_state + rng.normal(0.0, 0.002)

    alpha_mod = max(0.35, 1.0 + 0.35 * alpha_env_state)
    theta_mod = max(0.35, 1.0 + 0.25 * theta_env_state)

    # Random-walk frequencies (slow drift)
    theta_freq_curr += rng.normal(0.0, 0.002)
    alpha_freq_curr += rng.normal(0.0, 0.003)
    beta_freq_curr += rng.normal(0.0, 0.006)

    # Clamp ranges (physiological bounds)
    theta_freq_curr = min(7.2, max(5.2, theta_freq_curr))
    alpha_freq_curr = min(11.8, max(8.2, alpha_freq_curr))
    beta_freq_curr = min(24.0, max(14.0, beta_freq_curr))

    # Continuous phase update
    osc_phases += 2 * np.pi * np.array([theta_freq_curr, alpha_freq_curr, beta_freq_curr]) * DT
    osc_phases %= (2 * np.pi)

    # Small phase noise → avoids perfect periodicity
    osc_phases += rng.normal(0.0, [0.0015, 0.0012, 0.0020])

    # Deterministic components
    theta_det = THETA_DET_AMP_UV * theta_mod * np.sin(osc_phases[0])
    alpha_det = ALPHA_DET_AMP_UV * alpha_mod * np.sin(osc_phases[1])
    beta_det = BETA_DET_AMP_UV * np.sin(osc_phases[2])

    # Resonator (band-limited stochastic process)
    # Produces noise centered around each band frequency
    freqs = [theta_freq_curr, alpha_freq_curr, beta_freq_curr]
    noises = np.zeros(3)

    for i in range(3):
        w = 2 * np.pi * freqs[i] / FS
        coef = 2 * np.cos(w)

        x = rng.standard_normal()
        y = coef * res_y1[i] - res_damping[i] * res_y2[i] + x

        res_y2[i] = res_y1[i]
        res_y1[i] = y
        noises[i] = y

    theta_noise = THETA_NOISE_AMP * theta_mod * noises[0]
    alpha_noise = ALPHA_NOISE_AMP * alpha_mod * noises[1]
    beta_noise = BETA_NOISE_AMP * noises[2]

    return np.array([theta_det + theta_noise,
                     alpha_det + alpha_noise,
                     beta_det + beta_noise], dtype=np.float32)


def generate_1_f_noise_pink():
    """
    Generate approximate 1/f (pink) noise.

    Stateful → introduces temporal correlation (unlike white noise).
    """
    white = rng.standard_normal(N_CHANNELS)

    global pink_b0, pink_b1, pink_b2

    pink_b0 = 0.99765 * pink_b0 + white * 0.0990460
    pink_b1 = 0.96300 * pink_b1 + white * 0.2965164
    pink_b2 = 0.57000 * pink_b2 + white * 1.0526913

    return pink_b0 + pink_b1 + pink_b2 + white * 0.1848


def generate_drift(t):
    """Low-frequency baseline drift."""
    return 10.0 * np.sin(2 * np.pi * 0.20 * t) + 4.0 * np.sin(2 * np.pi * 0.07 * t)


def generate_blink(t):
    """Eye blink artifact (frontal dominant)."""

    global blink_age_s, blink_active

    # Poisson-like triggering
    if not blink_active and rng.random() < (BLINK_RATE_PER_SEC / FS):
        blink_active = True
        blink_age_s = 0.0

    if not blink_active:
        return np.zeros(N_CHANNELS, dtype=np.float32)

    duration = BLINK_DURATION_SEC
    sigma = duration / 6.0
    center = duration / 8.0

    env = np.exp(-0.5 * ((blink_age_s - center) / sigma) ** 2)

    pattern = np.array([1.0, 1.0, 0.3, 0.3, 0.1, 0.1, 0.05, 0.05])
    blink = BLINK_AMPLITUDE_UV * env * pattern

    blink_age_s += DT
    if blink_age_s >= duration:
        blink_active = False
        blink_age_s = 0.0

    return blink


def generate_emg():
    """High-frequency EMG bursts (muscle noise)."""

    global emg_age_s, emg_active, emg_prev_x, emg_prev_y

    if not emg_active and rng.random() < (EMG_RATE_PER_SEC / FS):
        emg_active = True
        emg_age_s = 0.0

    if not emg_active:
        return np.zeros(N_CHANNELS, dtype=np.float32)

    duration = EMG_DURATION_SEC
    tau = duration / 4.0

    env = np.exp(-emg_age_s / tau)

    x = rng.standard_normal(N_CHANNELS)

    # First-order high-pass IIR filter
    y = emg_hp_alpha * (emg_prev_y + x - emg_prev_x)

    emg_prev_x = x
    emg_prev_y = y

    emg = EMG_AMPLITUDE_UV * env * y

    emg_age_s += DT
    if emg_age_s >= duration:
        emg_active = False
        emg_prev_x[:] = 0.0
        emg_prev_y[:] = 0.0

    return emg.astype(np.float32)


# =========================
# SAMPLE GENERATION
# =========================
def generate_sample(t, sources, mixing_matrix):
    """
    Generate one EEG sample (all channels).

    Pipeline:
    brain sources → spatial mixing → add noise/artifacts
    """
    brain = generate_brain_sources(t, sources)

    eeg = mixing_matrix @ brain

    eeg += PINK_NOISE_SCALE * generate_1_f_noise_pink()
    eeg += generate_drift(t)
    eeg += generate_blink(t)
    eeg += generate_emg()
    eeg += rng.standard_normal(N_CHANNELS) * AMP_NOISE_STD_UV

    return eeg.astype(np.float32)


# =========================
# MAIN
# =========================
def main():
    outlet = create_lsl_stream()
    sources = initialize_sources()
    mixing_matrix = create_mixing_matrix()

    print(f"Streaming realistic EEG ({N_CHANNELS} channels) at {FS} Hz...")

    t = 0.0  # continuous simulation time

    try:
        while True:
            start = time.perf_counter()

            sample = generate_sample(t, sources, mixing_matrix)

            # Send sample with precise timestamp
            outlet.push_sample(sample.tolist(), local_clock())

            t += DT

            # Maintain real-time sampling rate (FS)
            elapsed = time.perf_counter() - start
            time.sleep(max(0, DT - elapsed))

    except KeyboardInterrupt:
        print("\nStreaming stopped.")


if __name__ == "__main__":
    main()

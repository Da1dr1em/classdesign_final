"""
Test Data Generator

Generate test audio signals with and without noise.
"""

import numpy as np
import soundfile as sf
import os

def generate_sine_wave(freq, dur=3.0):
    t = np.linspace(0, dur, int(44100*dur))
    return 0.8*np.sin(2*np.pi*freq*t)

def main():
    os.makedirs('examples\data', exist_ok=True)

    # 500Hz signal
    sig1 = generate_sine_wave(500, 3.0)
    sf.write('examples\data\sig1_500hz.wav', sig1, 44100)
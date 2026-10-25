"""Finds the key of a record with essentia's key profile for electronic dance music (Faraldo et al., 2016).

Drums carry no key and blur the estimate, so it listens to the other stems; the bass alone is
not enough either, since a bass line often rests on a note other than the tonic.
"""

import numpy as np

PITCH_CLASSES = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}  # fmt: skip
# the spellings DJs and key charts use
NAMES = {
    "minor": ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"],
    "major": ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
}


def camelot(pitch_class: int, scale: str) -> str:
    """Position on the Camelot wheel: A minor is 8A, C major 8B; a step is a fifth."""
    tonic_of_8 = 9 if scale == "minor" else 0
    number = ((pitch_class - tonic_of_8) * 7 + 7) % 12 + 1
    return f"{number}{'A' if scale == 'minor' else 'B'}"


def detect_key(stems: dict[str, np.ndarray], samplerate: int) -> dict:
    """key ("A minor"), camelot ("8A") and key_strength (0..1, how clearly the audio follows the key)."""
    import essentia

    essentia.log.infoActive = False
    essentia.log.warningActive = False
    import essentia.standard as es

    audio = sum(audio for stem, audio in stems.items() if stem != "drums")
    mono = (audio.mean(axis=1) if audio.ndim == 2 else audio).astype(np.float32)
    tonic, scale, strength = es.KeyExtractor(profileType="edma", sampleRate=samplerate)(mono)
    pitch_class = PITCH_CLASSES[tonic]
    return {
        "key": f"{NAMES[scale][pitch_class]} {scale}",
        "camelot": camelot(pitch_class, scale),
        "key_strength": round(float(strength), 2),
    }

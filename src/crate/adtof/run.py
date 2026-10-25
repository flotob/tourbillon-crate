# SPDX-License-Identifier: MIT
"""Runs ADTOF on one drums stem and writes its hits as json: {"35": [[seconds, confidence], ...], ...},
keyed by General MIDI drum (35 kick, 38 snare, 42 hi-hat, 47 toms, 49 cymbals).

Runs inside ADTOF's own environment and is never imported by crate (see crate/drums.py). Unlike the
rest of crate it is MIT licensed, since it runs together with ADTOF, which is CC BY-NC-SA 4.0.

    python run.py <audio> <output.json>
"""

import json
import logging
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.disable(logging.WARNING)

from adtof.model.model import Model
from adtof.model.peakPicking import PeakPicking
from adtof.model.track import Track


def main(audio: str, output: str):
    # the model of the paper, trained on all of ADTOF's data, with a peak threshold per drum
    model, hparams = Model.modelFactory(modelName="Frame_RNN", scenario="adtofAll", fold=0)
    if not model.weightLoadedFlag:
        raise SystemExit("ADTOF's model weights are missing")
    activations = model.predict(Track(audio, **hparams), **hparams)
    picker = PeakPicking()
    picker.setParameters(**hparams)
    onsets = picker.predict([activations], **hparams)[0]
    rate, last = hparams["sampleRate"], len(activations) - 1
    hits = {
        str(label): [[round(t, 4), round(float(activations[min(last, round(t * rate)), i]), 3)] for t in onsets[label]]
        for i, label in enumerate(hparams["labels"])
    }
    with open(output, "w") as f:
        json.dump(hits, f)


if __name__ == "__main__":
    main(*sys.argv[1:3])

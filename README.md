# CE-I-GLA
<div align="center">
  
**Color-Enhanced Image Griffin-Lim Algorithm: Image Sonification with Learned Audio Synthesis**

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-audio%20DSP-8A2BE2)

CE-I-GLA is an active research project on image sonification: conveying what is in an image through sound. It extends the published I-GLA pipeline by combining a learned, structure-preserving audio path with a color-aware semantic path, and tests how much visual information the resulting audio actually carries.

</div>

---

## Background

I-GLA (Image Griffin-Lim Algorithm) turns an image into audio by treating it as a spectrogram and inverting it with signal-processing methods. It was published in *Signal, Image and Video Processing* (Springer Nature, 2026). In that work, pretrained CNNs classifying the reconstructed spectrograms reached up to 90.62% accuracy on a 4-class benchmark.

CE-I-GLA builds on that foundation and targets its two main limits: iterative phase reconstruction is lossy, and a purely structural mapping discards color.

## The problem

- **Spectrogram inversion is lossy.** Classical phase reconstruction trades audio fidelity for computation, and the artifacts reduce how much of the image survives.
- **Structure alone is not enough.** Shape and edges can be encoded in sound, but color is a major part of how images are understood, and structural mapping drops it.
- **Sonification must be perceivable, not just invertible.** Information has to land in the frequency range and form that human hearing and downstream models can use.

## Approach

Two complementary audio paths, fused into one stream, with a learned decoder that checks how much of the image is recoverable from the audio.

1. **Structural path.** An enhanced I-GLA pipeline that shapes spectral energy for human hearing and replaces iterative phase recovery with learned synthesis.
2. **Color semantic path.** Perceptually grounded color analysis, with color regions rendered as distinct timbres and scanned across the image.
3. **Fusion and reconstruction.** The two audio streams are combined, and a learned decoder reconstructs the image from the audio alone, a direct test of the information the sound carries.

## Research questions

1. Does learned phase estimation beat iterative reconstruction in fidelity and downstream recognition?
2. Does adding a color path measurably increase the information carried by the audio?
3. How much of the original image can a model recover from the audio alone?
4. Can people use the audio to identify colors and objects?

These are hypotheses under investigation, not claimed results.

## Evaluation

Downstream CNN classification across multiple image datasets, reconstruction quality of the recovered images, ablations isolating each component, failure-case analysis, and an informal user study.

<!-- TODO: add a Results section once experiments are complete: comparison table against the original I-GLA and a few before/after reconstruction examples. -->

## Tech stack

Python · PyTorch · Librosa

<!-- TODO: add the full I-GLA citation/DOI, a LinkedIn or contact link, and a LICENSE file. -->

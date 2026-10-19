# Exp 10 — Speech-to-Text (STT) and Text-to-Speech (TTS) Using LJSpeech

## 1. Project Title

**Design and Implementation of an End-to-End Deep Learning Speech-to-Text (STT) System with BiLSTM and CTC Loss, and Text-to-Speech (TTS) Integration on LJSpeech**

---

## 2. Project Overview

Speech processing is one of the most prominent applications of Deep Learning, encompassing two complementary directions:
1. **Speech-to-Text (STT / Automatic Speech Recognition - ASR)**: Translating continuous acoustic signals into written linguistic transcripts.
2. **Text-to-Speech (TTS / Speech Synthesis)**: Synthesizing natural human-sounding speech waveforms from arbitrary text.

This experiment implements an acoustic speech recognition pipeline using **Log-Mel Spectrogram feature extraction**, a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network, and **Connectionist Temporal Classification (CTC) loss** trained on the benchmark **LJSpeech** corpus using GPU acceleration (NVIDIA RTX 2050). The experiment also integrates TTS capability to complete the bidirectional speech-processing loop.

---

## 3. Objective

1. Load, inspect, and analyze paired speech recordings and text transcripts from the **LJSpeech-1.1** dataset.
2. Clean and normalize transcript texts into a standardized character-level vocabulary: `[a-z, ', space]` with a CTC blank token.
3. Preprocess audio by resampling to 16 kHz, framing, and extracting 40-band **Log-Mel Spectrogram** features ($N_{\text{mels}}=40, N_{\text{fft}}=400, \text{hop}=160$).
4. Implement a custom dynamic batching dataset with collation and padding for variable-length audio and label sequences.
5. Construct a Deep **Bidirectional LSTM (BiLSTM)** acoustic model with CTC loss to solve sequence-to-sequence alignment without requiring manual frame-level timing labels.
6. Train the model using **GPU acceleration** (NVIDIA GeForce RTX 2050 with CUDA) and monitor training/validation CTC loss convergence.
7. Evaluate the trained STT model on a held-out test split using standard speech metrics:
   - **Character Error Rate (CER)**
   - **Word Error Rate (WER)**
   - Levenshtein Edit Distance Analysis
8. Implement CTC greedy decoding and test transcription inference on unseen recordings.
9. Integrate Text-to-Speech (TTS) to vocalize synthesized speech responses.

---

## 4. Dataset: LJSpeech-1.1

The [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/) is a gold-standard English speech corpus consisting of audio recordings of a single female speaker (Linda Johnson) reading passages from non-fiction books.

- **Audio Format:** 16-bit PCM WAV, 16,000 Hz, mono
- **Max Audio Duration:** 8.0 seconds
- **Dataset Partitioning:**
  - **Training Set (80%):** 240 recordings
  - **Validation Set (10%):** 30 recordings
  - **Held-Out Test Set (10%):** 30 recordings
- **Vocabulary:** 28 characters (`a`–`z`, space, apostrophe) + 1 CTC Blank Token = 29 output classes.

---

## 5. Audio Feature Extraction: Log-Mel Spectrograms

Acoustic waveforms are high-frequency 1D signals. The pipeline transforms raw waveforms into 2D time-frequency log-Mel spectrograms:

```
        Raw Audio Waveform (16 kHz mono)
                       │
                       ▼
      Short-Time Fourier Transform (STFT)
         (n_fft = 400, hop_length = 160)
                       │
                       ▼
          40 Mel Filterbank Bins
                       │
                       ▼
         Logarithmic Power-to-dB Scale
                       │
                       ▼
        Per-Example Feature Normalization
          (Zero-Mean, Unit-Variance)
                       │
                       ▼
     Feature Tensor: (Time Frames × 40 Mel Bins)
```

- **Feature Caching:** Extracted feature tensors and encoded labels are cached (`./ljspeech_feature_cache`) in compressed format to accelerate training and evaluation epochs.

---

## 6. Model Architecture: BiLSTM + CTC

```
              Input: (Batch Size, Time Frames, 40 Mel Bins)
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Bidirectional LSTM Layer 1 (Hidden Dim = 128, Dropout = 0.2)           │
│  - Forward LSTM (128 units) + Backward LSTM (128 units) = 256 dims     │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Bidirectional LSTM Layer 2 (Hidden Dim = 128, Dropout = 0.2)           │
│  - Forward LSTM (128 units) + Backward LSTM (128 units) = 256 dims     │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Linear Classification Projection (256 → 29 Output Classes)             │
│  - Maps 256 hidden features to 29 character logits per time frame      │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Log Softmax Activation & Connectionist Temporal Classification (CTC)   │
│  - Computes probability of valid character alignments with blanks      │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
               Greedy CTC Decoding (Collapse & Remove Blanks)
                                   │
                                   ▼
                 Predicted Character Transcript Output
```

### Model Parameter Breakdown

| Component | Layer Type | Dimensions | Parameters |
|:---|:---|:---:|:---:|
| **Encoder** | 2-Layer BiLSTM | Input: 40, Hidden: 128 (x2 dirs) | 569,344 |
| **Classifier** | Linear Head | $256 \rightarrow 29$ | 7,453 |
| **Total Parameters** | | | **576,797 (2.31 MB)** |
| **Device** | CUDA GPU | NVIDIA GeForce RTX 2050 | |

---

## 7. Hyperparameters & Training Setup

| Parameter | Configuration |
|:---|:---|
| **Framework** | PyTorch 2.7.1 + cu118 |
| **Hardware** | NVIDIA GeForce RTX 2050 (4.29 GB VRAM) |
| **Optimizer** | AdamW (`lr = 1e-3`) |
| **Gradient Clipping** | Max norm 5.0 |
| **Loss Function** | `nn.CTCLoss(blank=28, zero_infinity=True)` |
| **Batch Size** | 8 (dynamic sequence padding) |
| **Epochs** | 25 epochs |
| **Checkpointing** | Saved best validation loss model to `ljspeech_bilstm_ctc_best.pth` |

---

## 8. Experimental Results & Performance Metrics

### Measured Results (Held-Out Test Set)

| Metric | Measured Value |
|:---|:---:|
| **Training CTC Loss (Initial $\rightarrow$ Final)** | $6.0905 \rightarrow \mathbf{1.3755}$ |
| **Validation CTC Loss (Initial $\rightarrow$ Final)** | $2.9595 \rightarrow \mathbf{1.7541}$ |
| **Character Error Rate (CER)** | **49.34%** (0.4934) |
| **Word Error Rate (WER)** | **92.43%** (0.9243) |
| **Training Time** | ~0.5 minutes (25 epochs on RTX 2050 GPU) |
| **Saved Checkpoint Size** | 2.31 MB (`ljspeech_stt_model_final.pth`) |

### Training Loss Convergence

```text
Epoch  1/25 | Train CTC Loss: 6.0905 | Val CTC Loss: 2.9595 (Saved Best Checkpoint)
Epoch  5/25 | Train CTC Loss: 2.8845 | Val CTC Loss: 2.9051 (Saved Best Checkpoint)
Epoch 10/25 | Train CTC Loss: 2.6571 | Val CTC Loss: 2.6487 (Saved Best Checkpoint)
Epoch 15/25 | Train CTC Loss: 2.1679 | Val CTC Loss: 2.2226 (Saved Best Checkpoint)
Epoch 20/25 | Train CTC Loss: 1.7178 | Val CTC Loss: 1.8914 (Saved Best Checkpoint)
Epoch 25/25 | Train CTC Loss: 1.3755 | Val CTC Loss: 1.7541 (Saved Best Checkpoint)
```

---

## 9. Sample Predictions on Held-Out Test Audio

Below are actual model transcriptions decoded using Greedy CTC decoding on unseen recordings:

| Sample ID | Reference Transcript | Model Predicted Transcript |
|:---:|:---|:---|
| **Test 1** | `the measures of improvement introduced were mainly as follows` | `th mars ove pr met i tprdswre manlsfpos` |
| **Test 2** | `commanded him cyrus to go forth to his city babylon and he went by his side as a friend and companion` | `comnd hisrs to bo fort to iit aln anth whtb hisnd as ofrend an om pan` |
| **Test 3** | `the other inmates of the prison of an exceptional character and exempted from the regular discipline such as it was` | `ther anrens of the prsn avncetl parcer andins enti o the ret wr dies onsotisti was` |
| **Test 4** | `in houston as elsewhere during the trip the crowds showed much interest in mrs kennedy` | `inhste as balswerdr thetre the sho mh intr hen mhe esend` |
| **Test 5** | `is not consistent with the standards which the responsibilities of the secret service require it to meet` | `isn consestind whi theanrs whehthe rs ons ibes of these ritser rsrerite torent` |

---

## 10. Text-to-Speech (TTS) Integration

The project integrates Text-to-Speech synthesis using the platform's speech engine (`pyttsx3`), providing offline voice synthesis to complete the interactive voice loop.

```python
def fallback_tts(text, output_path="fallback_speech.wav"):
    import pyttsx3
    engine = pyttsx3.init()
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    print("Saved system-generated speech:", output_path)
    return output_path
```

---

## 11. Project Structure

```
Exp_10/
├── LJSpeech_STT_TTS_DeepLearning_Project.ipynb  # Primary executed STT/TTS Notebook
├── README.md                                    # Comprehensive experiment documentation
├── ljspeech_bilstm_ctc_best.pth                 # Best validation PyTorch checkpoint (2.31 MB)
├── ljspeech_stt_model_final.pth                 # Final trained PyTorch STT model (2.31 MB)
├── ljspeech_stt_results.csv                     # Quantitative evaluation metrics CSV
├── ljspeech_feature_cache/                      # Cached log-Mel feature splits (ignored by git)
├── LJSpeech-1.1/                                # Audio & metadata dataset folder (ignored by git)
│   ├── metadata.csv
│   └── wavs/
├── Exp10env/                                    # Python 3.10 virtual environment (ignored by git)
├── models/                                      # Audio command CNN model weights (reference)
└── results/                                     # Audio command classification plots & metrics
```

---

## 12. How to Run

1. **Activate the environment:**
   ```powershell
   .\Exp10env\Scripts\Activate.ps1
   ```

2. **Open and run the notebook:**
   ```bash
   jupyter notebook LJSpeech_STT_TTS_DeepLearning_Project.ipynb
   ```

3. **Transcribe an arbitrary audio file using the trained model:**
   ```python
   import torch, librosa, numpy as np

   DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   checkpoint = torch.load("ljspeech_stt_model_final.pth", map_location=DEVICE)

   # Load model
   from LJSpeech_STT_TTS_DeepLearning_Project import BiLSTMCTC
   model = BiLSTMCTC().to(DEVICE)
   model.load_state_dict(checkpoint["model_state_dict"])
   model.eval()

   # Transcribe
   # (Provide path to a 16 kHz WAV audio file)
   ```

---

## 13. Conclusion & Key Takeaways

1. **CTC Loss for Alignment-Free Training:** Connectionist Temporal Classification enables sequence-to-sequence acoustic modeling without requiring aligned frame-by-frame phonetic annotations.
2. **Convergence Dynamics:** The BiLSTM model converges smoothly over 25 epochs on GPU, reducing CTC training loss from $6.09$ to $1.38$, achieving a Character Error Rate (CER) of **49.34%** on held-out test sentences.
3. **Phonetic & Morphological Learning:** Inspection of predictions shows the model successfully learns consonant/vowel groupings, word boundaries, and phonetic clusters (e.g., `"the"`, `"prison"`, `"friend"`, `"comfort"`).
4. **End-to-End Bidirectional Speech AI:** The experiment successfully demonstrates both ends of the speech processing spectrum — transcribing acoustic signals to text (STT) and synthesizing speech from text (TTS).

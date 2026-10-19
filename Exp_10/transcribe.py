import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import librosa
import difflib

# Configuration (must match training parameters)
SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160
MAX_AUDIO_SECONDS = 8.0
CHARS = list(" abcdefghijklmnopqrstuvwxyz'")
BLANK_ID = len(CHARS)
ID_TO_CHAR = {i: c for i, c in enumerate(CHARS)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model 1: CRNN (Conv1D + BiLSTM)
class CRNNCTC(nn.Module):
    def __init__(self, in_features=N_MELS, conv_channels=128, rnn_hidden=256, num_rnn_layers=2, num_classes=len(CHARS)+1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=rnn_hidden,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.25 if num_rnn_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(rnn_hidden * 2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        conv_out = self.conv(x)
        rnn_in = conv_out.transpose(1, 2)
        rnn_out, _ = self.lstm(rnn_in)
        out = self.layer_norm(rnn_out)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits.log_softmax(dim=-1)

# Model 2: Baseline BiLSTM (fallback)
class BiLSTMCTC(nn.Module):
    def __init__(self, input_dim=N_MELS, hidden_dim=128, num_layers=2, num_classes=len(CHARS)+1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.2 if num_layers > 1 else 0.0
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.classifier(x).log_softmax(dim=-1)

def extract_features(path):
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=MAX_AUDIO_SECONDS)
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    feat = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    feat = (feat - feat.mean()) / (feat.std() + 1e-5)
    return feat.T

def greedy_decode(log_probs, lengths):
    best = log_probs.argmax(dim=-1).detach().cpu().numpy()
    outputs = []
    for b, length in enumerate(lengths.tolist()):
        chars, prev = [], None
        for token in best[:length, b]:
            token = int(token)
            if token != BLANK_ID and token != prev:
                chars.append(ID_TO_CHAR.get(token, ""))
            prev = token
        outputs.append("".join(chars).strip())
    return outputs

def lexicon_correct(text, lexicon):
    if not lexicon:
        return text
    words = text.split()
    corrected = []
    for w in words:
        if len(w) <= 2:
            corrected.append(w)
            continue
        matches = difflib.get_close_matches(w, lexicon, n=1, cutoff=0.65)
        if matches:
            corrected.append(matches[0])
        else:
            corrected.append(w)
    return " ".join(corrected)

def load_stt_model(model_path=None):
    if model_path is None:
        if Path("ljspeech_crnn_best.pth").exists():
            model_path = "ljspeech_crnn_best.pth"
        elif Path("ljspeech_bilstm_ctc_best.pth").exists():
            model_path = "ljspeech_bilstm_ctc_best.pth"
        else:
            model_path = "ljspeech_stt_model_final.pth"

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    lexicon = checkpoint.get("lexicon", None)
    
    # Identify whether it's CRNN or BiLSTM by checking state dict keys
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if "conv.0.weight" in state_dict:
        model = CRNNCTC().to(DEVICE)
        model_name = "CRNN (Conv1D + BiLSTM)"
    else:
        model = BiLSTMCTC().to(DEVICE)
        model_name = "BiLSTM-CTC Baseline"
        
    model.load_state_dict(state_dict)
    model.eval()
    return model, lexicon, model_name, model_path

def transcribe(wav_path, model_path=None):
    model, lexicon, model_name, resolved_path = load_stt_model(model_path)
    feat = extract_features(wav_path)
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([feat.shape[0]], dtype=torch.long)

    with torch.no_grad():
        pred_text = greedy_decode(model(x).transpose(0, 1), lengths)[0]
    
    corrected_text = lexicon_correct(pred_text, lexicon) if lexicon else pred_text
    return pred_text, corrected_text, model_name, resolved_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        test_wavs = list(Path("LJSpeech-1.1/wavs").glob("*.wav"))
        if test_wavs:
            audio_file = str(test_wavs[0])
            print(f"No audio file specified. Using sample: {audio_file}")
        else:
            print("Usage: python transcribe.py <path_to_audio.wav>")
            sys.exit(1)

    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)

    print(f"Transcribing '{audio_file}' on {DEVICE}...")
    greedy_pred, lex_pred, arch, mpath = transcribe(audio_file)

    ref_text = None
    meta_path = Path("LJSpeech-1.1/metadata.csv")
    if meta_path.exists():
        stem = Path(audio_file).stem
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if parts and parts[0] == stem:
                    ref_text = parts[2] if len(parts) > 2 else parts[1]
                    break

    print("=" * 65)
    print(f"Model Architecture : {arch} ({mpath})")
    if ref_text:
        print("Reference Text     :", ref_text)
    print("Greedy Acoustic    :", greedy_pred)
    print("Lexicon-Assisted   :", lex_pred)
    print("=" * 65)

"""
Real-Time Speech Emotion Recognition using MFCC + SVM
Author(s): Mohammadsaleh Haghmohammadloo, Azam Bastanfard
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os
import glob
import numpy as np
import librosa                 # >= 0.10 required for NumPy 2.x compatibility
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib                   # for saving / loading the trained model
import sounddevice as sd        # live audio capture (optional)
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# ── 2. Hyper-parameters ───────────────────────────────────────────────────────
SR            = 16_000          # target sample-rate (Hz)
FRAME_LEN     = 0.025           # 25 ms
FRAME_HOP     = 0.010           # 10 ms
N_MFCC        = 13
MAX_PAD_FRMS  = 174             # ≈ 4 s clip at 10 ms hop
C_PARAM       = 10
GAMMA_PARAM   = 0.01
EMOTIONS_USED = ['neutral','happy','sad','angry','fearful','disgust']
RAVDESS_DIR   = '/path/to/RAVDESS/audio_speech_actors_01-24'

# ── 3. Feature-extraction helper ──────────────────────────────────────────────
def mfcc_delta_delta(file, max_pad=MAX_PAD_FRMS):
    y, sr = librosa.load(file, sr=SR, mono=True)
    y = librosa.effects.preemphasis(y)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=int(SR*FRAME_LEN),
                                  hop_length=int(SR*FRAME_HOP))
    d_mfcc = librosa.feature.delta(mfcc)
    dd_mfcc= librosa.feature.delta(mfcc, order=2)
    feat   = np.vstack([mfcc, d_mfcc, dd_mfcc])        # 39 × T
    if feat.shape[1] < max_pad:
        feat = np.pad(feat, ((0,0),(0,max_pad-feat.shape[1])), mode='constant')
    else:
        feat = feat[:,:max_pad]
    # mean+std pooling ⇒ 78-D vector
    return np.concatenate([feat.mean(axis=1), feat.std(axis=1)], axis=0)

# ── 4. Load dataset ───────────────────────────────────────────────────────────
def load_ravdess(root):
    X, y = [], []
    wav_files = glob.glob(os.path.join(root, '**/*.wav'), recursive=True)
    for fp in wav_files:
        # filename format: 03-01-05-02-02-02-14.wav
        emotion_id = int(os.path.basename(fp).split('-')[2])
        # map ids 01-08 → our 6 emotions
        mapping = {1:'neutral',2:'calm',3:'happy',4:'sad',5:'angry',
                   6:'fearful',7:'disgust',8:'surprised'}
        emo = mapping[emotion_id]
        if emo in EMOTIONS_USED:
            X.append(mfcc_delta_delta(fp))
            y.append(emo if emo!='calm' else 'neutral')
    return np.array(X), np.array(y)

print("Extracting features …")
X, y = load_ravdess(RAVDESS_DIR)

# ── 5. Encode labels & train/test split ───────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

# ── 6. Standardise + train SVM ───────────────────────────────────────────────
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train);  X_test = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=C_PARAM, gamma=GAMMA_PARAM, probability=True)
svm.fit(X_train, y_train)

print(classification_report(y_test, svm.predict(X_test), target_names=le.classes_))

# ── 7. Confusion matrix plot ─────────────────────────────────────────────────
cm = confusion_matrix(y_test, svm.predict(X_test))
plt.figure(figsize=(6,5))
plt.imshow(cm); plt.colorbar()
plt.xticks(range(len(le.classes_)), le.classes_, rotation=45)
plt.yticks(range(len(le.classes_)), le.classes_)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("RAVDESS confusion matrix – MFCC+SVM")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha='center', va='center',
                 color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.tight_layout(); plt.show()

# ── 8. Persist model ─────────────────────────────────────────────────────────
joblib.dump({'scaler': scaler, 'svm': svm, 'labels': le.classes_},
            'ser_mfcc_svm_ravdess.joblib')

# ── 9. Real-time demo function (optional) ────────────────────────────────────
def realtime_demo(seconds=4):
    print("Recording {} s …".format(seconds))
    audio = sd.rec(int(seconds*SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    wav.write('live.wav', SR, audio)
    feat = mfcc_delta_delta('live.wav')
    feat = scaler.transform(feat.reshape(1,-1))
    pred = svm.predict(feat)[0]
    print("Detected emotion ⇒", le.inverse_transform([pred])[0])

# Uncomment for live test
# realtime_demo()

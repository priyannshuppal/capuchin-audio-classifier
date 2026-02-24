from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import librosa
import numpy as np
import tempfile
import os

app = FastAPI()

# Load trained model (make sure model.keras is in same folder)
model = tf.keras.models.load_model("model.keras")

# ---------- Audio Helpers ----------

def load_mp3_16k_mono(filename):
    audio_data, _ = librosa.load(filename, sr=16000, mono=True)
    wav = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    return wav

def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)

    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

# ---------- API ----------

@app.get("/")
def root():
    return {"status": "Capuchin Detector API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        wav = load_mp3_16k_mono(tmp_path)

        audio_slices = tf.keras.utils.timeseries_dataset_from_array(
            wav,
            wav,
            sequence_length=48000,
            sequence_stride=48000,
            batch_size=1
        )

        audio_slices = audio_slices.map(preprocess_mp3)
        audio_slices = audio_slices.batch(64)

        preds = model.predict(audio_slices)

        # Convert probabilities to 0/1
        binary = [1 if p > 0.5 else 0 for p in preds]

        # Post-processing (same logic as notebook)
        from itertools import groupby
        capuchin_calls = int(tf.reduce_sum(
            [key for key, group in groupby(binary)]
        ).numpy())

        return {
            "filename": file.filename,
            "capuchin_calls": capuchin_calls
        }

    finally:
        os.remove(tmp_path)

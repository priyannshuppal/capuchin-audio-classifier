from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import librosa
import io

app = FastAPI()

# Load model (update path if needed)
model = tf.keras.models.load_model("model")

def preprocess_audio(file):
    audio_bytes = file.read()
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

    spec = tf.signal.stft(audio, frame_length=320, frame_step=32)
    spec = tf.abs(spec)
    spec = tf.expand_dims(spec, axis=-1)
    spec = tf.expand_dims(spec, axis=0)

    return spec

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = preprocess_audio(file.file)
    pred = model.predict(data)[0][0]

    return {
        "capuchin_probability": float(pred),
        "capuchin_detected": bool(pred > 0.5)
    }

import librosa
import numpy as np
import whisper
import sounddevice as sd

model = whisper.load_model("base")

SAMPLING_RATE = 16000
BLOCK_SIZE = 2048

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", flush=True)
    
    audio_data = indata[:, 0]
    audio_data = librosa.util.normalize(audio_data.astype(np.float32))
    
    print(f"Received audio block of {frames} frames", flush=True)
    
    result = model.transcribe(audio_data)
    transcribed_text = result['text']
    
    print(f"Transcribed Text: {transcribed_text}", flush=True)

def transcribe_realtime():
    print("Real-time transcription started... Press Ctrl+C to stop.")
    
    try:
        with sd.InputStream(samplerate=SAMPLING_RATE, channels=1, callback=audio_callback, blocksize=BLOCK_SIZE):
            while True:
                pass 
            
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        sd.stop()
        print("Transcription stopped.")

if __name__ == "__main__":
    transcribe_realtime()

import torch
import threading
import time
import numpy as np
import sounddevice as sd
from rich.console import Console
from queue import Queue
from tts import TextToSpeechService

try:
    import whisper
    stt = whisper.load_model("base.en")
except Exception as e:
    print(f"Error initializing whisper: {e}")
    raise

console = Console()

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

try:
    tts = TextToSpeechService()
    print(f"Devices: {tts.devices}")
    print(f"Models: {len(tts.models)} models loaded.")
except Exception as e:
    print(f"Error initializing TextToSpeechService: {e}")
    raise

# Set the default output device to "Microphone (Yeti Nano), "Speakers (Realtek(R) Audio)"
sd.default.device = (1, 5)  # (input_device, output_device)

print(sd.query_devices())

def record_audio(stop_event, data_queue):
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)

def transcribe(audio_np: np.ndarray) -> str:
    result = stt.transcribe(audio_np, fp16=True)
    text = result["text"].strip()
    return text

def get_llm_response(text: str) -> str:
    import ollama
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': text},
    ])
    response_text = response['message']['content']
    print(response_text)
    return response_text

def play_audio(sample_rate, audio_array):
    
    print(f"Sample rate: {sample_rate}, Audio array shape: {audio_array.shape}")
    sd.play(audio_array, sample_rate)
    sd.wait()

if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input("Press Enter to start recording, then press Enter again to stop.")
            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status("Generating response...", spinner="earth"):
                    response = get_llm_response(text)
                    sample_rate, audio_array = tts.long_form_synthesize(response)

                console.print(f"[cyan]Assistant: {response}")
                if np.any(audio_array):
                    play_audio(sample_rate, audio_array)
                else:
                    console.print("[red]Generated audio is empty.")
            else:
                console.print("[red]No audio recorded. Please ensure your microphone is working.")

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
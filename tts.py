import torch
import warnings
import numpy as np
import nltk
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoProcessor, BarkModel

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)

class TextToSpeechService:
    def __init__(self, devices=None):
        """
        Initializes the TextToSpeechService class.

        Args:
            devices (list, optional): List of devices to use for the model. Defaults to all available GPUs.
        """
        if devices is None:
            if torch.cuda.is_available():
                self.devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
            else:
                self.devices = ['cpu']
                print("CUDA is not available. Using CPU.")
        else:
            self.devices = devices

        print(f"Devices to be used: {self.devices}")

        self.processors = []
        self.models = []
        for device in self.devices:
            try:
                processor = AutoProcessor.from_pretrained("suno/bark-small")
                model = BarkModel.from_pretrained("suno/bark-small")
                model.to(device)
                self.processors.append(processor)
                self.models.append(model)
                print(f"Loaded model on device: {device}")
            except Exception as e:
                print(f"Error loading model on device {device}: {e}")

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_6", device_index: int = 0):
        """
        Synthesizes audio from the given text using the specified voice preset.

        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_6".
            device_index (int, optional): The index of the device to use.

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        if not self.processors or not self.models:
            raise RuntimeError("Models are not initialized properly. Please check your device settings.")

        processor = self.processors[device_index]
        model = self.models[device_index]
        device = self.devices[device_index]

        inputs = processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_array = model.generate(**inputs, pad_token_id=10000)

        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = model.generation_config.sample_rate
        return sample_rate, audio_array

    def long_form_synthesize(self, text: str, voice_preset: str = "v2/en_speaker_6"):
        """
        Synthesizes audio from the given long-form text using the specified voice preset.

        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_1".

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        if not self.models:
            raise RuntimeError("Models are not initialized properly. Please check your device settings.")

        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.1 * self.models[0].generation_config.sample_rate))  # Reduced silence duration

        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = [executor.submit(self.synthesize, sent, voice_preset, i % len(self.devices)) for i, sent in enumerate(sentences)]
            results = [future.result() for future in futures]

        pieces = []
        for sample_rate, audio_array in results:
            pieces += [audio_array, silence.copy()]

        return sample_rate, np.concatenate(pieces)
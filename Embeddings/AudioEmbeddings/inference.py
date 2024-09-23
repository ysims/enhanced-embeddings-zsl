from models.VGGish import VGGNet
from models.YAMNet import YAMNet
from models.Inception import InceptionV4
import torch
import numpy as np

from dataset.dataset_utils import wavfile_to_examples

# Takes a .wav file and returns the semantic audio vector to be used for that sound
# Sound embeddings of each 1-second clip are created, and these are averaged to create the final embedding (Xie, 2021)
def audio_to_embedding(wav_file, model, device, channels):
    with torch.no_grad():
        if model.name == "Inception":
            bins = 480
        else:
            bins = 64
        # Convert the wav file given to the input type for vggish
        wav_embedding = wavfile_to_examples(wav_file, bins)
        if channels == 1:
            input = np.array([[wav_embedding[0]]])
        elif channels == 3:
            input = np.array([[wav_embedding[0], wav_embedding[0], wav_embedding[0]]])
        input = torch.from_numpy(input).float().to(device)
        embedding = model.inference(input)

        return embedding

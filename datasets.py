import os
import numpy as np
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor
import utils
import torch


class AST_SpeechQualityDataset(Dataset):
    def __init__(self, df, data_dir, dim):
        self.df = df
        self.data_dir = data_dir
        self.dim = dim

        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor.sampling_rate = 48000  # Set to 48 kHz for our 48k audio
        self.feature_extractor.max_length = 1024      # Truncate inputs after 1024 patches
        self.feature_extractor.num_mel_bins = 128     # Customize if needed; keep original as default
        self.feature_extractor.return_attention_mask = True


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        db = self.df['db'].iloc[index]

        # wav files -------------------------------------
        file_name = os.path.join(self.data_dir, self.df['file_path'].iloc[index])

        waveform, sample_rate = utils.process_audio_file(file_name)
        waveform = waveform.squeeze()

        self.feature_extractor.mean = self.df['db_mean'].iloc[index]  #-4.2677393      # Replace if recalculated mean for 48 kHz data
        self.feature_extractor.std = self.df['db_std'].iloc[index]  #4.5689974        # Replace if recalculated std for 48 kHz data

        features = self.feature_extractor(
            waveform, 
            sampling_rate=sample_rate, 
            return_attention_mask=True, 
            return_tensors="pt"
        )['input_values']

        features = features.squeeze()
        
        # target y value ----------------------------------------------------------------------------
        y = self.df[self.dim].iloc[index].astype('float32')
        y = (y - 1) / 4

        return index, features, y
    
class ASTVal(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir
        
        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor.sampling_rate = 48000  # Set to 48 kHz for our 48k audio
        self.feature_extractor.max_length = 1024      # Truncate inputs after 1024 patches
        self.feature_extractor.num_mel_bins = 128     # Customize if needed; keep original as default
        self.feature_extractor.return_attention_mask = True


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        db = self.df['db'].iloc[index]

        # wav files -------------------------------------
        file_name = os.path.join(self.data_dir, self.df['file_path'].iloc[index])

        waveform, sample_rate = utils.process_audio_file(file_name)
        waveform = waveform.squeeze()

        self.feature_extractor.mean = self.df['db_mean'].iloc[index]  #-4.2677393      # Replace if recalculated mean for 48 kHz data
        self.feature_extractor.std = self.df['db_std'].iloc[index]  #4.5689974        # Replace if recalculated std for 48 kHz data

        features = self.feature_extractor(
            waveform, 
            sampling_rate=sample_rate, 
            return_attention_mask=True, 
            return_tensors="pt"
        )['input_values']

        features = features.squeeze()

        return index, features


# region Import.

import numpy as np

from torch.utils.data import Dataset

# endregion


__all__ = ["TrainingDataset"]


class TrainingDataset(Dataset):
    def __init__(self, path_data : str) -> None:
        super().__init__()
        
        data = np.load(path_data)
        
        self.audio_feat = data["audio"].astype(np.float32)  # num_clips X time X dim_feat.
        self.motion_feat = data["motion"].astype(np.float32)  # num_clips X time X dim_feat.
        self.index = data["index"].astype(int)  # num_clips X num_blocks.

        self.motion_latent_code = data["motion_latent_code"]
        self.lexicon_size = data["lexicon_size"]
        self.lexeme_center_sorted = data["lexeme_center_sorted"]
        self.lexemes_sorted = data["lexemes_sorted"]
        self.lexemes_sorted_index = data["lexemes_sorted_index"]

        if "lexeme" in dict(data).keys():
            self.lexeme = data["lexeme"].astype(np.float32)  # num_clips X num_blocks X dim_feat.
        else:
            self.lexeme = np.zeros((self.index.shape[0], self.index.shape[1], self.audio_feat.shape[-1])).astype(np.float32)

        if "lexeme_index" in dict(data).keys():
            self.lexeme_index = data["lexeme_index"].astype(int)  # num_clips X num_blocks.
        else:
            self.lexeme_index = np.zeros((self.index.shape[0], self.index.shape[1])).astype(int)
        
        self.max_index = int(np.max(self.index))
    
    def __len__(self):
        return self.audio_feat.shape[0]
    
    def __getitem__(self, idx):
        return {
            "audio": self.audio_feat[idx, :, :],
            "motion": self.motion_feat[idx, :, :],
            "lexeme": self.lexeme[idx, :, :],
            "lexeme_index": self.lexeme_index[idx, :],
            "index": self.index[idx, :],

            "motion_latent_code": self.motion_latent_code[idx, :, :],
            "lexicon_size": self.lexicon_size,
            "lexeme_center_sorted": self.lexeme_center_sorted,
            "lexemes_sorted": self.lexemes_sorted,
            "lexemes_sorted_index": self.lexemes_sorted_index
        }
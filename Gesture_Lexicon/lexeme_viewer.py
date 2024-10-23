# region Import.

import os
import sys
import argparse
import torch
import torch.nn as nn

import numpy as np

# from Gesture_Generator.network import GRUDecoder_forlxm, CellStateInitializer_forlxm
from Gesture_Lexicon.model import Transformer
from Gesture_Lexicon.model import VQVAE_gcn
from Gesture_Lexicon.model import VQVAE_conv

module_path = os.path.dirname(os.path.abspath(__file__))
if module_path not in sys.path:
    sys.path.append(module_path)

from sampling import Inference

# endregion


__all__ = ["show_lexeme"]


def get_args_parser():
    parser = argparse.ArgumentParser('gesture lexicon', add_help=False)

    parser.add_argument('--data_dir', type=str, default='../data/Trinity/Processed/Training_Data')
    parser.add_argument('--motiondecoder', type=str, default='../Gesture_Generator/Trinity/Processed/Training_Data')
    parser.add_argument('--checkpoint_config', type=str, default='./Training/Trinity/_RNN_20231117_101946/config.json5')
    parser.add_argument('--save', action='store_true', default=True)

    return parser


class show_lexeme(nn.Module):

    def __init__(self):
        super().__init__()

        self.motion_decoder_vqvae = VQVAE_gcn(mo_dim=48, feature_size=96, lexicon_size=50, rotation_size=3, joint_size=16, beta=0.25)
        # self.motion_decoder_vqvae = VQVAE_conv(n_hiddens=128, n_residual_hiddens=32, n_residual_layers=2,
        #          n_embeddings=50, embedding_dim=96, beta=0.25, save_img_embedding_map=False)

    def forward(self, lexemes):

        # lexemes = lexemes.squeeze(0)
        #
        # # lexeme motion
        # D, B = lexemes.shape
        # lexemes = lexemes.reshape(B, D)

        # VQVAE_conv
        N, D, B = lexemes.shape
        lexemes = lexemes.reshape(B, D, N)

        _, x_hat, _ = self.motion_decoder_vqvae.forward_forlxm(lexemes)

        L, D, B = x_hat.shape
        x_hat = x_hat.reshape(1, D, L * B)

        sliiced_x_hats = x_hat[:, :, 10:(x_hat.shape[2] - 10)]

        if lexemes.shape[0] == 1:
            return x_hat  # lexeme motion
        else:
            return sliiced_x_hats  # whole motion


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    with open(os.path.join(args.data_dir, "valid.npz"), "rb") as f:
        data = np.load(f)
        lexeme = data['lexeme']
        lexeme = torch.from_numpy(lexeme)  # (174, 10, 96)
        lexeme = lexeme.permute(0, 2, 1)

    check_lexemes = show_lexeme()

    lexeme_motions = check_lexemes(args.data_dir)

    print('done')
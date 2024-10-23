# region Import.

import os
import sys
import pickle
import argparse
import json5

import numpy as np
import joblib as jl
import torch

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

module_path = os.path.dirname(os.path.abspath(__file__))
if module_path not in sys.path:
    sys.path.append(module_path)

from sampling import Inference

# endregion


__all__ = ["build_lexicon", "predict_lexeme"]


def get_args_parser():
    parser = argparse.ArgumentParser('gesture lexicon', add_help=False)

    parser.add_argument('--data_dir', type=str, default='../data/Trinity/Processed/Training_Data')
    parser.add_argument('--checkpoint_path', type=str, default='./Training/Trinity/_RVQ_conv_qqloss2/Checkpoints/trained_model.pth')
    parser.add_argument('--checkpoint_config', type=str, default='./Training/Trinity/_RVQ_conv_qqloss2/config.json5')
    parser.add_argument('--lexicon_size', type=int, default=1000) # in paper, 50 (Trinity)
    parser.add_argument('--num_kmeans_rerun', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save', action='store_true', default=True)

    return parser


def build_lexicon(path_dataset: str, path_pretrained_net: str, path_config_train: str,
                  lexicon_size: int, num_rerun: int = 10, device: str = "cuda:0", save: bool = True):
    print('build lexicon...')

    # data_dir = os.path.dirname(path_dataset)

    inference = Inference(path_dataset, path_pretrained_net, device, path_config_train)

    lexemes, lexemes_indices = inference.infer()  # num_clips X num_blocks X dim_feat.

    # # @ TSNE Visualize
    #
    # outputs_lexeme = np.array(lexemes)
    #
    # outputs_lexeme = outputs_lexeme.reshape(lexemes.shape[0] * lexemes.shape[2], -1)
    #
    # outputs_indices = np.array(lexemes_indices)
    # outputs_indices = outputs_indices.squeeze(1)
    # unique_elements, counts = np.unique(outputs_indices, return_counts=True, axis=0)
    #
    # colors = np.random.rand(len(unique_elements), 3)
    # color_list = []
    #
    # for idx, output_index in enumerate(outputs_indices):
    #     for i, unique_index in enumerate(unique_elements):
    #         if np.all(output_index == unique_index):
    #             color_list.append(i)
    #             break
    #
    # tsne_model = TSNE(n_components=2)
    # outputs = tsne_model.fit_transform(outputs_lexeme)
    # plt.scatter(x=outputs[:, 0],
    #             y=outputs[:, 1],
    #             c=colors[color_list])
    # plt.axis([-100, 100, -100, 100])
    # plt.savefig('tsne_fig/train_tsne.png')
    #
    # zipped_list = zip(color_list, outputs[:, 0], outputs[:, 1])
    # zipped_list = sorted(zipped_list, key=lambda x: x[0])
    # plt.cla()
    #
    # # lexeme 별
    # for i in range(1, len(color_list)):
    #     if np.all(colors[zipped_list[i][0]] != colors[zipped_list[i - 1][0]]):
    #         plt.axis([-100, 100, -100, 100])
    #         plt.savefig(f'tsne_fig/train/{zipped_list[i][0]}.png')
    #         plt.cla()
    #
    #     plt.scatter(x=zipped_list[i][1],
    #                 y=zipped_list[i][2],
    #                 c=colors[zipped_list[i][0]])
    #
    # #

    # lexemes = lexemes.reshape(lexemes.shape[1]//10, 10, lexemes.shape[2])
    #
    # lexemes_indices = lexemes_indices.reshape(lexemes_indices.shape[1]//10, 10)


    if save:
        print('save...')
        data = dict(np.load(path_dataset, allow_pickle=True))
        data["lexeme"] = lexemes
        data["lexeme_index"] = lexemes_indices
        data["motion_latent_code"] = lexemes
        data["lexeme_center_sorted"] = lexemes
        np.savez(path_dataset, **data)

    return lexemes


def predict_lexeme(path_dataset: str, path_pretrained_net: str, path_config_train: str, device: str = "cuda:0", save: bool = True, isinference:bool = False):
    print('save lexemes...')

    inference = Inference(path_dataset, path_pretrained_net, device, path_config_train, isinference)

    lexemes, lexemes_indices = inference.infer()  # num_clips X num_blocks X dim_feat.


    # # @ TSNE Visualize
    #
    # outputs_lexeme = np.array(lexemes)
    # outputs_lexeme = outputs_lexeme.reshape(lexemes.shape[0] * lexemes.shape[2], -1)
    #
    # outputs_indices = np.array(lexemes_indices)
    # outputs_indices = outputs_indices.squeeze(1)
    # unique_elements, counts = np.unique(outputs_indices, return_counts=True, axis=0)
    #
    # colors = np.random.rand(len(unique_elements), 3)
    # color_list = []
    #
    # for idx, output_index in enumerate(outputs_indices):
    #     for i, unique_index in enumerate(unique_elements):
    #         if np.all(output_index == unique_index):
    #             color_list.append(i)
    #             break
    #
    # tsne_model = TSNE(n_components=2)
    # outputs = tsne_model.fit_transform(outputs_lexeme)
    # plt.scatter(x=outputs[:, 0],
    #             y=outputs[:, 1],
    #             c=colors[color_list])
    # plt.axis([-100, 100, -100, 100])
    # plt.savefig('tsne_fig/train_tsne.png')
    #
    # zipped_list = zip(color_list, outputs[:, 0], outputs[:, 1])
    # zipped_list = sorted(zipped_list, key=lambda x: x[0])
    # plt.cla()
    #
    # # lexeme 별
    # for i in range(1, len(color_list)):
    #     if np.all(colors[zipped_list[i][0]] != colors[zipped_list[i - 1][0]]):
    #         plt.axis([-100, 100, -100, 100])
    #         plt.savefig(f'tsne_fig/train/{zipped_list[i][0]}.png')
    #         plt.cla()
    #
    #     plt.scatter(x=zipped_list[i][1],
    #                 y=zipped_list[i][2],
    #                 c=colors[zipped_list[i][0]])
    #
    # #


    # lexemes = lexemes.reshape(lexemes.shape[1]//10, 10, lexemes.shape[2])
    # lexemes_indices = lexemes_indices.reshape(lexemes_indices.shape[1]//10, 10)

    # lexemes_indices = lexemes_indices.reshape(-1)

    if save:
        print('save...')
        data = dict(np.load(path_dataset, allow_pickle=True))  # if inferecne save at 'Test_Data/valid.npz'
        data["lexeme"] = lexemes
        data["lexeme_index"] = lexemes_indices
        data["motion_latent_code"] = lexemes
        data["lexeme_center_sorted"] = lexemes
        np.savez(path_dataset, **data)


    return lexemes




if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    train_data_path = os.path.join(args.data_dir, 'train.npz')
    valid_data_path = os.path.join(args.data_dir, 'valid.npz')
    lxm_scaler_path = os.path.join(args.data_dir, 'train_lexeme_scaler.sav')

    with open(os.path.join(args.data_dir, 'config.json5'), 'r') as f:
        dataset_config = json5.load(f)
    dataset_config['lexicon_size'] = args.lexicon_size
    with open(os.path.join(args.data_dir, 'config.json5'), 'w') as f:
        json5.dump(dataset_config, f, indent=4)

    _ = build_lexicon(train_data_path, args.checkpoint_path, args.checkpoint_config,
                               args.lexicon_size, args.num_kmeans_rerun, args.device, args.save)

    lexemes = predict_lexeme(valid_data_path, args.checkpoint_path, args.checkpoint_config, args.device, args.save)

    print('done.')


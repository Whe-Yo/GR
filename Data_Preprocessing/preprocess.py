# region Import.

import os
import sys
import json5
import shutil
import time
import pandas as pd

module_path = os.path.dirname(os.path.abspath(__file__))
if module_path not in sys.path:
    sys.path.append(module_path)

from Utils.audio_features import *
from Utils.motion_features import *
from Utils.beat_features import *
from Utils.utils import *


# endregion


class Preprocessor:
    def __init__(self, path_config: str, isInference: False, random_cutsample_sec: None) -> None:
        with open(path_config, "r") as f:
            self.config = json5.load(f)

            self.sleeptime = 0.1

            self.uniform_len = int(self.config["fps"] * self.config["onset_bounds"][1])

            # Trinity2 Dataset의 beat start&end time을 사용하고 싶다면
            if isInference:
                self.istrinity2 = False
            else:
                self.istrinity2 = self.config["usingTrinity2Text"]

            if random_cutsample_sec is not None:
                self.random_cutsample_sec = random_cutsample_sec
            else:
                self.random_cutsample_sec = None





    def preprocess(self) -> None:
        # region Configs.

        dir_data = self.config["dir_data"]
        dir_motion = os.path.join(dir_data, "Motion")
        dir_audio = os.path.join(dir_data, "Audio")
        if self.istrinity2:
            dir_aligntimes = os.path.join(dir_data, "Trinity2/") + 'AlignmentTimes.csv'
            alignmenttimes = pd.read_csv(dir_aligntimes).values
            audiostarttimes = alignmenttimes[:, 2]
            motionstarttimes = alignmenttimes[:, 3]

        dir_save = self.config["dir_save"]
        dir_features = os.path.join(dir_save, "Features")
        os.makedirs(dir_save, exist_ok=True)
        os.makedirs(dir_features, exist_ok=True)

        name_files = []
        # r = root, d = directories, f = files.
        for r, d, f in os.walk(dir_motion):
            for file in sorted(f):
                if '.bvh' in file:
                    ff = os.path.join(r, file)
                    basename = os.path.splitext(os.path.basename(ff))[0]
                    name_files.append(basename)
        name_files_train = self.config["name_files_train"]
        name_files_valid = self.config["name_files_valid"]

        fps = int(self.config["fps"])
        sr = int(self.config["sr"])

        dim_mel = int(self.config["dim_mel"])

        joints_selected = self.config["joints_selected"]

        # # Trinity 2 beat 읽기

        if self.istrinity2:
            dir_text = os.path.join(dir_data, "Text")

        #

        # path_pretrained_motif_net = self.config["path_pretrained_motif_net"]
        # path_config_motif_train = self.config["path_config_motif_train"]
        # device = self.config["device"]

        with open(os.path.join(dir_save, "config.json5"), "w") as f:
            json5.dump(self.config, f, indent=4)

        # endregion

        # region Load audio. save whole raw audio

        print("\nLoading audio......")

        dir_wav = os.path.join(dir_features, "WAV_Audio")
        if os.path.exists(dir_wav):
            print("Found audio data, skip.")
        else:
            os.makedirs(dir_wav)
            time.sleep(self.sleeptime)
            if self.istrinity2:
                _ = load_audio_fromtrinity(dir_audio=dir_audio, names_file=name_files, dir_save=dir_wav, sr=sr, save=True, audiostarttimes=audiostarttimes)
            else:
                _ = load_audio(dir_audio=dir_audio, names_file=name_files, dir_save=dir_wav, sr=sr, save=True)

        # endregion

        # region Extract mel-spectrum.

        print("\nExtracting mel-spectrum......")

        dir_mel = os.path.join(dir_features, "Mel")
        if os.path.exists(dir_mel):
            print("Found mel-spectrum, skip.")
        else:
            os.makedirs(dir_mel)
            time.sleep(self.sleeptime)
            _ = extract_melspec(dir_loaded_audio=dir_wav, names_file=name_files, dir_save=dir_mel,
                                fps=fps, sr=sr, dim_mel=dim_mel, mel_filter_len=self.config["mel_filter_len"], mel_hop_len=self.config["mel_hop_len"],
                                save=True)

        # endregion

        # region Detect audio onset.

        print("\nDetecting audio onset......")

        dir_onset = os.path.join(dir_features, "Onset")
        if os.path.exists(dir_onset):
            print("Found audio onset, skip.")
        else:
            os.makedirs(dir_onset)
            time.sleep(self.sleeptime)

            if self.istrinity2:
                _ = detect_onset_fromtrinity(dir_loaded_audio=dir_wav, names_file=name_files, dir_save=dir_onset, dir_text=dir_text,
                                 fps=fps, sr=sr, bounds=self.config["onset_bounds"], save=True)
            else: # onset 탐지 및 새로운 onset 삽입
                _ = detect_onset(dir_loaded_audio=dir_wav, dir_loaded_mel=dir_mel, names_file=name_files, dir_save=dir_onset,
                                 fps=fps, sr=sr, bounds=self.config["onset_bounds"], save=True)

        # endregion

        # region Transfer bvh motion to facing coordinate.

        print("\nTransferring bvh motion to facing coordinate......")

        dir_motion_faced = os.path.join(dir_features, "BVH_Motion_Faced")
        if os.path.exists(dir_motion_faced):
            print("Found bvh motion faced, skip.")
        else:
            os.makedirs(dir_motion_faced)  # new normalized & centered bvh motion
            time.sleep(self.sleeptime)
            if self.istrinity2:
                transfer_to_facing_coordinate_fromtrinity(dir_motion=dir_motion, names_file=name_files, dir_save=dir_motion_faced,
                                              rotation_order=self.config["bvh_rotation_order"], motionstarttimes=motionstarttimes)
            else:
                transfer_to_facing_coordinate(dir_motion=dir_motion, names_file=name_files, dir_save=dir_motion_faced,
                                              rotation_order=self.config["bvh_rotation_order"])

        # endregion

        # region Make motion template.

        print("\nMaking motion template......")
        time.sleep(self.sleeptime)
        make_motion_template(dir_motion=dir_motion, names_file=name_files, dir_save=dir_save,
                             fps=fps, rotation_order=self.config["bvh_rotation_order"], motion_duration=300)

        # endregion

        # region Extract bvh motion.

        print("\nExtracting bvh motion......")

        dir_bvh_motion = os.path.join(dir_features, "BVH_Motion")
        if os.path.exists(dir_bvh_motion):
            print("Found bvh motion, skip.")
        else:
            os.makedirs(dir_bvh_motion)
            time.sleep(self.sleeptime)
            _ = extract_bvh_motion(dir_motion=dir_motion_faced, names_file=name_files, dir_save=dir_bvh_motion,
                                   fps=fps, joints_selected=joints_selected, save=True)
            shutil.copy(os.path.join(dir_bvh_motion, 'data_pipe.sav'), os.path.join(dir_save, 'data_pipe.sav'))

        # endregion

        # region Align mel with motion.

        print("\nAligning mel with motion......")

        dir_mel_motion_aligned = os.path.join(dir_features, "Mel_Motion_Aligned")
        if os.path.exists(dir_mel_motion_aligned):
            print("Found mel and motion align result, skip.")
        else:
            os.makedirs(dir_mel_motion_aligned)
            time.sleep(self.sleeptime)
            if self.random_cutsample_sec is not None:
                _, _ = align_mel_with_motion_randomcutsample(dir_mel=dir_mel, dir_motion=dir_bvh_motion, names_file=name_files,
                                             dir_save=dir_mel_motion_aligned, save=True, fps=fps, random_cutsample_sec=self.random_cutsample_sec)
            else:
                _, _ = align_mel_with_motion(dir_mel=dir_mel, dir_motion=dir_bvh_motion, names_file=name_files,
                                         dir_save=dir_mel_motion_aligned, save=True)

        # endregion

        # region Uniform data fragment length.

        print("\nUniforming data fragment length......")

        dir_data_len_uniform = os.path.join(dir_features, "Data_Len_Uniform")
        if os.path.exists(dir_data_len_uniform):
            print("Found data len uniform align result, skip.")
        else:
            os.makedirs(dir_data_len_uniform)
            time.sleep(self.sleeptime)
            if self.istrinity2:
                _, _, _ = uniform_data_fragment_length_tsm_fromtrinity(dir_mel_motion_aligned=dir_mel_motion_aligned,
                                                           dir_onset=dir_onset, dir_wav=dir_wav, names_file=name_files,
                                                           dir_save=dir_data_len_uniform, sr=sr, fps=fps,
                                                           rotation_order=self.config["bvh_rotation_order"],
                                                           uniform_len=self.uniform_len, save=True)
            else:
                _, _, _ = uniform_data_fragment_length_tsm(dir_mel_motion_aligned=dir_mel_motion_aligned,
                                                           dir_onset=dir_onset, dir_wav=dir_wav, names_file=name_files,
                                                           dir_save=dir_data_len_uniform, sr=sr, fps=fps,
                                                           rotation_order=self.config["bvh_rotation_order"],
                                                           uniform_len=self.uniform_len, save=True)

        # endregion

        # region Prepare motion feature(Euler to exp map).

        print("\nPreparing motion feature......")

        dir_motion_expmap = os.path.join(dir_features, "Motion_Expmap")
        if os.path.exists(dir_motion_expmap):
            print("Found motion feature prepare result, skip.")
        else:
            os.makedirs(dir_motion_expmap)
            time.sleep(self.sleeptime)
            _ = prepare_motion_feature(dir_data_len_uniform=dir_data_len_uniform, names_file=name_files,
                                       dir_save=dir_motion_expmap, root_joint=self.config["bvh_root_joint"],
                                       rotation_order=self.config["bvh_rotation_order"], save=True)

        # endregion

        # region Prepare audio feature(Add positional encoding).

        print("\nPreparing audio feature......")

        dir_audio_feat = os.path.join(dir_features, "Audio_Feature")
        if os.path.exists(dir_audio_feat):
            print("Found audio feature result, skip.")
        else:
            os.makedirs(dir_audio_feat)
            time.sleep(self.sleeptime)
            _ = prepare_audio_feature(dir_data_len_uniform=dir_data_len_uniform, names_file=name_files, dir_save=dir_audio_feat,
                                      uniform_len=self.uniform_len, dim_pos_enc=self.config["dim_pos_enc"],
                                      use_pos_enc=self.config['use_pos_enc'],
                                      save=True)


        # endregion

        # region Split dataset.

        if self.config["split_data"]:
            print("\nSpliting dataset......")
            time.sleep(self.sleeptime)

            # Spliting training dataset.
            _, _, _ = split_dataset(dir_audio_feat=dir_audio_feat, dir_motion_expmap=dir_motion_expmap,
                                    dir_data_len_uniform=dir_data_len_uniform,
                                    names_file=name_files_train, dir_save=dir_save,
                                    uniform_len=self.uniform_len,
                                    num_blocks_per_clip=self.config["num_blocks_per_clip"],
                                    step=self.config["step"], dataset_type="train",
                                    save=True)

            # Spliting validation dataset.
            _, _, _ = split_dataset(dir_audio_feat=dir_audio_feat, dir_motion_expmap=dir_motion_expmap,
                                    dir_data_len_uniform=dir_data_len_uniform,
                                    names_file=name_files_valid, dir_save=dir_save,
                                    uniform_len=self.uniform_len,
                                    num_blocks_per_clip=self.config["num_blocks_per_clip"],
                                    step=self.config["step"], dataset_type="valid",
                                    train_scaler_dir=dir_save,
                                    save=True)

        # endregion


# region Test.

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # raise ValueError('Must give config file path argument!')
        path_config = "Config/Trinity/config.json5"
        path_config = "Config/BEAT/config.json5"

    else:
        path_config = sys.argv[-1]

    preprocessor = Preprocessor(path_config, isInference=False, random_cutsample_sec=None)
    preprocessor.preprocess()

# endregion
# region Import.

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import time
import torch
import json5

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_sampler import *
from model import *


# endregion


class Trainer:
    def __init__(self, config, log_tag) -> None:
        # region Config.

        self.config = config

        # self.use_checkpoint = True
        self.use_checkpoint = False
        loaded_log_dir = '_VQVAE_gcn_20240214_182125'

        # endregion

        # region Load data preprocessing config.

        with open(os.path.join(self.config["dir_data"], 'config.json5'), 'r') as f:
            self.config_data_preprocessing = json5.load(f)
        self.fps = self.config_data_preprocessing['fps']

        # endregion

        # region Log.

        self.log_frequency = 10

        date_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        if self.use_checkpoint == False:
            self.log_dir = os.path.join(self.config["dir_log"],
                                        log_tag + '_' + self.config['network']['name'] + '_' + date_time)
            os.makedirs(self.log_dir, exist_ok=True)
            tensorboard_log_dir = os.path.join(self.config["dir_log"],
                                               'Tensorboard_Log', log_tag + '_' + self.config['network']['name'] + '_' + date_time)
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            self.writer = SummaryWriter(logdir=tensorboard_log_dir)
            os.makedirs(os.path.join(self.log_dir, 'Checkpoints'), exist_ok=True)
        else:
            self.log_dir = os.path.join(self.config["dir_log"], log_tag + loaded_log_dir)
            tensorboard_log_dir = os.path.join(self.config["dir_log"],
                                               'Tensorboard_Log', log_tag + loaded_log_dir)
            self.writer = SummaryWriter(logdir=tensorboard_log_dir)



        # endregion

        # region Copy config file to log dir.

        with open(os.path.join(self.log_dir, 'config.json5'), 'w') as f:
            json5.dump(self.config, f, indent=4)

        # endregion

        # region Device.

        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')

        # endregion

        # region Data loader.

        self.dataset_train = TrainingDataset(os.path.join(self.config["dir_data"], "train.npz"))
        # self.data_loader_train = DataLoader(self.dataset_train, batch_size=self.config['batch_size'], shuffle=True)
        self.data_loader_train = DataLoader(self.dataset_train, batch_size=self.config['batch_size'], shuffle=True)

        self.dataset_val = TrainingDataset(os.path.join(self.config["dir_data"], "valid.npz"))
        # self.data_loader_val = DataLoader(self.dataset_val, batch_size=len(self.dataset_val), shuffle=False)
        self.data_loader_val = DataLoader(self.dataset_val, batch_size=len(self.dataset_val), shuffle=False)

        # endregion

        cudnn.benchmark = True

        # region Network.

        if self.config['network']['name'] == 'Conv1d':
            self.net = Conv1d(self.config['network']['encoder_config'],
                              self.config['network']['decoder_config'])
        elif self.config['network']['name'] == 'Transformer':
            self.net = Transformer(**self.config['network']['hparams'])
        elif self.config['network']['name'] == 'VQVAE_gcn':
            self.net = VQVAE_gcn(**self.config['network']['hparams'])
        elif self.config['network']['name'] == 'VQVAE_conv':
            self.net = VQVAE_conv(**self.config['network']['hparams'])
        elif self.config['network']['name'] == 'autoencoder_conv':
            self.net = autoencoder_conv(**self.config['network']['hparams'])
        elif self.config['network']['name'] == 'VAE_conv':
            self.net = VAE_conv(**self.config['network']['hparams'])
        elif self.config['network']['name'] == 'RVQ_gcn':
            self.net = RVQ_gcn(**self.config['network']['hparams'])
        elif self.config['network']['name'] == 'RVQ_conv':
            self.net = RVQ_conv(**self.config['network']['hparams'])
        else:
            raise NotImplementedError

        self.net.to(self.device)

        # endregion

        # region Optimizer.

        if self.config['optimizer']['name'] == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.net.parameters(),
                                              lr=self.config['optimizer']['lr'],
                                              betas=self.config['optimizer']['betas'],
                                              eps=self.config['optimizer']['eps'],
                                              weight_decay=self.config['optimizer']['weight_decay'])
        elif self.config['optimizer']['name'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(params=self.net.parameters(),
                                               lr=self.config['optimizer']['lr'],
                                               betas=self.config['optimizer']['betas'],
                                               eps=self.config['optimizer']['eps'],
                                               weight_decay=self.config['optimizer']['weight_decay'])
        elif self.config['optimizer']['name'] == 'SGD':
            self.optimizer = torch.optim.SGD(params=self.net.parameters(),
                                             lr=self.config['optimizer']['lr'],
                                             momentum=0.1,
                                             dampening=0,
                                             weight_decay=0)
        else:
            raise NotImplementedError

        # endregion

        # region Criterion.

        # F.mse_loss = torch.nn.functional.mse_loss()
        # F.mse_loss = torch.nn.L1Loss()

        # endregion

    def train(self) -> None:
        # region Start timing.

        since = time.time()

        # endregion

        # region Epoch loop.
        num_epoch = self.config["num_epoch"]

        start_epoch = 0
        if self.use_checkpoint:
            checkpoint = torch.load(os.path.join(self.log_dir, 'Checkpoints', 'trained_model.pth'))

            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']

        for epoch in range(start_epoch, num_epoch + 1):
            print(f'\nEpoch: {epoch}/{num_epoch}')

            self.net.train()

            loss_train, loss_rot_train, loss_vel_train, loss_acc_train = 0., 0., 0., 0.
            loss_commit_train, loss_contrast_train, loss_activity_train = 0., 0., 0.
            counter = 0

            # region Data loader loop.

            pbar = tqdm(total=len(self.dataset_train), ascii=True)
            for _, batch in enumerate(self.data_loader_train):
                # region Prepare data.

                motion_block = batch["motion"].to(self.device)  # batch_size X dim_feat X time.

                # endregion

                # region Forward.

                self.optimizer.zero_grad()


                if self.config['network']['name'] in ['Conv1d', 'Transformer', 'VQVAE_gcn', 'VQVAE_conv', 'RVQ_gcn', 'RVQ_conv', 'autoencoder_conv', 'VAE_conv']:
                    quantized, motion_block_hat, _, loss_commit, loss_contrast, loss_activity = self.net(motion_block)
                else:
                    raise NotImplementedError

                # endregion

                # region Loss and net weights update.

                loss_rot = F.mse_loss(motion_block, motion_block_hat)
                # mse_loss = nn.MSELoss(reduction='mean')
                # loss_rot = mse_loss(motion_block, motion_block_hat)
                loss_vel = F.mse_loss(motion_block[:, :, 1:] - motion_block[:, :, :-1],
                                          motion_block_hat[:, :, 1:] - motion_block_hat[:, :, :-1])
                loss_acc = F.mse_loss(
                    motion_block[:, :, 2:] + motion_block[:, :, :-2] - 2 * motion_block[:, :, 1:-1],
                    motion_block_hat[:, :, 2:] + motion_block_hat[:, :, :-2] - 2 * motion_block_hat[:, :, 1:-1])
                loss_commit = loss_commit.to(self.device)
                loss_contrast = loss_contrast.to(self.device)
                loss_activity = loss_activity.to(self.device)
                loss = self.config["loss"]["rot"] * loss_rot + self.config["loss"]["vel"] * loss_vel + self.config["loss"]["acc"] * loss_acc + \
                       self.config["loss"]["com"] * loss_commit + self.config["loss"]["ctr"] * loss_contrast + self.config["loss"]["act"] * loss_activity
                # loss = self.config["loss"]["rot"] * loss_rot + self.config["loss"]["com"] * loss_commit

                loss.backward()
                # loss_rot.backward()
                self.optimizer.step()

                loss_rot_train += loss_rot.item() * motion_block.shape[0] * self.config["loss"]["rot"]
                loss_vel_train += loss_vel.item() * motion_block.shape[0] * self.config["loss"]["vel"]
                loss_acc_train += loss_acc.item() * motion_block.shape[0] * self.config["loss"]["acc"]
                loss_commit_train += loss_commit.item() * motion_block.shape[0] * self.config["loss"]["com"]
                loss_contrast_train += loss_contrast.item() * motion_block.shape[0] * self.config["loss"]["ctr"]
                loss_activity_train += loss_activity.item() * motion_block.shape[0] * self.config["loss"]["act"]
                loss_train += loss.item() * motion_block.shape[0]
                counter += motion_block.shape[0]

                # endregion

                # region Pbar update.

                pbar.set_description('lr: %s' % (str(self.optimizer.param_groups[0]['lr'])))
                pbar.update(motion_block.shape[0])

                # endregion

            pbar.close()

            # endregion

            # region Epoch loss and log.

            loss_train /= counter
            loss_rot_train /= counter
            loss_vel_train /= counter
            loss_acc_train /= counter
            loss_commit_train /= counter
            loss_contrast_train /= counter
            loss_activity_train /= counter

            print('Training',
                  f'Loss: {loss_train:.5f}',
                  f'Rot Loss: {loss_rot_train:.4f} /',
                  f'Vel Loss: {loss_vel_train:.4f} /',
                  f'Acc Loss: {loss_acc_train:.4f} /',
                  f'Commit Loss: {loss_commit_train:.4f} /',
                  f'Contrast Loss: {loss_contrast_train:.4f} /',
                  f'Activity Loss: {loss_activity_train:.4f} /',
                  )


            if epoch % self.log_frequency == 0:
                self.writer.add_scalar(tag="Train/Loss", scalar_value=loss_train, global_step=epoch)
                self.writer.add_scalar("Rot_Loss/Train", loss_rot_train, epoch)
                self.writer.add_scalar("Vel_Loss/Train", loss_vel_train, epoch)
                self.writer.add_scalar("Acc_Loss/Train", loss_acc_train, epoch)
                self.writer.add_scalar("Commit_Loss/Train", loss_commit_train, epoch)
                self.writer.add_scalar("Contrast_Loss/Train", loss_contrast_train, epoch)
                self.writer.add_scalar("Activity_Loss/Train", loss_activity_train, epoch)

            # endregion

            # region Checkpoints.

            if epoch % self.config['checkpoint_save_epoch_num'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, os.path.join(self.log_dir, 'Checkpoints', f'checkpoint_{epoch // 1000}k{epoch % 1000}.pth'))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, os.path.join(self.log_dir, 'Checkpoints', 'trained_model.pth'))

            # endregion

            # region Validation.

            valid_num_epoch = self.config['valid_num_epoch']
            if epoch % valid_num_epoch == 0:
                self.net.eval()

                loss_valid, loss_rot_valid, loss_vel_valid, loss_acc_valid = 0., 0., 0., 0.
                loss_commit_valid, loss_contrast_valid, loss_activity_valid = 0., 0., 0.
                counter = 0

                with torch.no_grad():
                    for _, batch in enumerate(self.data_loader_val):
                        motion_block = batch["motion"].to(self.device)  # batch_size X dim_feat X time.

                        if self.config['network']['name'] in ['Conv1d', 'Transformer', 'VQVAE_gcn', 'VQVAE_conv', 'RVQ_gcn', 'RVQ_conv', 'autoencoder_conv', 'VAE_conv']:
                            _, motion_block_hat, _, loss_commit, loss_contrast, loss_activity = self.net(motion_block)
                        else:
                            raise NotImplementedError

                        loss_rot = F.mse_loss(motion_block, motion_block_hat)
                        loss_vel = F.mse_loss(motion_block[:, :, 1:] - motion_block[:, :, :-1],
                                                  motion_block_hat[:, :, 1:] - motion_block_hat[:, :, :-1])
                        loss_acc = F.mse_loss(
                            motion_block[:, :, 2:] + motion_block[:, :, :-2] - 2 * motion_block[:, :, 1:-1],
                            motion_block_hat[:, :, 2:] + motion_block_hat[:, :, :-2] - 2 * motion_block_hat[:, :, 1:-1])
                        loss_commit = loss_commit.to(self.device)
                        loss_contrast = loss_contrast.to(self.device)
                        loss_activity = loss_activity.to(self.device)
                        loss = self.config["loss"]["rot"] * loss_rot + self.config["loss"]["vel"] * loss_vel + self.config["loss"]["acc"] * loss_acc + \
                               self.config["loss"]["com"] * loss_commit + self.config["loss"]["ctr"] * loss_contrast + self.config["loss"]["act"] * loss_activity
                        # loss = self.config["loss"]["rot"] * loss_rot + self.config["loss"]["com"] * loss_commit

                        loss_rot_valid += loss_rot.item() * motion_block.shape[0] * self.config["loss"]["rot"]
                        loss_vel_valid += loss_vel.item() * motion_block.shape[0] * self.config["loss"]["vel"]
                        loss_acc_valid += loss_acc.item() * motion_block.shape[0] * self.config["loss"]["acc"]
                        loss_commit_valid += loss_commit.item() * motion_block.shape[0] * self.config["loss"]["com"]
                        loss_contrast_valid += loss_contrast.item() * motion_block.shape[0] * self.config["loss"]["ctr"]
                        loss_activity_valid += loss_activity.item() * motion_block.shape[0] * self.config["loss"]["act"]
                        loss_valid += loss.item() * motion_block.shape[0]
                        counter += motion_block.shape[0]

                    loss_valid /= counter
                    loss_rot_valid /= counter
                    loss_vel_valid /= counter
                    loss_acc_valid /= counter
                    loss_commit_valid /= counter
                    loss_contrast_valid /= counter
                    loss_activity_valid /= counter

                    self.writer.add_scalar(tag="Valid/Loss", scalar_value=loss_valid, global_step=epoch)
                    self.writer.add_scalar("Rot_Loss/Valid", loss_rot_valid, epoch)
                    self.writer.add_scalar("Vel_Loss/Valid", loss_vel_valid, epoch)
                    self.writer.add_scalar("Acc_Loss/Valid", loss_acc_valid, epoch)
                    self.writer.add_scalar("Commit_Loss/Valid", loss_commit_valid, epoch)
                    self.writer.add_scalar("Contrast_Loss/Valid", loss_contrast_valid, epoch)
                    self.writer.add_scalar("Activity_Loss/Valid", loss_activity_valid, epoch)

            # endregion

        # endregion

        # region Save network.

        # torch.save(self.net.state_dict(), os.path.join(self.log_dir, 'Checkpoints', 'trained_model.pth'))

        # endregion

        # region End timing.

        time_elapsed = time.time() - since
        print('\nTraining completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # endregion

        self.writer.close()


if __name__ == '__main__':

    # @ select model and config

    if len(sys.argv) < 2:
        # raise ValueError('Must give config file path argument!')
        # path_config = "Config/Trinity/config_conv1d.json5"
        # path_config = "Config/Trinity/config_transformer.json5"
        # path_config = "Config/Trinity/config_vqvae_gcn.json5"
        # path_config = "Config/Trinity/config_vqvae_conv.json5"

        # path_config = "Config/Trinity/config_autoencoder_conv.json5"
        # path_config = "Config/Trinity/config_vae_conv.json5"
        # path_config = "Config/Trinity/config_rvq_gcn.json5"
        path_config = "Config/Trinity/config_rvq_conv.json5"
        with open(path_config, 'r') as f:
            config = json5.load(f)
    else:
        with open(sys.argv[1], 'r') as f:
            config = json5.load(f)

    # @

    # for log
    log_tag = sys.argv[-1] if len(sys.argv) == 3 else ''

    # start train
    trainer = Trainer(config, log_tag)
    trainer.train()
# region Import.

import os
import sys
import time
import torch
import json5
import math

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np

from scipy.linalg import sqrtm

from dataset import *
from utils import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# endregion

def calculate_fgd(mu1, sigma1, mu2, sigma2):
    """Frechet Geometry Distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

    # Handle imaginary component from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fgd = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fgd

def compute_statistics(data):
    """Compute the mean and covariance matrix of the given data."""
    # torch.Tensor일 경우 numpy 배열로 변환
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.cpu()  # cuda 장치에 있는 텐서를 CPU 메모리로 복사
        data = data.detach().numpy()  # detach()를 사용하여 grad 정보 제거 후 numpy로 변환
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=False)
    return mu, sigma


class Trainer:
    def __init__(self, config, log_tag) -> None:
        # region Config.

        self.config = config

        # endregion

        # region Load data preprocessing config.

        with open(os.path.join(self.config["dir_data"], 'config.json5'), 'r') as f:
            self.config_data_preprocessing = json5.load(f)
        self.fps = self.config_data_preprocessing['fps']
        self.uniform_len = self.config_data_preprocessing["uniform_len"]
        self.num_blocks = self.config_data_preprocessing["num_blocks_per_clip"]

        # endregion

        # region Log.

        date_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.log_dir = os.path.join(self.config["dir_log"],
                                    log_tag + '_' + self.config['network']['name'] + '_' + date_time)
        os.makedirs(self.log_dir, exist_ok=True)

        tensorboard_log_dir = os.path.join(self.config["dir_log"],
                                           'Tensorboard_Log', log_tag + '_' + self.config['network']['name'] + '_' + date_time)
        os.makedirs(tensorboard_log_dir, exist_ok=True)

        self.writer = SummaryWriter(logdir=tensorboard_log_dir)

        os.makedirs(os.path.join(self.log_dir, 'Checkpoints'), exist_ok=True)

        # endregion

        # region Device.

        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')

        # endregion

        # region Data loader.

        self.dataset_train = TrainingDataset(os.path.join(self.config["dir_data"], "train.npz"))
        self.data_loader_train = DataLoader(self.dataset_train, batch_size=self.config['batch_size'], shuffle=True)

        self.dataset_val = TrainingDataset(os.path.join(self.config["dir_data"], "valid.npz"))
        self.data_loader_val = DataLoader(self.dataset_val, batch_size=self.config['batch_size'], shuffle=False)

        # endregion

        # region Network.

        self.net = initialize_net(self.config, self.config_data_preprocessing)

        self.net.to(self.device)

        # endregion

        # region Copy config file to log dir.

        with open(os.path.join(self.log_dir, 'config.json5'), 'w') as f:
            json5.dump(self.config, f, indent=4)

        # endregion

        # region Optimizer.

        if self.config['optimizer']['name'] == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.net.parameters(),
                                              lr=self.config['optimizer']['lr'],
                                              betas=self.config['optimizer']['betas'],
                                              eps=self.config['optimizer']['eps'],
                                              weight_decay=self.config['optimizer']['weight_decay'])
        else:
            raise NotImplementedError

        # endregion

        # region Criterion.

        self.criterion_l1 = torch.nn.L1Loss()

        # endregion

    def train(self) -> None:
        print('\nStart training......')

        # region Start timing.

        since = time.time()

        # endregion

        # region Epoch loop.

        num_epoch = self.config["num_epoch"]
        for epoch in range(1, num_epoch + 1):
            print(f'\nEpoch: {epoch}/{num_epoch}.')

            self.net.train()

            loss_train, loss_rot_train, loss_fgd_train, loss_act_train = 0., 0., 0., 0.
            counter = 0

            # region Data loader loop.

            pbar = tqdm(total=len(self.dataset_train), ascii=True)
            for _, batch in enumerate(self.data_loader_train):
                self.optimizer.zero_grad()

                batch_size = batch["audio"].shape[0]

                infer_res = infer_train(batch, self.device, self.net,
                                        self.uniform_len, self.num_blocks,
                                        self.config['network']['name'])

                motion_gt = infer_res[0]
                motion_pred = infer_res[1]

                # region Loss and net weights update.

                loss_rot = F.mse_loss(motion_gt, motion_pred)

                motion_gt_fgd = motion_gt.reshape(-1, motion_gt.shape[-1])
                motion_pred_fgd = motion_pred.reshape(-1, motion_pred.shape[-1])
                mu1, sigma1 = compute_statistics(motion_gt_fgd)
                mu2, sigma2 = compute_statistics(motion_pred_fgd)
                loss_fgd = calculate_fgd(mu1, sigma1, mu2, sigma2)

                distances_input = []
                distances_hat = []
                for i in range(0, motion_gt.shape[1], 10):
                    if motion_gt.shape[1] > (i+10):
                        dis_input = torch.norm(motion_gt[:, i, :] - motion_gt[:, i+10, :])
                        dis_hat = torch.norm(motion_pred[:, i, :] - motion_pred[:, i+10, :])
                        distances_input.append(dis_input)
                        distances_hat.append(dis_hat)
                    else:
                        dis_input = torch.norm(motion_gt[:, i, :] - motion_gt[:, -1, :])
                        dis_hat = torch.norm(motion_pred[:, i, :] - motion_pred[:, -1, :])
                        distances_input.append(dis_input)
                        distances_hat.append(dis_hat)
                distances_input = torch.tensor(distances_input)
                distances_hat = torch.tensor(distances_hat)

                # input과 hat의 활동량 비슷하게 유지, 다만 peak action의 위치는 상관없음
                distance_farthest_input, _ = torch.max(distances_input, 0)
                distance_farthest_hat, _ = torch.max(distances_hat, 0)

                loss_act = (distance_farthest_input - distance_farthest_hat).abs().to(self.device)

                loss = self.config["loss"]["rot"] * loss_rot + self.config["loss"]["fgd"] * loss_fgd + self.config["loss"]["act"] * loss_act

                loss.backward()
                self.optimizer.step()

                loss_rot_train += loss_rot.item() * batch_size * self.config["loss"]["rot"]
                loss_fgd_train += loss_fgd.item() * batch_size * self.config["loss"]["fgd"]
                loss_act_train += loss_act.item() * batch_size * self.config["loss"]["act"]
                loss_train += loss.item() * batch_size

                counter += batch_size

                # endregion

                # region Pbar update.

                pbar.set_description('lr: %s' % (str(self.optimizer.param_groups[0]['lr'])))
                pbar.update(batch_size)

                # endregion

            pbar.close()

            # endregion

            # region Epoch loss and log.

            loss_rot_train /= counter
            loss_fgd_train /= counter
            loss_act_train /= counter
            loss_train /= counter

            print('Training',
                  f'Loss: {loss_train:.4f} /',
                  f'Rot Loss: {loss_rot_train:.4f} /',
                  f'FGD Loss: {loss_fgd_train:.4f} /',
                  f'Act Loss: {loss_act_train:.4f} /',
                  )

            self.writer.add_scalar("Loss/Train", loss_train, epoch)
            self.writer.add_scalar("Rot_Loss/Train", loss_rot_train, epoch)
            self.writer.add_scalar("FGD_Loss/Train", loss_fgd_train, epoch)
            self.writer.add_scalar("Act_Loss/Train", loss_act_train, epoch)

            # endregion

            # region Checkpoints.

            if epoch % self.config['checkpoint_save_num_epoch'] == 0:
                torch.save(self.net.state_dict(),
                           os.path.join(self.log_dir, 'Checkpoints', f'checkpoint_{epoch // 1000}k{epoch % 1000}.pth'))

            # endregion

            # region Validation.

            valid_num_epoch = self.config['valid_num_epoch']
            if epoch % valid_num_epoch == 0:
                self.net.eval()

                loss_valid, loss_rot_valid, loss_fgd_valid, loss_act_valid = 0., 0., 0., 0.
                counter = 0

                with torch.no_grad():
                    for _, batch in enumerate(self.data_loader_val):
                        batch_size = batch["audio"].shape[0]

                        infer_res = infer_train(batch, self.device, self.net,
                                                self.uniform_len, self.num_blocks,
                                                self.config['network']['name'])

                        motion_gt = infer_res[0]
                        motion_pred = infer_res[1]

                        # region Loss and net weights update.

                        loss_rot = F.mse_loss(motion_gt, motion_pred)

                        motion_gt_fgd = motion_gt.reshape(-1, motion_gt.shape[-1])
                        motion_pred_fgd = motion_pred.reshape(-1, motion_pred.shape[-1])
                        mu1, sigma1 = compute_statistics(motion_gt_fgd)
                        mu2, sigma2 = compute_statistics(motion_pred_fgd)
                        loss_fgd = calculate_fgd(mu1, sigma1, mu2, sigma2)

                        distances_input = []
                        distances_hat = []
                        for i in range(0, motion_gt.shape[1], 10):
                            if motion_gt.shape[1] > (i + 10):
                                dis_input = torch.norm(motion_gt[:, i, :] - motion_gt[:, i + 10, :])
                                dis_hat = torch.norm(motion_pred[:, i, :] - motion_pred[:, i + 10, :])
                                distances_input.append(dis_input)
                                distances_hat.append(dis_hat)
                            else:
                                dis_input = torch.norm(motion_gt[:, i, :] - motion_gt[:, -1, :])
                                dis_hat = torch.norm(motion_pred[:, i, :] - motion_pred[:, -1, :])
                                distances_input.append(dis_input)
                                distances_hat.append(dis_hat)
                        distances_input = torch.tensor(distances_input)
                        distances_hat = torch.tensor(distances_hat)

                        # input과 hat의 활동량 비슷하게 유지, 다만 peak action의 위치는 상관없음
                        distance_farthest_input, _ = torch.max(distances_input, 0)
                        distance_farthest_hat, _ = torch.max(distances_hat, 0)

                        loss_act = (distance_farthest_input - distance_farthest_hat).abs().to(self.device)

                        loss = self.config["loss"]["rot"] * loss_rot + self.config["loss"]["fgd"] * loss_fgd + \
                               self.config["loss"]["act"] * loss_act

                        loss_rot_valid += loss_rot.item() * batch_size * self.config["loss"]["rot"]
                        loss_fgd_valid += loss_fgd.item() * batch_size * self.config["loss"]["fgd"]
                        loss_act_valid += loss_act.item() * batch_size * self.config["loss"]["act"]
                        loss_valid += loss.item() * batch_size

                        counter += batch_size

                        # endregion

                # region Valid loss and log.

                loss_rot_valid /= counter
                loss_fgd_valid /= counter
                loss_act_valid /= counter
                loss_valid /= counter

                print('Valid',
                      f'Loss: {loss_valid:.4f} /',
                      f'Rot Loss: {loss_rot_valid:.4f} /',
                      f'FGD Loss: {loss_fgd_valid:.4f} /',
                      f'Act Loss: {loss_act_valid:.4f} /',
                      )

                self.writer.add_scalar("Loss/valid", loss_valid, epoch)
                self.writer.add_scalar("Rot_Loss/valid", loss_rot_valid, epoch)
                self.writer.add_scalar("FGD_Loss/valid", loss_fgd_valid, epoch)
                self.writer.add_scalar("Act_Loss/valid", loss_act_valid, epoch)

                # endregion

            # endregion

        # endregion

        # region Save network.

        torch.save(self.net.state_dict(), os.path.join(self.log_dir, 'Checkpoints', 'trained_model.pth'))

        # endregion

        # region End timing.

        time_elapsed = time.time() - since
        print('\nTraining completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # endregion

        self.writer.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # raise ValueError('Must give config file path argument!')
        path_config = "Config/Trinity/config.json5"
        with open(path_config, 'r') as f:
            config = json5.load(f)

    else:
        with open(sys.argv[1], 'r') as f:
            config = json5.load(f)

    log_tag = sys.argv[-1] if len(sys.argv) == 3 else ''

    trainer = Trainer(config, log_tag)
    trainer.train()
# region Import.

from network import *

from Gesture_Lexicon.model import VQVAE_gcn, VQVAE_conv, RVQ_gcn, RVQ_conv, Transformer, autoencoder_conv, VAE_conv

import numpy as np

# from Gesture_Lexicon.lexeme_viewer import show_lexeme

# endregion


__all__ = ["infer_train", "infer_train_BU", "infer_train_forvqvae", "infer_train_forkmeans", "infer_train_forlexeme", "initialize_net", "vqvae_net"]


def infer_train(batch, device, net, uniform_len, num_blocks, name_net):
    if name_net == "RNN":
        aud = batch["audio"].to(device)  # [N, L, D]
        mo = batch["motion"].to(device)  # [N, L, D]
        lxm = batch["lexeme"].to(device)  # [N, B, D]

        mo_gt = mo[:, uniform_len: (num_blocks-1)*uniform_len, :]
        # lxm_gt = lxm[:, 1: (num_blocks-1), :]

        x_audio = aud.permute((0, 2, 1)).contiguous()
        # x_motion = mo.permute((0, 2, 1)).contiguous()
        x_lexeme = lxm.permute((0, 2, 1)).contiguous()

        mo_hat = net(x_audio, x_lexeme)

        mo_hat = mo_hat.permute((0, 2, 1)).contiguous()

        return mo_gt, mo_hat

    else:
        raise NotImplementedError

def infer_train_BU(batch, device, net, uniform_len, num_blocks, name_net):
    if name_net == "RNN":
        aud = batch["audio"].to(device)  # [N, L, D]
        mo = batch["motion"].to(device)  # [N, L, D]
        lxm = batch["lexeme"].to(device)  # [N, B, D]

        mo_gt = mo[:, uniform_len: (num_blocks-1)*uniform_len, :]
        # lxm_gt = lxm[:, 1: (num_blocks-1), :]

        x_audio = aud.permute((0, 2, 1)).contiguous()
        x_motion = mo.permute((0, 2, 1)).contiguous()
        x_lexeme = lxm.permute((0, 2, 1)).contiguous()

        mo_hat = net(x_audio, x_motion, x_lexeme)

        mo_hat = mo_hat.permute((0, 2, 1)).contiguous()

        return mo_gt, mo_hat

    else:
        raise NotImplementedError

def infer_train_forvqvae(batch, device, net, uniform_len, num_blocks, name_net, batch_size):
    # show_lexeme_net = show_lexeme().to(device)

    if name_net == "RNN":
        mo = batch["motion"].to(device)  # [N, L, D]
        x_motion = mo.permute(0, 2, 1).contiguous() # [B, R * J, T]
        x_motion = x_motion.view(x_motion.shape[2] // uniform_len, x_motion.shape[1], uniform_len)

        batch_size = batch_size * 10

        # # uniform len 만큼 나눠 test
        # gap = x_motion.shape[0] % batch_size
        # x_motion = x_motion[:-gap]
        # x_motion_list = x_motion.view(x_motion.shape[0]//batch_size, batch_size, x_motion.shape[1], x_motion.shape[2])
        # x_motion_last = x_motion[-gap:]
        # mo_hat = []
        # for x_mo in x_motion_list:
        #     _, mo_h, _, _, _ = net.forward(x_mo)
        #     mo_hat.append(mo_h)
        # if gap != 0:
        #     _, mo_h, _, _, _ = net.forward(x_motion_last)
        #     mo_hat.append(mo_h)
        # mo_hat = torch.cat(mo_hat, dim=0)
        # mo_hat = mo_hat.view(1, mo_hat.shape[1], mo_hat.shape[0]*mo_hat.shape[2])

        # # 한번에 테스트
        # _, mo_hat, _, _, _ = net.forward(x_motion)
        _, mo_hat, _, _, _, _ = net.forward(x_motion)
        mo_hat = mo_hat.contiguous().view(1, mo_hat.shape[1], mo_hat.shape[0] * mo_hat.shape[2])

        mo_gt = mo[:, uniform_len:-uniform_len, :]
        mo_hat = mo_hat.permute(0, 2, 1).contiguous()
        mo_hat = mo_hat[:, uniform_len:-uniform_len, :]

        return mo_gt, mo_hat

    else:
        raise NotImplementedError


def infer_train_forkmeans(batch, device, net, uniform_len, num_blocks, name_net):

    if name_net == "RNN":
        aud = batch["audio"].to(device)  # [N, L, D]  1 3470 80
        mo = batch["motion"].to(device)  # [N, L, D]  1 3470 48
        lxm = batch["lexeme"].to(device)  # [N, B, D]  1 347 96
        lxm_idx = batch["lexeme_index"]
        # lxm_idx = np.array(lxm_idx)

        mo_gt = mo  # 1 3450 162

        x_audio = aud.permute((0, 2, 1)).contiguous()
        x_motion = mo.permute((0, 2, 1)).contiguous()
        x_lexeme = lxm.permute((1, 0, 2)).contiguous()

        # x_lexeme = x_lexeme.permute((2, 0, 1)).contiguous()
        # x_lexeme = x_lexeme.squeeze(0)
        z_q, mo_hat, vq_loss = net.forward_forlxm(x_lexeme)

        # mo_gt = mo_gt.reshape(1, mo_gt.shape[0]*10, mo_gt.shape[1])
        mo_hat = mo_hat.reshape(1, mo_hat.shape[0] * 10, mo_hat.shape[1])
        mo_gt = mo[:, uniform_len: (num_blocks - 1) * uniform_len, :]
        mo_hat = mo_hat[:, uniform_len: (num_blocks - 1) * uniform_len, :]

        return mo_gt, mo_hat

    else:
        raise NotImplementedError


# for check lexeme
def infer_train_forlexeme(batch, device, net, uniform_len, num_blocks, name_net, cluster):

    # tmp_datapath = '../data/Trinity/Processed/Training_Data/train.npz'
    # tmp_data = dict(np.load(tmp_datapath))

    if name_net == "RNN":
        aud = batch["audio"].to(device)  # [N, L, D]  1 3470 80
        mo = batch["motion"].to(device)  # [N, L, D]  1 3470 48
        lxm = batch["lexeme"].to(device)  # [N, B, D]  1 347 96
        lxm_idx = batch["lexeme_index"]
        lxm_idx = np.array(lxm_idx)

        x_lexeme = lxm.squeeze(0)
        z_q, mo_hat = net.forward_forlxm(x_lexeme)

        # mo_gt = mo_gt.reshape(mo_gt.shape[1]//10, mo_gt.shape[2], 10)
        lxm_idx = lxm_idx.squeeze(0)
        for idx, indice in enumerate(lxm_idx):
            if indice == cluster:
                for i, _ in enumerate(mo_hat):
                    # mo_gt[i, :, :] = mo_gt[idx, :, :]
                    mo_hat[i, :, :] = mo_hat[idx, :, :]
                break

        mo_gt = mo[:, uniform_len:-uniform_len, :]
        # mo_hat = mo_hat.permute(0, 2, 1).contiguous()
        mo_hat = mo_hat.contiguous().view(1, mo_hat.shape[0] * mo_hat.shape[2], mo_hat.shape[1])
        mo_hat = mo_hat[:, uniform_len:-uniform_len, :]

        return mo_gt, mo_hat

    else:
        raise NotImplementedError


def initialize_net(config, config_data_preprocessing):
    if config['network']['name'] == "RNN":
        net = MotionGenerator_RNN(**config['network']['hparams'])
        # net = show_lexeme

    else:
        raise NotImplementedError

    return net

def vqvae_net(config, config_data_preprocessing):
    if config['network']['name'] == "VQVAE_gcn":
        net = VQVAE_gcn(**config['network']['hparams'])
    elif config['network']['name'] == "VQVAE_conv":
        net = VQVAE_conv(**config['network']['hparams'])
    elif config['network']['name'] == 'autoencoder_conv':
        net = autoencoder_conv(**config['network']['hparams'])
    elif config['network']['name'] == 'VAE_conv':
        net = VAE_conv(**config['network']['hparams'])
    elif config['network']['name'] == 'RVQ_gcn':
        net = RVQ_gcn(**config['network']['hparams'])
    elif config['network']['name'] == 'RVQ_conv':
        net = RVQ_conv(**config['network']['hparams'])
    elif config['network']['name'] == 'Transformer':
        net = Transformer(**config['network']['hparams'])
    else:
        raise NotImplementedError

    return net
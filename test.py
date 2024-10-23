import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

def euler_to_6d(motion: pd.DataFrame, rotation_order: str) -> pd.DataFrame:
    new_motion = pd.DataFrame(index=motion.index)

    joint_infos = {}
    for col in motion.columns:
        infos = col.split('_')
        channel = infos[-1][1:]
        joint = '_'.join(infos[:-1])

        if joint not in joint_infos.keys():
            joint_infos[joint] = [channel]
        else:
            joint_infos[joint].append(channel)

    for joint, channels in joint_infos.items():
        if "rotation" in channels:
            r_cols = [
                "%s_%srotation" % (joint, rotation_order[0]),
                "%s_%srotation" % (joint, rotation_order[1]),
                "%s_%srotation" % (joint, rotation_order[2])
            ]
            r = motion[r_cols].to_numpy()

            mtx = Rotation.from_euler(rotation_order, r, degrees=True).as_matrix()
            sixd = np.reshape(mtx[:, :2, :], (-1, 6))

            # exps = unroll(np.array([euler2expmap(f, rotation_order, True) for f in r]))

            new_motion[["%s_ZZrotation" % joint,
                        "%s_ZXrotation" % joint,
                        "%s_ZYrotation" % joint,
                        "%s_XZrotation" % joint,
                        "%s_XXrotation" % joint,
                        "%s_XYrotation" % joint]] = pd.DataFrame(data=sixd[:, :], index=new_motion.index)

        if "position" in channels:
            p_cols = [
                "%s_Xposition" % joint,
                "%s_Yposition" % joint,
                "%s_Zposition" % joint
            ]

            new_motion[p_cols] = motion[p_cols].copy()

        if ('position' not in channels) and ('rotation' not in channels):
            print(channels)
            raise ValueError('Motion channel is wrong.')

    assert 2 * len(motion.columns) == len(new_motion.columns)

    return new_motion  # ZXY to gamma beta alpha

def sixd_to_euler(motion: pd.DataFrame, rotation_order: str) -> pd.DataFrame:
    new_motion = pd.DataFrame(index=motion.index)

    joint_infos = {}
    for col in motion.columns:
        infos = col.split('_')
        if ("position" in infos[-1]) or ("rotation" in infos[-1]):
            # channel = infos[-1][1:]
            channel = infos[-1][2:]
        else:
            channel = infos[-1]
        joint = '_'.join(infos[:-1])

        if joint not in joint_infos.keys():
            joint_infos[joint] = [channel]
        else:
            joint_infos[joint].append(channel)

    for joint, channels in joint_infos.items():
        # if ("rotation" in channels) and ("rotationN" in channels):
        if "rotation" in channels:
            r_cols = [
                "%s_ZZrotation" % joint,
                "%s_ZXrotation" % joint,
                "%s_ZYrotation" % joint,
                "%s_XZrotation" % joint,
                "%s_XXrotation" % joint,
                "%s_XYrotation" % joint
            ]
            r = motion[r_cols].to_numpy()

            a1, a2 = r[..., :3], r[..., 3:]
            b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
            b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
            b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
            b3 = np.cross(b1, b2, axis=-1)
            motion_b = np.stack((b1, b2, b3), axis=-2)

            motion_b = Rotation.from_matrix(motion_b).as_euler(rotation_order, degrees=True)

            new_motion[["%s_%srotation" % (joint, rotation_order[0]),
                        "%s_%srotation" % (joint, rotation_order[1]),
                        "%s_%srotation" % (joint, rotation_order[2])]] = pd.DataFrame(data=motion_b[:, :], index=new_motion.index)

        if "position" in channels:
            p_cols = [
                "%s_Xposition" % joint,
                "%s_Yposition" % joint,
                "%s_Zposition" % joint
            ]

            new_motion[p_cols] = motion[p_cols].copy()

        if ('position' not in channels) and ('rotation' not in channels):
            print(channels)
            raise ValueError('Motion channel is wrong.')

    return new_motion

if __name__ == '__main__':

    motion = pd.read_csv(os.path.join("Recording_001_motion.csv"), index_col=0)

    motion_filtered = motion.drop([c for c in motion.columns if (('position' in c) and ('Hips' + '_' not in c))], axis=1)

    new_motion = euler_to_6d(motion_filtered, 'ZXY')  # 6d

    new_motion.to_csv(os.path.join("Recording_001_motion_6d.csv"))

    new_motion = sixd_to_euler(new_motion, 'ZXY')

    new_motion.to_csv(os.path.join("Recording_001_motion_euler.csv"))




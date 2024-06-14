import torch
import torch.nn as nn
import absl.flags as flags
from absl import app
import numpy as np
import torch.nn.functional as F

from network.fs_net_repo.PoseR import Rot_red, Rot_green
from network.fs_net_repo.PoseTs import Pose_Ts
from network.fs_net_repo.FaceRecon import FaceRecon

FLAGS = flags.FLAGS


class PoseNet9D(nn.Module):
    def __init__(self):
        # 继承自nn.Module时使用的一种做法。
        super(PoseNet9D, self).__init__()
        # Used the fsnet rot_green and rot_red directly
        # 直接利用了“fsnet”工具包或模型中已经实现好的“rot_green”和“rot_red”功能，提高了开发效率或利用了已优化的算法实现。
        
        # 这两个网络对应了图中的 pose and size Estimation中的计算R的部分
        self.rot_green = Rot_green() 
        self.rot_red = Rot_red()

        # 这一个网络代表了 HS-encoder
        self.face_recon = FaceRecon()
        
        # 这一个网络对应了图pose and size Estimation中的计算 t，S
        self.ts = Pose_Ts()

    def forward(self, points, obj_id):
        # bs为Batch Size 为批大小，p_num为点的个数
        bs, p_num = points.shape[0], points.shape[1]
        # 中心化点云数据，沿着张量的第二个维度（索引从0开始，所以1对应的是第二维）计算平均值。
        # keepdim=True: 默认情况下，当沿着某维度执行聚合操作时（如求和、平均等），该维度会被“压缩”掉，也就是说该维度会消失。
        # 但通过设置keepdim=True，可以保留被聚合维度的大小为1，这样输出张量的维度数量与原张量相同，只是被聚合的维度变成了长度为1。
        recon, face, feat = self.face_recon(points - points.mean(dim=1, keepdim=True), obj_id)

        if FLAGS.train:
            recon = recon + points.mean(dim=1, keepdim=True)
            # handle face
            face_normal = face[:, :, :18].view(bs, p_num, 6, 3)  # normal
            # face是一个至少三维的张量，这里的索引操作[:, :, :18]是从face的第三维度（即最后一个轴，假设face是三维的）选取前18个元素。
            # 这里的:表示选取所有元素，所以在第一和第二维度上选取了所有元素，而在第三个维度上选取了前18个元素。
            # 这通常意味着face张量的每个元素（例如，每个顶点或每个面）有18个相关联的值，而此操作只保留了这些值的前18个。
            # view 用于在不改变数据的情况下重塑张量的形状。这里的bs, p_num, 6, 和 3是新的形状维度。 这里是在做什么？

            face_normal = face_normal / torch.norm(face_normal, dim=-1, keepdim=True)  # bs x nunm x 6 x 3
            # norm计算范数 -1：沿着最后一个维度计算
            face_dis = face[:, :, 18:24]  # bs x num x  6
            face_f = F.sigmoid(face[:, :, 24:])  # bs x num x 6
        else:
            # 这些变量初始化为None
            face_normal, face_dis, face_f, recon = [None]*4
        #  rotation
        # permute：换维度
        green_R_vec = self.rot_green(feat.permute(0, 2, 1))  # b x 4
        red_R_vec = self.rot_red(feat.permute(0, 2, 1))   # b x 4

        # normalization
        p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        # sigmoid for confidence
        f_green_R = F.sigmoid(green_R_vec[:, 0])
        f_red_R = F.sigmoid(red_R_vec[:, 0])

        # translation and size  它使用了PyTorch库的功能，目的是将两个张量的特征沿着特定的维度合并在一起。
        feat_for_ts = torch.cat([feat, points-points.mean(dim=1, keepdim=True)], dim=2)
        T, s = self.ts(feat_for_ts.permute(0, 2, 1))
        Pred_T = T + points.mean(dim=1)  # bs x 3
        Pred_s = s  # this s is not the object size, it is the residual

        return recon, face_normal, face_dis, face_f, p_green_R, p_red_R, f_green_R, f_red_R, Pred_T, Pred_s


def main(argv):
    classifier_seg3D = PoseNet9D()

    points = torch.rand(2, 1000, 3)
    import numpy as np
    obj_idh = torch.ones((2, 1))
    obj_idh[1, 0] = 5
    '''
    if obj_idh.shape[0] == 1:
        obj_idh = obj_idh.view(-1, 1).repeat(points.shape[0], 1)
    else:
        obj_idh = obj_idh.view(-1, 1)

    one_hot = torch.zeros(points.shape[0], 6).scatter_(1, obj_idh.cpu().long(), 1)
    '''
    recon, f_n, f_d, f_f, r1, r2, c1, c2, t, s = classifier_seg3D(points, obj_idh)
    t = 1



if __name__ == "__main__":
    print(1)
    from config.config import *
    app.run(main)






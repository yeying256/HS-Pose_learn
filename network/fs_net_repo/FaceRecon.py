# Modified from FS-Net
import torch.nn as nn
import network.fs_net_repo.gcn3d as gcn3d
import torch
import torch.nn.functional as F
from absl import app
import absl.flags as flags

FLAGS = flags.FLAGS

# 通过一系列卷积和池化操作从 3D 点云数据中提取特征，并在训练时进行重建和面部重建预测。
# 模型使用了基于 GCN（图卷积网络）的层来处理 3D 点云，并结合了全局和局部特征来实现面部重建。
class FaceRecon(nn.Module):
    def __init__(self):
        super(FaceRecon, self).__init__()
        self.neighbor_num = FLAGS.gcn_n_num
        # 通常代表每个节点邻居点的数量，

        self.support_num = FLAGS.gcn_sup_num
        # 定义了支持向量或支持图的数量

        # 3D convolution for point cloud
        self.conv_0 = gcn3d.HSlayer_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1 = gcn3d.HS_layer(128, 128, support_num=self.support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = gcn3d.HS_layer(128, 256, support_num=self.support_num)
        self.conv_3 = gcn3d.HS_layer(256, 256, support_num=self.support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d.HS_layer(256, 512, support_num=self.support_num)

        # 标准化，正则化层 归一化，讲数据放到1左右。均值为0方差为1
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        self.recon_num = 3
        self.face_recon_num = FLAGS.face_recon_c

        dim_fuse = sum([128, 128, 256, 256, 512, FLAGS.obj_c])
        # 16: total 6 categories, 256 is global feature

        if FLAGS.train:
            # Sequential 按顺序堆叠了一系列的层
            self.conv1d_block = nn.Sequential(
                nn.Conv1d(dim_fuse, 512, 1), # 第一层卷积，输入通道数为dim_fuse，输出通道数为512，卷积核大小为1
                nn.BatchNorm1d(512),# 批归一化层，作用于512个特征通道
                nn.ReLU(inplace=True),# ReLU激活函数，inplace=True意味着操作将在输入张量上直接进行，节省内存
                nn.Conv1d(512, 512, 1),# 第二层卷积，输入和输出通道数均为512，卷积核大小为1
                nn.BatchNorm1d(512),# 批归一化层，再次作用于512个特征通道
                nn.ReLU(inplace=True),# ReLU激活函数
                nn.Conv1d(512, 256, 1),# 第三层卷积，输入通道数为512，输出通道数为256，卷积核大小为1
                nn.BatchNorm1d(256),# 批归一化层，作用于256个特征通道
                nn.ReLU(inplace=True),# ReLU激活函数
            )

            # 重建头
            self.recon_head = nn.Sequential(
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, self.recon_num, 1),
            )

            self.face_head = nn.Sequential(
                nn.Conv1d(FLAGS.feat_face + 3, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                
                nn.Conv1d(512, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),

                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                
                nn.Conv1d(128, self.face_recon_num, 1),  # Relu or not?
            )

    # 在变量后面家双引号是代表注释，用来描述输入变量的预期结构
    # bs代表batch size 批处理大小，多少个批。 vertex_num代表顶点的数量 每个顶点有3个坐标值
    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                cat_id: "tensor (bs, 1)",
                ):
        """
        输入参数
        vertices: 形状为(bs, vertex_num, 3)的张量，其中bs是batch size，vertex_num是顶点的数量，3表示每个顶点在三维空间中的坐标（x, y, z）。
        这意味着vertices张量包含了批次中每个样本的所有顶点位置信息。
        cat_id: 形状为(bs, 1)的张量，可能是类别ID或其他某种分类标签，通常用于条件生成或分类任务中。每个样本对应一个类别ID。
        Return: (bs, vertice_num, class_num)
        """
        #  concate feature
        bs, vertice_num, _ = vertices.size()

        # cat_id to one-hot
        # 如果我们要训练的模型只有一种类型，那么
        if cat_id.shape[0] == 1:
            # 其实是保持了大小obj_idh与cat_id相等
            obj_idh = cat_id.view(-1, 1).repeat(cat_id.shape[0], 1)
        else:
            # 本质上也没有改变形状。。
            obj_idh = cat_id.view(-1, 1)

        # one-hot编码 
        # FLAGS.obj_c 是总共有多少个类别，那么bs是这次我们只训练哪些类别，bs里面的内容是总类别的索引。
        # 先定义了一个bs X obj_c大小的
        # scatter_是改变前面的数值，第一个参数1代表改变列的数值。 如果obj_idh为[[3],[1],[2]]，bs=3,obj_c =5那么one_hot =
        # [0,0,0,1,0]
        # [0,1,0,0,0]
        # [0,0,1,0,0]
        one_hot = torch.zeros(bs, FLAGS.obj_c).to(cat_id.device).scatter_(1, obj_idh.long(), 1)
        # bs x verticenum x 6
        
        # ss = time.time()
        # neighbor_num为周围点的数量，vertices代表顶点信息"tensor (bs, vetice_num, 3)"
        fm_0 = F.relu(self.conv_0(vertices, self.neighbor_num), inplace=True)
        # x.transpose(1, 2)变换维度，如，一共有3个维度，10，20,30，则变换后维度为：10,30,20
        fm_1 = F.relu(self.bn1(self.conv_1(vertices, fm_0, self.neighbor_num).transpose(1, 2)).transpose(1, 2), inplace=True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)

        # v_pool_1// 8 和 neighbor_num 比比谁小就传进去谁 //是整数除法
        fm_2 = F.relu(self.bn2(self.conv_2(v_pool_1, fm_pool_1, 
                                           min(self.neighbor_num, v_pool_1.shape[1] // 8)).transpose(1, 2)).transpose(1, 2), inplace=True)
        fm_3 = F.relu(self.bn3(self.conv_3(v_pool_1, fm_2, 
                                           min(self.neighbor_num, v_pool_1.shape[1] // 8)).transpose(1, 2)).transpose(1, 2), inplace=True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        fm_4 = self.conv_4(v_pool_2, fm_pool_2, min(self.neighbor_num, v_pool_2.shape[1] // 8))
        f_global = fm_4.max(1)[0]  # (bs, f)

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor_new(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor_new(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor_new(fm_4, nearest_pool_2).squeeze(2)
        one_hot = one_hot.unsqueeze(1).repeat(1, vertice_num, 1)  # (bs, vertice_num, cat_one_hot)

        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, one_hot], dim=2)
        '''
        feat_face = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=2)
        feat_face = torch.mean(feat_face, dim=1, keepdim=True)  # bs x 1 x channel
        feat_face_re = feat_face.repeat(1, feat.shape[1], 1)
        '''

        if FLAGS.train:
            feat_face_re = f_global.view(bs, 1, f_global.shape[1]).repeat(1, feat.shape[1], 1).permute(0, 2, 1)
            # feat is the extracted per pixel level feature

            conv1d_input = feat.permute(0, 2, 1)  # (bs, fuse_ch, vertice_num)
            conv1d_out = self.conv1d_block(conv1d_input)

            recon = self.recon_head(conv1d_out)
            # average pooling for face prediction
            feat_face_in = torch.cat([feat_face_re, conv1d_out, vertices.permute(0, 2, 1)], dim=1)
            face = self.face_head(feat_face_in)
            return recon.permute(0, 2, 1), face.permute(0, 2, 1), feat
        else:
            recon, face = None, None
            return recon, face, feat


def main(argv):
    classifier_seg3D = FaceRecon()

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
    recon, face, feat = classifier_seg3D(points, obj_idh)
    face = face.squeeze(0)
    t = 1



if __name__ == "__main__":
    print(1)
    from config.config import *
    app.run(main)



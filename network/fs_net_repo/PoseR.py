import torch.nn as nn
import torch
import torch.nn.functional as F
import absl.flags as flags
from absl import app
from config.config import *
FLAGS = flags.FLAGS


class Rot_green(nn.Module):
    def __init__(self):
        super(Rot_green, self).__init__()
        self.f = FLAGS.feat_c_R
        self.k = FLAGS.R_c
        # 初始化了两个变量 self.f 和 self.k，它们分别代表特征通道数和旋转参数的数量，其值来自全局变量 FLAGS.feat_c_R 和 FLAGS.R_c。

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)

        # 大约会有20%的神经元被随机地设置为0。
        self.drop1 = nn.Dropout(0.2)

        # 归一化处理层：使其均值接近0，方差接近1。
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        # 移除张量x在第三个维度（索引为2）上的单位维度
        x = x.squeeze(2)
        # x.contiguous()确保张量 x 的数据是连续存储在内存中的。
        x = x.contiguous()

        return x


class Rot_red(nn.Module):
    # 同green一样，没什么区别
    def __init__(self):
        super(Rot_red, self).__init__()
        self.f = FLAGS.feat_c_R
        self.k = FLAGS.R_c

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()

        return x


def main(argv):
    points = torch.rand(2, 1350, 1500)  # batchsize x feature x numofpoint
    rot_head = Rot_red()
    rot = rot_head(points)
    t = 1


if __name__ == "__main__":
    app.run(main)

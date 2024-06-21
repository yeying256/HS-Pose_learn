"""
@Author: Linfang Zheng 
@Contact: zhenglinfang@icloud.com
@Time: 2023/03/06
@Note: Modified from 3D-GCN: https://github.com/zhihao-lin/3dgcn
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.fs_net_loss import FLAGS
# 三维数据上的图卷积网络

def get_neighbor_index(vertices: "(bs, vertice_num, 3)", neighbor_num: int):
    """
    这段代码定义了一个名为 get_neighbor_index 的函数，其目的是为了计算每个顶点（点云中的点）的最近邻顶点索引。
    这是点云处理和图形神经网络（GNNs）中常见的预处理步骤，特别是在构建点云的邻接结构时。
    Return: (bs, vertice_num, neighbor_num)
    """
    # 计算每两个点之间的点积 计算所有点之间的距离
    inner = torch.bmm(vertices, vertices.transpose(1, 2))  # (bs, v, v)

    # 计算每个定点的坐标平方 
    quadratic = torch.sum(vertices ** 2, dim=2)  # (bs, v)
    
    # quadratic.unsqueeze(1)： (bs, 1, vertice_num)
    # quadratic.unsqueeze(2)： (bs, vertice_num, 1)
    # 利用了python的广播机制
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)

    # 用于查找沿着指定维度的 k 个最大或最小元素。在这个情况下，我们关注的是最小元素，因为我们正在寻找距离最短的邻居。
    # k为返回多少个元素。加一代表加上自身，然后去掉，dim=-1沿着最后一个维度。
    # largest=False: 这个参数决定了 topk 是否应该查找最大的 k 个值。设置为false就是找最小值
    # torch.topk() 返回一个元组，其中第一个元素是 k 个最小值（或最大值，取决于 largest 参数）
    # 第二个元素是这些值对应的原始张量中的索引。通过 [1]，我们选择了第二个元素，即索引。
    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    # neighbor_index 将是一个形状为 (bs, vertice_num, neighbor_num + 1) 的张量
    # 去掉每一个数据的第一个数据。
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index

# 两组点云找最近的点的index
def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
    """
    target是要找的目标点，source是要从source中的每个点取出一个离target最近的点的索引。
    也就是说，最终返回的索引是source的索引。每一行都代表离着target每一个点最近的source的点的索引。
    Return: (bs, v1, 1)
    """
    # 执行批量矩阵乘法
    inner = torch.bmm(target, source.transpose(1, 2))  # (bs, v1, v2)

    s_norm_2 = torch.sum(source ** 2, dim=2)  # (bs, v2) 
    t_norm_2 = torch.sum(target ** 2, dim=2)  # (bs, v1)
    # 计算距离
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    # k只找一个点，dim 是从最后一个维度开始找 
    # torch.topk() 返回一个元组，其中第一个元素是 k 个最小值（或最大值，取决于 largest 参数）
    # 第二个元素是这些值对应的原始张量中的索引。通过 [1]，我们选择了第二个元素，即索引。
    nearest_index = torch.topk(d_norm_2, k=1, dim=-1, largest=False)[1]
    
    return nearest_index


def indexing_neighbor_new(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)"):
    """从一个张量中基于给定的索引提取邻域特征。给定的索引从一个大的特征张量中抽取局部的邻域特征。

    Args:
        tensor (bs, vertice_num, dim): 输入张量
        index (bs, vertice_num, neighbor_num): _description_  形状为 (bs, vertice_num, neighbor_num) 的张量，存储了每个顶点的邻居顶点索引。
        其中，邻居点的索引就是对这一个点周围最近的neighbor_num个点的索引（不包括自己）

    Returns:
        _type_: _description_ 按照原始的批次、顶点和邻域结构排列，每个顶点的邻域特征已经被正确地抽取出来了。
    """
    bs, num_points, num_dims = tensor.size()
    # arange：等差数列,因为要加上批次
    idx_base = torch.arange(0, bs, device=tensor.device).view(-1, 1, 1) * num_points

    # 要加上批次，第一个批次的数根据广播机制都加0,第二个批次的数都加上每个批次的点云个数。
    idx = index + idx_base 
    # 将这个多维张量转化为一维张量。
    idx = idx.view(-1)

    # 先把tensor展成了一个长条形的矩阵，每一行代表一个点的三个坐标，然后抽取第idx行组成一个新的矩阵。
    feature = tensor.reshape(bs * num_points, -1)[idx, :]
    _, out_num_points, n = index.size()

    # view改变形状不改变数据 num_dims是三维，n是邻域点的数目，out_num_points是每个点
    feature = feature.view(bs, out_num_points, n, num_dims)
    return feature

def get_neighbor_direction_norm(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, neighbor_num)", return_unnormed = False):
    """
    计算每个顶点与其邻居顶点之间的方向向量，并对其进行归一化。
    Return: (bs, vertice_num, neighobr_num, 3)
    此函数返回了每个点的
    """
    neighbors = indexing_neighbor_new(vertices, neighbor_index)  # (bs, v, n, 3)
    # vertices.unsqueeze(2)是在索引为2的位置增加一个新的维度，这样才能做广播减法，而后获得了方向向量。
    neighbor_direction = neighbors - vertices.unsqueeze(2)

    # 将每一个向量转化为单位方向向量
    neighbor_direction_norm = F.normalize(neighbor_direction, dim=-1)
    if return_unnormed:
        return neighbor_direction_norm.float(), neighbor_direction
    else:
        return neighbor_direction_norm.float()

class HSlayer_surface(nn.Module):
    """Extract structure feafure from surface, independent from vertice coordinates"""

# 128,gcn_sup_num
    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.feat_k = 8
        self.kernel_num = kernel_num
        self.support_num = support_num
        self.relu = nn.ReLU(inplace=True)
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.STE_layer = nn.Conv1d(3, kernel_num, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(2*kernel_num, kernel_num, kernel_size=1, bias=False)
        self.initialize()

    def initialize(self):
        # ？？？？？？？
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)

        # -stdv 和 stdv 分别表示均匀分布的最小值和最大值。这意味着 self.directions 中的每个元素都会被一个从 -stdv 到 stdv 范围内的随机数替换。
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                neighbor_num: 'int'):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """

        # -1和-2表示交换倒数第一个维度和倒数第二个维度
        f_STE = self.STE_layer(vertices.transpose(-1,-2)).transpose(-1,-2).contiguous()
        # RF-P是距离的感受野，这里只是算了一下距离感受野的方向向量
        receptive_fields_norm, _ = get_receptive_fields(neighbor_num, vertices, mode='RF-P')
        
        feature = self.graph_conv(receptive_fields_norm, vertices, neighbor_num)
        feature = self.ORL_forward(feature, vertices, neighbor_num)

        return feature + f_STE 
    
    def graph_conv(self, receptive_fields_norm,
                   vertices: "(bs, vertice_num, 3)",
                   neighbor_num: 'int',):
        """ 3D graph convolution using receptive fields. More details please check 3D-GCN: https://github.com/zhihao-lin/3dgcn

        Return (bs, vertice_num, kernel_num): the extracted feature.
        """
        bs, vertice_num, _ = vertices.size()


        # 是为了计算接收域（receptive fields）与支持方向（support directions）之间的关系，这在诸如图神经网络（GNNs）或点云处理等高级深度学习模型中很常见。
        # 取范数
        support_direction_norm = F.normalize(self.directions, dim=0)  # (3, s * k)
        # 这是在干什么？
        theta = receptive_fields_norm @ support_direction_norm  # (bs, vertice_num, neighbor_num, s*k)

        # 激活函数
        theta = self.relu(theta)
        theta = theta.reshape(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        theta = torch.max(theta, dim=2)[0]  # (bs, vertice_num, support_num, kernel_num)
        feature = torch.mean(theta, dim=2)  # (bs, vertice_num, kernel_num)
        return feature

    def ORL_forward(self, feature, vertices, neighbor_num):
        f_global = get_ORL_global(feature, vertices, neighbor_num) 
        feat = torch.cat([feature, f_global], dim=-1)
        feature = self.conv2(feat.transpose(-1,-2)).transpose(-1,-2).contiguous() + feature
        return feature

# 这一部分代表了图中左上角的部分
class HS_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace=True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1)* out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))

        self.feat_k = 8
        self.STE_layer = nn.Conv1d(self.in_channel, self.out_channel, kernel_size=1, bias=False)

        self.conv2 = nn.Conv1d(2*out_channel, out_channel, kernel_size=1, bias=False)

        self.initialize()

    def initialize(self):

        # 这些也看不太明白  ？？？？
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    # 输入的点云
    def forward(self, vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)",
                neighbor_num: "int"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        f_STE = self.STE_layer(feature_map.transpose(-1,-2)).transpose(-1,-2).contiguous()
        receptive_fields_norm, neighbor_index = get_receptive_fields(neighbor_num, 
                                                                       vertices, 
                                                                       feature_map=feature_map, 
                                                                       mode='RF-F')
        feature = self.graph_conv(receptive_fields_norm, neighbor_index, feature_map, vertices, neighbor_num)
        feature_fuse = self.ORL_forward(feature, vertices, neighbor_num)
        return feature_fuse + f_STE 
    
    def graph_conv(self, receptive_fields_norm, neighbor_index, feature_map,
                   vertices: "(bs, vertice_num, 3)",
                   neighbor_num: 'int',):
        """ 3D graph convolution using receptive fields. More details please check 3D-GCN: https://github.com/zhihao-lin/3dgcn
        
        Return (bs, vertice_num, kernel_num): the extracted feature.
        图卷积神经网络
        """


        # ？？？？？
        bs, vertice_num, _ = vertices.size()
        support_direction_norm = F.normalize(self.directions, dim=0)
        theta = receptive_fields_norm @ support_direction_norm  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        theta = self.relu(theta)
        theta = theta.reshape(bs, vertice_num, neighbor_num, -1) # (bs, vertice_num, neighbor_num, support_num * out_channel)

        feature_map = feature_map @ self.weights + self.bias  # (bs, vertice_num, neighbor_num, (support_num + 1) * out_channel)
        feature_center = feature_map[:, :, :self.out_channel]  # (bs, vertice_num, out_channel)
        feature_support = feature_map[:, :, self.out_channel:]  # (bs, vertice_num, support_num * out_channel)

        feature_support = indexing_neighbor_new(feature_support, neighbor_index)  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = theta * feature_support  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = activation_support.view(bs, vertice_num, neighbor_num, self.support_num, self.out_channel)
        activation_support = torch.max(activation_support, dim=2)[0]  # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.mean(activation_support, dim=2)  # (bs, vertice_num, out_channel)
        feature = feature_center + activation_support  # (bs, vertice_num, out_channel)

        # ？？？？？
        return feature

    def ORL_forward(self, feature_fuse, vertices, neighbor_num):
        f_global = get_ORL_global(feature_fuse, vertices, neighbor_num) 
        feat = torch.cat([feature_fuse, f_global], dim=-1)
        feature_fuse = self.conv2(feat.transpose(-1,-2)).transpose(-1,-2).contiguous() + feature_fuse
        return feature_fuse

def get_receptive_fields(neighbor_num: "int", 
                         vertices: "(bs, vertice_num, 3)", 
                         feature_map: "(bs, vertice_num, in_channel)" = None, 
                         mode: 'string' ='RF-F'):
    """ Form receptive fields amd norm the direction vectors according to the mode.
    
    Args:
        neighbor_num (int): neighbor number.
        vertices (tensor): The 3D point cloud for forming receptive fields 
        feature_map (tensor, optional): The features for finding neighbors and should be provided if 'RF-F' is used. Defaults to None. 
        mode (str, optional): The metrics for finding the neighbors. 
        Should only use 'RF-F' or 'RF-P'. 'RF-F' means forming the receptive fields using feature-distance, RF-F特征感受野
        and 'RF-P' means using point-distance. Defaults to 'RF-F'.距离感受野
    """
    assert mode in ['RF-F', 'RF-P']
    if mode == 'RF-F':
        assert feature_map is not None, "The feature_map should be provided if 'RF-F' is used"
        feat = feature_map
    else:
        # 距离感受野只是把点云传入进去
        feat = vertices
    # 获得每个点云的邻居节点的index 
    neighbor_index = get_neighbor_index(feat, neighbor_num)
    # 获取每个点的 方向向量，且这些向量是单位向量   
    neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
    return neighbor_direction_norm, neighbor_index

def get_ORL_global(feature:'(bs, vertice_num, in_channel)', vertices: '(bs, vertice_num, 3)', 
                          neighbor_num:'int'):
    vertice_num = feature.size(1)
    neighbor_index = get_neighbor_index(vertices, neighbor_num)
    feature = indexing_neighbor_new(feature, neighbor_index) # batch_size, num_points, k, num_dims
    feature = torch.max(feature, dim=2)[0]
    f_global = torch.mean(feature, dim=1, keepdim=True).repeat(1, vertice_num, 1)
    return f_global

class Pool_layer(nn.Module):
    # 点云数据的池化操作 
    def __init__(self, pooling_rate: int = 4, neighbor_num: int = 4):
        super().__init__()
        # 指定顶点数量减少的比例pooling_rate
        self.pooling_rate = pooling_rate 
        #  指定用于池化计算的邻居顶点数量
        self.neighbor_num = neighbor_num

    def forward(self,
                vertices: "(bs, vertice_num, 3)", 
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()

        # 这是个索引
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        # 从feature_map中 返回所有的邻居点特征，neighbor_feature为 bs：批量，vertice_num是每个点，neighbor_num是每个点的邻居点数目，channel_num是3，
        neighbor_feature = indexing_neighbor_new(feature_map,
                                             neighbor_index)  # (bs, vertice_num, neighbor_num, channel_num)
        # 找每个点最近的那个点的特征。[0]是为了返回最大值，二1十返回最大值所在的索引。
        pooled_feature = torch.max(neighbor_feature, dim=2)[0]  # (bs, vertice_num, channel_num)

        # 压缩率 压缩到多少
        pool_num = int(vertice_num / self.pooling_rate)
        # torch.randperm(vertice_num) 是生成一个vertice_num个元素的张量，里面的数据无序排列，都是从0到vertice_num-1的随机数。
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :]  # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, sample_idx, :]  # (bs, pool_num, channel_num)
        return vertices_pool, feature_map_pool


def test():
    import time
    bs = 8
    v = 1024
    dim = 3
    n = 20
    vertices = torch.randn(bs, v, dim)

    s = 3
    conv_1 = HSlayer_surface(kernel_num=32, support_num=s, neighbor_num=n)
    conv_2 = HS_layer(in_channel=32, out_channel=64, support_num=s)
    pool = Pool_layer(pooling_rate=4, neighbor_num=4)

    print("Input size: {}".format(vertices.size()))
    start = time.time()
    f1 = conv_1(vertices)
    print("\n[1] Time: {}".format(time.time() - start))
    print("[1] Out shape: {}".format(f1.size()))
    start = time.time()
    f2 = conv_2(vertices, f1, n)
    print("\n[2] Time: {}".format(time.time() - start))
    print("[2] Out shape: {}".format(f2.size()))
    start = time.time()
    v_pool, f_pool = pool(vertices, f2)
    print("\n[3] Time: {}".format(time.time() - start))
    print("[3] v shape: {}, f shape: {}".format(v_pool.size(), f_pool.size()))


if __name__ == "__main__":
    test()

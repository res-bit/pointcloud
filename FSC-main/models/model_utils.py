import torch
from torch import nn, einsum
import torch.nn.functional as F
from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation
from typing import Iterable
import numpy as np
import pdb

class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  if_bn=True, activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class MLP(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Linear(last_channel, out_channel))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out

def sample_and_group(xyz, points, npoint, nsample, radius, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, N)（B：批量大小 3：维度 N：点个数）
        points: Tensor, (B, f, N)（f:特征维度）
        npoint: int （采样点数）
        nsample: int （每个采样点的邻域点数）
        radius: float （邻域半径）
        use_xyz: boolean （是否将坐标信息包含在特征中）

    Returns:
        new_xyz: Tensor, (B, 3, npoint) （采样的关键点坐标）
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample) （新的特征）
        idx_local: Tensor, (B, npoint, nsample) （邻域点索引）
        grouped_xyz: Tensor, (B, 3, npoint, nsample) （分组后邻域点的相对坐标）

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous() # (B, 3, N)->（B, N, 3）
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint)) # (B, 3, npoint)得到npoint个关键点坐标
    #球查询，得到每个关键点的nsample个邻域点
    idx = ball_query(radius, nsample, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous()) # (B, npoint, nsample)
    #分组操作，得到每个关键点的nsample个邻域点的相对坐标
    grouped_xyz = grouping_operation(xyz, idx) # (B, 3, npoint, nsample)
    #计算领域点的相对坐标
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, nsample)

    if points is not None:#如果有额外特征
        #分组操作，得到每个关键点的nsample个邻域点的特征
        grouped_points = grouping_operation(points, idx) # (B, f, npoint, nsample)
        #是否将坐标信息包含在特征中
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    #返回新的关键点坐标，新的特征，邻域点索引，分组后邻域点的相对坐标
    return new_xyz, new_points, idx, grouped_xyz

def sample_and_group_all(xyz, points, use_xyz=True):#对所有点进行分组，而不只是关键点   
    """
    Args:
        xyz: Tensor, (B, 3, nsample)批量大小，维度，点数
        points: Tensor, (B, f, nsample)输入点云的特征，维度f
        use_xyz: boolean是否使用坐标信息

    Returns:
        new_xyz: Tensor, (B, 3, 1) 全局的关键点坐标（所有点的中心）
        new_points: Tensor, (B, f|f+3|3, 1, nsample)分组后的全局特征
        idx: Tensor, (B, 1, nsample)所有点的索引
        grouped_xyz: Tensor, (B, 3, 1, nsample) 分组后的全局点的相对坐标
    """
    #获取批量大小和点数
    b, _, nsample = xyz.shape
    device = xyz.device
    #全局的关键点坐标（所有点的中心）初始化为0张量
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    #将所有点的坐标重排为（B, 3, 1, nsample）
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    #索引为0到nsample-1，即所有点的索引
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

class PointNet_SA_Module(nn.Module):
    def __init__(self, npoint, nsample, radius, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        #如果使用坐标信息，输入通道数加3
        if use_xyz:
            in_channel += 3
        #初始化最后一个通道数
        last_channel = in_channel
        #初始化卷积层
        self.mlp_conv = []
        #遍历mlp列表
        for out_channel in mlp:
            #添加卷积层
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points):
        """
        Args:
            xyz: Tensor, (B, 3, N)输入点云的坐标
            points: Tensor, (B, f, N)输入点云的特征

        Returns:
            new_xyz: Tensor, (B, 3, npoint) 返回的关键点坐标
            new_points: Tensor, (B, mlp[-1], npoint) 返回关键点的特征
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(xyz, points, self.npoint, self.nsample, self.radius, self.use_xyz)

        #使用MLP提取局部特征
        new_points = self.mlp_conv(new_points)
        #对特征进行最大池化得到全局特征
        new_points = torch.max(new_points, 3)[0]

        return new_xyz, new_points

class PointNet_FP_Module(nn.Module):
    def __init__(self, in_channel, mlp, use_points1=False, in_channel_points1=None, if_bn=True):
        """
        Args:
            in_channel: int, input channel of points2 输入的低分辨率点云的通道数
            mlp: list of int MLP的输出通道数列表
            use_points1: boolean, if use points 是否使用高分辨率点云的原始特征
            in_channel_points1: int, input channel of points1高分辨率点云的特征通道数
        """
        super(PointNet_FP_Module, self).__init__()
        self.use_points1 = use_points1
        #如果使用高分辨率点云的原始特征，输入通道数加上高分辨率点云的特征通道数
        if use_points1:
            in_channel += in_channel_points1

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: Tensor, (B, 3, N)高
            xyz2: Tensor, (B, 3, M)低
            points1: Tensor, (B, in_channel, N)高分辨率点云特征
            points2: Tensor, (B, in_channel, M)低分辨率点云特征

        Returns:MLP_CONV
            new_points: Tensor, (B, mlp[-1], N)高分辨率点云的新特征
        """
        #计算高分辨率点云中每个点到低分辨率点云中最近的 3 个点的距离和索引。
        dist, idx = three_nn(xyz1.permute(0, 2, 1).contiguous(), xyz2.permute(0, 2, 1).contiguous())
        #除数为0时，将其设置为一个很小的数
        dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
        #计算权重，权重为 1/d，然后归一化
        recip_dist = 1.0/dist
        norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
        weight = recip_dist / norm
        #根据权重对低分辨率点云的特征进行插值
        interpolated_points = three_interpolate(points2, idx, weight) # B, in_channel, N

        if self.use_points1:
            new_points = torch.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points

        new_points = self.mlp_conv(new_points)
        return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape#（B,N,C）
    _, M, _ = dst.shape# (B,M,C)
    #-2*src*dst^T
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    #B, S, N = xyz.shape 寻找new_xyz中每个点在xyz中的nsample个最近邻点
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :, pad: nsample+pad]
    return idx.int()

def nearest_distances(x, y):#计算两组点云中点x到另一组点云中最近的点的距离
    # x query, y target
    inner = -2 * torch.matmul(x.transpose(2, 1), y)  # x B 3 N; y B 3 M
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)

    pairwise_distance = xx.transpose(2, 1) + inner + yy
    nearest_distance = torch.sqrt(torch.min(pairwise_distance, dim=2, keepdim=True).values)

    return nearest_distance

def self_nearest_distances(x):#计算点云中每个点到其他点的最近距禨
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # x B 3 N
    xx = torch.sum(x ** 2, dim=1, keepdim=True)

    pairwise_distance = xx.transpose(2, 1) + inner + xx
    pairwise_distance += torch.eye(x.shape[2]).to(pairwise_distance.device) * 2
    nearest_distance = torch.sqrt(torch.min(pairwise_distance, dim=2, keepdim=True).values)

    return nearest_distance

def self_nearest_distances_K(x, k=3):#计算点云中每个点到其他最近k个点的平均距禨
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # x B 3 N
    xx = torch.sum(x ** 2, dim=1, keepdim=True)

    pairwise_distance = xx.transpose(2, 1) + inner + xx
    pairwise_distance += torch.eye(x.shape[2]).to(pairwise_distance.device) * 2
    pairwise_distance *= -1
    k_nearest_distance = pairwise_distance.topk(k=k, dim=2)[0]
    k_nearest_distance *= -1

    nearest_distance = torch.sqrt(torch.mean(k_nearest_distance, dim=2, keepdim=True))

    return nearest_distance

def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous() # (B, N, 3)
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint)) # (B, 3, npoint)
    if idx is None:
        idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx) # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points, idx) # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

class PointNet_SA_Module_KNN(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True, if_idx=False):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module_KNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx)

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points

def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), 
                               furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd

def get_nearest_index(target, source, k=1, return_dis=False):
    """
    Args:
        target: (bs, 3, v1)
        source: (bs, 3, v2)
    Return:
        nearest_index: (bs, v1, 1)
    """
    #计算目标点云和源点云之间的矩阵乘积（点积）
    inner = torch.bmm(target.transpose(1, 2), source)  # (bs, v1, v2)
    s_norm_2 = torch.sum(source**2, dim=1)  # (bs, v2)
    t_norm_2 = torch.sum(target**2, dim=1)  # (bs, v1)
    #计算欧几里得距离的平方
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(
        2) - 2 * inner  # (bs, v1, v2)
    #最近的k个点的索引和对应距离
    nearest_dis, nearest_index = torch.topk(d_norm_2,
                                            k=k,
                                            dim=-1,
                                            largest=False)
    if not return_dis:
        return nearest_index
    else:
        return nearest_index, nearest_dis

def indexing_neighbor(x, index):#根据索引获取邻域点的特征
    """
    Args:
        x: (bs, dim, num_points0)（批量大小， 特征维度， 点的数量）
        index: (bs, num_points, k)（批量大小， 查询点的数量， 查询点的临近数量）
    Return:
        feature: (bs, dim, num_points, k)
    """
    batch_size, num_points, k = index.size()

    id_0 = torch.arange(batch_size).view(-1, 1, 1)

    x = x.transpose(2, 1).contiguous()  # (bs, num_points, num_dims)
    feature = x[id_0, index]  # (bs, num_points, k, num_dims)
    feature = feature.permute(0, 3, 1,
                              2).contiguous()  # (bs, num_dims, num_points, k)

    return feature

class cross_attention(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, src2, pos=None):
        # pdb.set_trace()
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)
        b, c, _ = src1.shape
        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)
        src1 = self.norm13(src1)
        src2 = self.norm13(src2)
        q  = self.with_pos_embed(src1, pos)
        src12 = self.multihead_attn(query=q,
                                     key=src2,
                                     value=src2)[0]
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = src1.permute(1, 2, 0)

        return src1

class self_attention(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)
        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.activation1 = torch.nn.GELU()
        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, pos=None):
        src1 = self.input_proj(src1)
        b, c, _ = src1.shape
        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src1 = self.norm13(src1)
        q=k=self.with_pos_embed(src1,pos)
        src12 = self.multihead_attn(query=q,
                                     key=k,
                                     value=src1)[0]
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = src1.permute(1, 2, 0)

        return src1

class SDG_Decoder(nn.Module):

    def __init__(self, hidden_dim,channel,ratio):
        super().__init__()
        self.sa1 = self_attention(hidden_dim, hidden_dim, dropout=0.0, nhead=8)
        self.sa2 = self_attention(hidden_dim, channel * ratio, dropout=0.0, nhead=8)

    def forward(self, input):
        f = self.sa1(input)
        f = self.sa2(f)
        return f

class PointNetFeatureExtractor(nn.Module):
    r"""PointNet feature extractor (extracts either global or local, i.e.,
    per-point features).

    Based on the original PointNet paper:.

    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::

            @article{qi2016pointnet,
              title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
              author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
              journal={arXiv preprint arXiv:1612.00593},
              year={2016}
            }

    Args:
        in_channels (int): Number of channels in the input pointcloud
            (default: 3, for X, Y, Z coordinates respectively).
        feat_size (int): Size of the global feature vector
            (default: 1024)
        layer_dims (Iterable[int]): Sizes of fully connected layers
            to be used in the feature extractor (excluding the input and
            the output layer sizes). Note: the number of
            layers in the feature extractor is implicitly parsed from
            this variable.
        global_feat (bool): Extract global features (i.e., one feature
            for the entire pointcloud) if set to True. If set to False,
            extract per-point (local) features (default: True).
        activation (function): Nonlinearity to be used as activation
                    function after each batchnorm (default: F.relu)
        batchnorm (bool): Whether or not to use batchnorm layers
            (default: True)
        transposed_input (bool): Whether the input's second and third dimension
            is already transposed. If so, a transpose operation can be avoided,
            improving performance.
            See documentation for the forward method for more details.

    For example, to specify a PointNet feature extractor with 4 linear
    layers (sizes 6 -> 10, 10 -> 40, 40 -> 500, 500 -> 1024), with
    3 input channels in the pointcloud and a global feature vector of size
    1024, see the example below.

    Example:

        >>> pointnet = PointNetFeatureExtractor(in_channels=3, feat_size=1024,
                                           layer_dims=[10, 20, 40, 500])
        >>> x = torch.rand(2, 3, 30)
        >>> y = pointnet(x)
        print(y.shape)

    """

    def __init__(self,
                 in_channels: int = 3,
                 feat_size: int = 1024,
                 layer_dims: Iterable[int] = [64, 128],
                 global_feat: bool = True,
                 activation=F.relu,
                 batchnorm: bool = True,
                 transposed_input: bool = False):
        super(PointNetFeatureExtractor, self).__init__()

        if not isinstance(in_channels, int):
            raise TypeError('Argument in_channels expected to be of type int. '
                            'Got {0} instead.'.format(type(in_channels)))
        if not isinstance(feat_size, int):
            raise TypeError('Argument feat_size expected to be of type int. '
                            'Got {0} instead.'.format(type(feat_size)))
        if not hasattr(layer_dims, '__iter__'):
            raise TypeError('Argument layer_dims is not iterable.')
        for idx, layer_dim in enumerate(layer_dims):
            if not isinstance(layer_dim, int):
                raise TypeError('Elements of layer_dims must be of type int. '
                                'Found type {0} at index {1}.'.format(
                                    type(layer_dim), idx))
        if not isinstance(global_feat, bool):
            raise TypeError('Argument global_feat expected to be of type '
                            'bool. Got {0} instead.'.format(
                                type(global_feat)))

        # Store feat_size as a class attribute
        self.feat_size = feat_size

        # Store activation as a class attribute
        self.activation = activation

        # Store global_feat as a class attribute
        self.global_feat = global_feat

        # Add in_channels to the head of layer_dims (the first layer
        # has number of channels equal to `in_channels`). Also, add
        # feat_size to the tail of layer_dims.
        if not isinstance(layer_dims, list):
            layer_dims = list(layer_dims)
        layer_dims.insert(0, in_channels)
        layer_dims.append(feat_size)

        self.conv_layers = nn.ModuleList()
        if batchnorm:
            self.bn_layers = nn.ModuleList()
        for idx in range(len(layer_dims) - 1):
            self.conv_layers.append(nn.Conv1d(layer_dims[idx],
                                              layer_dims[idx + 1], 1))
            if batchnorm:
                self.bn_layers.append(nn.BatchNorm1d(layer_dims[idx + 1]))

        # Store whether or not to use batchnorm as a class attribute
        self.batchnorm = batchnorm

        self.transposed_input = transposed_input

    def forward(self, x: torch.Tensor):
        r"""Forward pass through the PointNet feature extractor.

        Args:
            x (torch.Tensor): Tensor representing a pointcloud
                (shape: :math:`B \times N \times D`, where :math:`B`
                is the batchsize, :math:`N` is the number of points
                in the pointcloud, and :math:`D` is the dimensionality
                of each point in the pointcloud).
                If self.transposed_input is True, then the shape is
                :math:`B \times D \times N`.

        """
        if not self.transposed_input:
            x = x.transpose(1, 2)

        # Number of points
        num_points = x.shape[2]

        # By default, initialize local features (per-point features)
        # to None.
        local_features = None

        # Apply a sequence of conv-batchnorm-nonlinearity operations

        # For the first layer, store the features, as these will be
        # used to compute local features (if specified).
        if self.batchnorm:
            x = self.activation(self.bn_layers[0](self.conv_layers[0](x)))
        else:
            x = self.activation(self.conv_layers[0](x))
        if self.global_feat is False:
            local_features = x

        # Pass through the remaining layers (until the penultimate layer).
        for idx in range(1, len(self.conv_layers) - 1):
            if self.batchnorm:
                x = self.activation(self.bn_layers[idx](
                    self.conv_layers[idx](x)))
            else:
                x = self.activation(self.conv_layers[idx](x))

        # For the last layer, do not apply nonlinearity.
        if self.batchnorm:
            x = self.bn_layers[-1](self.conv_layers[-1](x))
        else:
            x = self.conv_layers[-1](x)

        # Max pooling.
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.feat_size)

        # If extracting global features, return at this point.
        if self.global_feat:
            return x

        # If extracting local features, compute local features by
        # concatenating global features, and per-point features
        x = x.view(-1, self.feat_size, 1).repeat(1, 1, num_points)
        return torch.cat((x, local_features), dim=1)

def query_knn_point(k, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    _, group_idx = dist.topk(k, largest=False)
    return group_idx

def group_local(xyz, k=20, return_idx=False):
    """
    Input:
        x: point cloud, [B, 3, N]
    Return:
        group_xyz: [B, 3, N, K]
    """
    xyz = xyz.transpose(2, 1).contiguous()
    idx = query_knn_point(k, xyz, xyz)
    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz.permute(0, 3, 1, 2)
    if return_idx:
        return group_xyz, idx

    return group_xyz

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]C-通道数
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class EdgeConv(torch.nn.Module):
    """
    Input:
        x: point cloud, [B, C1, N]
    Return:
        x: point cloud, [B, C2, N]
    """

    def __init__(self, input_channel, output_channel, k):
        super(EdgeConv, self).__init__()
        self.num_neigh = k#邻域点的数量

        #定义卷积层，输入通道数为2倍的输入通道数，输出通道数为输出通道数的一半，处理边特征
        self.conv = nn.Sequential(
            nn.Conv2d(2 * input_channel, output_channel // 2, kernel_size=1),
            nn.BatchNorm2d(output_channel // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(output_channel // 2, output_channel // 2, kernel_size=1),
            nn.BatchNorm2d(output_channel // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(output_channel // 2, output_channel, kernel_size=1)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: 点云特征，形状为 [B, C1, N]
        Returns:
            central_feature: 点云特征，形状为 [B, C2, N]
        """
        batch_size, dims, num_points = inputs.shape
        if self.num_neigh is not None:
            neigh_feature = group_local(inputs, k=self.num_neigh).contiguous()
            central_feat = inputs.unsqueeze(dim=3).repeat(1, 1, 1, self.num_neigh)
        else:
            central_feat = torch.zeros(batch_size, dims, num_points, 1).to(inputs.device)
            neigh_feature = inputs.unsqueeze(-1)
        edge_feature = central_feat - neigh_feature
        feature = torch.cat((edge_feature, central_feat), dim=1)
        feature = self.conv(feature)
        central_feature = feature.max(dim=-1, keepdim=False)[0]
        return central_feature

class SinusoidalPositionalEmbedding(nn.Module):#正弦位置编码
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:#位置编码的维度必须是偶数
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        #计算位置编码
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)输入的位置索引，形状任意

        Returns:
            embeddings: torch.Tensor (*, D)位置编码，形状为（*，d_modeld_model）
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    Args:
        x: Tensor, 输入点云特征，形状为 (B, C, N)，其中：
            - B 是批量大小。
            - C 是特征维度。
            - N 是点的数量。
        k: int, 每个点的邻域点数量。
        idx: Tensor, 邻域点的索引，形状为 (B, N, k)，可选。

    Returns:
        feature: Tensor, 图特征，形状为 (B, 2*C, N, k)。
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    #为每个批次的索引添加偏移量
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    #B,N,K
    idx = idx + idx_base

    #将索引展平
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    #将中心特征扩展到其领域
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # 计算边特征（邻域点特征 - 中心点特征）
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp.squeeze()

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """
    """
    Convert Euler angles to rotation matrix.

    Args:
        angle: Tensor, 输入的欧拉角，形状为 [3] 或 [B, 3]。
            - [3]: 单个角度向量，表示绕 x、y、z 轴的旋转角度。
            - [B, 3]: 批量角度向量，B 表示批量大小。

    Returns:
        rot_mat: Tensor, 输出的旋转矩阵，形状为 [3, 3] 或 [B, 3, 3]。
    """
    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    # zero = torch.zeros([b], requires_grad=False, device=angle.device)[0]
    # one = torch.ones([b], requires_grad=False, device=angle.device)[0]
    zero = z.detach()*0
    one = zero.detach()+1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    # print(rot_mat)
    return rot_mat


def distribute(depth, _x, _y, size_x, size_y, image_height, image_width):
    """
    Distributes the depth associated with each point to the discrete coordinates (image_height, image_width) in a region
    of size (size_x, size_y).
    :param depth:
    :param _x:
    :param _y:
    :param size_x:
    :param size_y:
    :param image_height:
    :param image_width:
    :return:
    """
    """
    Distributes the depth associated with each point to the discrete coordinates (image_height, image_width) in a region
    of size (size_x, size_y).

    Args:
        depth: Tensor, 点云的深度值，形状为 [batch, num_points]。
        _x: Tensor, 点云的 x 坐标，形状为 [batch, num_points]。
        _y: Tensor, 点云的 y 坐标，形状为 [batch, num_points]。
        size_x: int, x 方向的分布区域大小。
        size_y: int, y 方向的分布区域大小。
        image_height: int, 图像的高度。
        image_width: int, 图像的宽度。

    Returns:
        weighed_value_scattered: Tensor, 加权分布的深度值，形状为 [batch, image_height * image_width]。
        weight_scattered: Tensor, 权重分布，形状为 [batch, image_height * image_width]。
    """
    # 确保分布区域大小为偶数或 1
    assert size_x % 2 == 0 or size_x == 1
    assert size_y % 2 == 0 or size_y == 1
    batch, _ = depth.size()
    epsilon = torch.tensor([1e-12], requires_grad=False, device=depth.device)
    # 生成分布区域的偏移量
    _i = torch.linspace(-size_x / 2, (size_x / 2) - 1, size_x, requires_grad=False, device=depth.device)
    _j = torch.linspace(-size_y / 2, (size_y / 2) - 1, size_y, requires_grad=False, device=depth.device)

    # 扩展点的 x 和 y 坐标到分布区域
    extended_x = _x.unsqueeze(2).repeat([1, 1, size_x]) + _i  # [batch, num_points, size_x]
    extended_y = _y.unsqueeze(2).repeat([1, 1, size_y]) + _j  # [batch, num_points, size_y]

    extended_x = extended_x.unsqueeze(3).repeat([1, 1, 1, size_y])  # [batch, num_points, size_x, size_y]
    extended_y = extended_y.unsqueeze(2).repeat([1, 1, size_x, 1])  # [batch, num_points, size_x, size_y]
    # 将坐标取整，映射到离散像素网格
    extended_x.ceil_()
    extended_y.ceil_()
    # 扩展深度值到分布区域
    value = depth.unsqueeze(2).unsqueeze(3).repeat([1, 1, size_x, size_y])  # [batch, num_points, size_x, size_y]

    # all points that will be finally used生成遮罩，过滤掉超出图像边界的点
    masked_points = ((extended_x >= 0)
                     * (extended_x <= image_height - 1)
                     * (extended_y >= 0)
                     * (extended_y <= image_width - 1)
                     * (value >= 0))

    true_extended_x = extended_x
    true_extended_y = extended_y

    # to prevent error
    extended_x = (extended_x % image_height)
    extended_y = (extended_y % image_width)

    # [batch, num_points, size_x, size_y]
    distance = torch.abs((extended_x - _x.unsqueeze(2).unsqueeze(3))
                         * (extended_y - _y.unsqueeze(2).unsqueeze(3)))
    weight = (masked_points.float()
          * (1 / (value + epsilon)))  # [batch, num_points, size_x, size_y]
    weighted_value = value * weight

    weight = weight.view([batch, -1])
    weighted_value = weighted_value.view([batch, -1])

    coordinates = (extended_x.view([batch, -1]) * image_width) + extended_y.view(
        [batch, -1])
    coord_max = image_height * image_width
    true_coordinates = (true_extended_x.view([batch, -1]) * image_width) + true_extended_y.view(
        [batch, -1])
    true_coordinates[~masked_points.view([batch, -1])] = coord_max
    weight_scattered = torch.zeros(
        [batch, image_width * image_height],
        device=depth.device).scatter_add(1, coordinates.long(), weight)

    masked_zero_weight_scattered = (weight_scattered == 0.0)
    weight_scattered += masked_zero_weight_scattered.float()

    weighed_value_scattered = torch.zeros(
        [batch, image_width * image_height],
        device=depth.device).scatter_add(1, coordinates.long(), weighted_value)

    return weighed_value_scattered,  weight_scattered


def points2depth(points, image_height, image_width, size_x=4, size_y=4):
    """
    :param points: [B, num_points, 3]
    :param image_width:
    :param image_height:
    :param size_x:
    :param size_y:
    :return:
        depth_recovered: [B, image_width, image_height]
    """

    epsilon = torch.tensor([1e-12], requires_grad=False, device=points.device)
    # epsilon not needed, kept here to ensure exact replication of old version
    coord_x = (points[:, :, 0] / (points[:, :, 2] + epsilon)) * (image_width / image_height)  # [batch, num_points]
    coord_y = (points[:, :, 1] / (points[:, :, 2] + epsilon))  # [batch, num_points]

    batch, total_points, _ = points.size()
    depth = points[:, :, 2]  # [batch, num_points]
    # pdb.set_trace()
    _x = ((coord_x + 1) * image_height) / 2
    _y = ((coord_y + 1) * image_width) / 2

    weighed_value_scattered, weight_scattered = distribute(
        depth=depth,
        _x=_x,
        _y=_y,
        size_x=size_x,
        size_y=size_y,
        image_height=image_height,
        image_width=image_width)

    depth_recovered = (weighed_value_scattered / weight_scattered).view([
        batch, image_height, image_width
    ])

    return depth_recovered


# source: https://discuss.pytorch.org/t/batched-index-select/9115/6
def batched_index_select(inp, dim, index):
    """
    input: B x * x ... x *
    dim: 0 < scalar
    index: B x M
    """
    views = [inp.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(inp.shape))]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


def point_fea_img_fea(point_fea, point_coo, h, w):
    """
    each point_coo is of the form (x*w + h). points not in the canvas are removed
    :param point_fea: [batch_size, num_points, feat_size]
    :param point_coo: [batch_size, num_points]
    :return:
    """
    assert len(point_fea.shape) == 3
    assert len(point_coo.shape) == 2
    assert point_fea.shape[0:2] == point_coo.shape

    coo_max = ((h - 1) * w) + (w - 1)
    mask_point_coo = (point_coo >= 0) * (point_coo <= coo_max)
    point_coo *= mask_point_coo.float()
    point_fea *= mask_point_coo.float().unsqueeze(-1)

    bs, _, fs = point_fea.shape
    point_coo = point_coo.unsqueeze(2).repeat([1, 1, fs])
    img_fea = torch.zeros([bs, h * w, fs], device=point_fea.device).scatter_add(1, point_coo.long(), point_fea)

    return img_fea


def distribute_img_fea_points(img_fea, point_coord):
    """
    :param img_fea: [B, C, H, W]
    :param point_coord: [B, num_points], each coordinate  is a scalar value given by (x * W) + y
    :return
        point_fea: [B, num_points, C], for points with coordinates outside the image, we return 0
    """
    B, C, H, W = list(img_fea.size())
    img_fea = img_fea.permute(0, 2, 3, 1).view([B, H*W, C])

    coord_max = ((H - 1) * W) + (W - 1)
    mask_point_coord = (point_coord >= 0) * (point_coord <= coord_max)
    mask_point_coord = mask_point_coord.float()
    point_coord = mask_point_coord * point_coord
    point_fea = batched_index_select(
        inp=img_fea,
        dim=1,
        index=point_coord.long())
    point_fea = mask_point_coord.unsqueeze(-1) * point_fea
    return point_fea


class PCViews:
    """For creating images from PC based on the view information. Faster as the
    repeated operations are done only once whie initialization.
    """

    def __init__(self,TRANS,RESOLUTION):
        _views = np.asarray([
            [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[1 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
            [[0, -np.pi / 2, np.pi / 2], [0, 0, TRANS]]])
        self.num_views = 3
        angle = torch.tensor(_views[:, 0, :]).float().cuda()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        self.translation = torch.tensor(_views[:, 1, :]).float().cuda()
        self.translation = self.translation.unsqueeze(1)
        self.resolution = RESOLUTION

    def get_img(self, points):
        """Get image based on the prespecified specifications.

        Args:
            points (torch.tensor): of size [B, _, 3]
        Returns:
            img (torch.tensor): of size [B * self.num_views, RESOLUTION,
                RESOLUTION]
        """
        b, _, _ = points.shape
        v = self.translation.shape[0]

        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))

        img = points2depth(
            points=_points,
            image_height=self.resolution,
            image_width=self.resolution,
            size_x=1,
            size_y=1,
        )
        return img

    @staticmethod
    def point_transform(points, rot_mat, translation):
        """
        :param points: [batch, num_points, 3]
        :param rot_mat: [batch, 3]
        :param translation: [batch, 1, 3]
        :return:
        """
        rot_mat = rot_mat.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = points - translation
        return points


















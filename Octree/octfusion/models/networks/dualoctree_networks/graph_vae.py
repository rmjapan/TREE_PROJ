# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn
from torch.nn import init

import sys


from Octree.octfusion.models.networks import modules
sys.path.append('/home/ryuichi/tree/Other/related_work/octfusion/models/networks')
from dualoctree_networks.distributions import DiagonalGaussianDistribution
# import modules の代わりに絶対パスでインポート


import Octree.octfusion.models.networks.dualoctree_networks.mpu as mpu
from ocnn.nn import octree2voxel
from ocnn.octree import Octree, Points

def init_weights(net, init_type='normal', gain=0.01):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

    # propagate to children
    for m in net.children():
        m.apply(init_func)

class GraphVAE(torch.nn.Module):

    def __init__(self, depth, channel_in, nout, full_depth=2, depth_stop = 6, depth_out=8, use_checkpoint = False, resblk_type='bottleneck', bottleneck=4,resblk_num=3, code_channel=3, embed_dim=3):
        # super().__init__(depth, channel_in, nout, full_depth, depth_stop, depth_out, use_checkpoint, resblk_type, bottleneck,resblk_num)
        # this is to make the encoder and decoder symmetric

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.channel_in = channel_in
        self.nout = nout
        self.full_depth = full_depth
        self.depth_stop = depth_stop
        self.depth_out = depth_out
        self.use_checkpoint = use_checkpoint
        self.resblk_type = resblk_type
        self.bottleneck = bottleneck
        self.resblk_num = resblk_num
        self.neural_mpu = mpu.NeuralMPU(self.full_depth, self.depth_stop, self.depth_out)
        self._setup_channels_and_resblks()
        n_edge_type, avg_degree = 7, 7
        self.dropout = 0.0
        n_edge_type, avg_degree = 7, 7

        # encoder
        self.conv1 = modules.GraphConv(
            channel_in, self.channels[depth], n_edge_type, avg_degree, depth-1)
        self.encoder = torch.nn.ModuleList(
            [modules.GraphResBlocks(self.channels[d], self.channels[d],self.dropout,
            self.resblk_nums[d] - 1, n_edge_type, avg_degree, d-1, self.use_checkpoint)
            for d in range(depth, depth_stop-1, -1)])
        self.downsample = torch.nn.ModuleList(
            [modules.GraphDownsample(self.channels[d], self.channels[d-1], n_edge_type, avg_degree, depth-1)
            for d in range(depth, depth_stop, -1)])

        self.encoder_norm_out = modules.DualOctreeGroupNorm(self.channels[depth_stop])

        self.nonlinearity = torch.nn.GELU()
        
        # decoder
        self.decoder = torch.nn.ModuleList(
            [modules.GraphResBlocks(self.channels[d], self.channels[d],self.dropout,
         self.resblk_nums[d], n_edge_type, avg_degree, d-1, self.use_checkpoint)
            for d in range(depth_stop, depth + 1)])
        self.decoder_mid = torch.nn.Module()
        self.decoder_mid.block_1 = modules.GraphResBlocks(self.channels[depth_stop], self.channels[depth_stop],self.dropout,
         self.resblk_nums[depth_stop], n_edge_type, avg_degree, depth_stop-1, self.use_checkpoint)
        self.decoder_mid.block_2 = modules.GraphResBlocks(self.channels[depth_stop], self.channels[depth_stop],self.dropout,
         self.resblk_nums[depth_stop], n_edge_type, avg_degree, depth_stop-1, self.use_checkpoint)
        
        self.upsample = torch.nn.ModuleList(
            [modules.GraphUpsample(self.channels[d-1], self.channels[d], n_edge_type, avg_degree, depth-1)
            for d in range(depth_stop + 1, depth + 1)])

        # header
        self.predict = torch.nn.ModuleList(
            [self._make_predict_module(self.channels[d], 2)  # 这里的2就是当前节点是否要劈成八份的label
            for d in range(depth_stop, depth + 1)])
        self.regress = torch.nn.ModuleList(
            [self._make_predict_module(self.channels[d], 4)  # ここでの4というのは王先生が言及されたように、MPUの各ノードが持つ4つの特徴値は、法線ベクトル（3次元）とオフセット値（1次元）を表しています（Sdf値だね.)
            for d in range(depth_stop, depth + 1)])
        

        self.code_channel = code_channel
        ae_channel_in = self.channels[self.depth_stop]
        self.KL_conv = modules.Conv1x1(ae_channel_in, 2 * embed_dim, use_bias = True)
        self.post_KL_conv = modules.Conv1x1(embed_dim, ae_channel_in, use_bias = True)

    def _setup_channels_and_resblks(self):
        # self.resblk_num = [3] * 7 + [1] + [1] * 9
        # self.resblk_num = [3] * 16
        self.resblk_nums = [self.resblk_num] * 16      # resblk_num[d] 为深度d（分辨率）下resblock的数量。
        # self.channels = [4, 512, 512, 256, 128, 64, 32, 32, 24, 16, 8]  # depth i的channel为channels[i]
        self.channels = [4, 512, 512, 256, 128, 64, 32, 32, 24, 8]

    def _make_predict_module(self, channel_in, channel_out=2, num_hidden=32):
        return torch.nn.Sequential(
        modules.Conv1x1GnGeluSequential(channel_in, num_hidden),
        modules.Conv1x1(num_hidden, channel_out, use_bias=True))

    def _get_input_feature(self, doctree):
        return doctree.get_input_feature()

    def octree_encoder_step(self, octree, doctree):
        depth, depth_stop = self.depth, self.depth_stop
        data = self._get_input_feature(doctree)

        convs = dict()
        convs[depth] = data   # channel为4
        for i, d in enumerate(range(depth, depth_stop-1, -1)):   # encoder的操作是从depth到depth_stop为止
        # perform graph conv
            convd = convs[d]  # get convd
            if d == self.depth:  # the first conv
                convd = self.conv1(convd, doctree, d)
            convd = self.encoder[i](convd, doctree, d)
            convs[d] = convd  # update convd
            # print(convd.shape)

        # downsampleing
            if d > depth_stop:  # init convd
                nnum = doctree.nnum[d]
                lnum = doctree.lnum[d-1]
                leaf_mask = doctree.node_child(d-1) < 0
                convs[d-1] = self.downsample[i](convd, doctree, d-1, leaf_mask, nnum, lnum)

        convs[depth_stop] = self.encoder_norm_out(convs[depth_stop], doctree, depth_stop)
        convs[depth_stop] = self.nonlinearity(convs[depth_stop])

        return convs
    
    def octree_encoder(self, octree, doctree): # encoder的操作是从depth到full-deth为止，在这里就是从6到2
        convs = self.octree_encoder_step(octree, doctree) # conv的channel随着八叉树深度从6到2的变化为[32, 64, 128, 256, 512]
        # reduce the dimension
        code = self.KL_conv(convs[self.depth_stop])
        # print(code.max())
        # print(code.min())
        posterior = DiagonalGaussianDistribution(code)
        return posterior

    def octree_decoder(self, code, doctree_out, update_octree=False):
        #quant code [bs, 3, 16, 16, 16]
        code = self.post_KL_conv(code)   # [bs, code_channel, 16, 16, 16]
        octree_out = doctree_out.octree

        logits = dict()
        reg_voxs = dict()
        deconvs = dict()

        depth_stop = self.depth_stop

        deconvs[depth_stop] = code

        deconvs[depth_stop] = self.decoder_mid.block_1(deconvs[depth_stop], doctree_out, depth_stop)
        deconvs[depth_stop] = self.decoder_mid.block_2(deconvs[depth_stop], doctree_out, depth_stop)

        for i, d in enumerate(range(self.depth_stop, self.depth_out+1)): # decoder的操作是从full_depth到depth_out为止
            if d > self.depth_stop:
                nnum = doctree_out.nnum[d-1]
                leaf_mask = doctree_out.node_child(d-1) < 0
                deconvs[d] = self.upsample[i-1](deconvs[d-1], doctree_out, d, leaf_mask, nnum)

            octree_out = doctree_out.octree
            deconvs[d] = self.decoder[i](deconvs[d], doctree_out, d)

            # predict the splitting label
            logit = self.predict[i]([deconvs[d], doctree_out, d])
            nnum = doctree_out.nnum[d]
            logits[d] = logit[-nnum:]

            # update the octree according to predicted labels
            if update_octree:   # 测试阶段：如果update_octree为true，则从full_depth开始逐渐增长八叉树，直至depth_out
                label = logits[d].argmax(1).to(torch.int32)
                octree_out = doctree_out.octree
                octree_out.octree_split(label, d)
                if d < self.depth_out:
                    octree_out.octree_grow(d + 1)  # 对初始化的满八叉树，根据预测的标签向上增长至depth_out
                    octree_out.depth += 1
                doctree_out = DualOctree(octree_out)
                doctree_out.post_processing_for_docnn()

            # predict the signal
            reg_vox = self.regress[i]([deconvs[d], doctree_out, d])

            # TODO: improve it
            # pad zeros to reg_vox to reuse the original code for ocnn
            node_mask = doctree_out.graph[d]['node_mask']
            shape = (node_mask.shape[0], reg_vox.shape[1])
            reg_vox_pad = torch.zeros(shape, device=reg_vox.device)
            reg_vox_pad[node_mask] = reg_vox
            reg_voxs[d] = reg_vox_pad

        return logits, reg_voxs, doctree_out.octree

    def create_full_octree(self, octree_in: Octree):
        r''' Initialize a full octree for decoding.
        '''

        device = octree_in.device
        batch_size = octree_in.batch_size
        octree = Octree(self.depth, self.full_depth, batch_size, device)
        for d in range(self.full_depth+1):
            octree.octree_grow_full(depth=d)
        return octree

    def create_child_octree(self, octree_in: Octree):
        octree_out = self.create_full_octree(octree_in)
        octree_out.depth = self.full_depth
        for d in range(self.full_depth, self.depth_stop):
            label = octree_in.nempty_mask(d).long()
            octree_out.octree_split(label, d)
            octree_out.octree_grow(d + 1)
            octree_out.depth += 1
        return octree_out

    def forward(self, octree_in, octree_out=None, pos=None, evaluate=False): # ここでのposのサイズは[batch_size * 5000, 4]で、すべてのバッチのポイントが連結されており、4番目の次元でバッチインデックスを表しています
        # デュアル八分木を生成する
        doctree_in = DualOctree(octree=octree_in)
        doctree_in.post_processing_for_docnn()

        # octree_outが提供されていない場合、最初から作成する
        update_octree = octree_out is None
        if update_octree:
            # 完全な八分木を作成し、目的の深さまで成長させる
            octree_out = self.create_full_octree(octree_in)
            octree_out.depth = self.full_depth
            for d in range(self.full_depth, self.depth_stop):
                # 入力八分木の非空ノードを分割ガイドとして使用
                label = octree_in.nempty_mask(d).long()
                octree_out.octree_split(label, d)
                octree_out.octree_grow(d + 1)
                octree_out.depth += 1

        # 出力八分木からデュアル八分木を作成
        doctree_out = DualOctree(octree_out)
        doctree_out.post_processing_for_docnn()

        # 入力八分木をエンコードして潜在分布を取得
        posterior = self.octree_encoder(octree_in, doctree_in)
        # 事後分布からサンプリング
        z = posterior.sample()

        # 評価中に潜在コードの統計を表示
        if evaluate:
            z = posterior.sample()
            print(z.max(), z.min(), z.mean(), z.std())

        # 潜在コードをデコードして八分木を再構築
        out = self.octree_decoder(z, doctree_out, update_octree)
        
        # 出力辞書を準備
        output = {'logits': out[0], 'reg_voxs': out[1], 'octree_out': out[2]}
        
        # KLダイバージェンス損失を計算
        kl_loss = posterior.kl()
        output['kl_loss'] = kl_loss.mean()
        output['code_max'] = z.max()
        output['code_min'] = z.min()

        # 点の位置が提供されている場合、MPUを使用して関数値を計算
        if pos is not None:
            output['mpus'] = self.neural_mpu(pos, out[1], out[2])

        # 任意の点のSDF値を取得するための関数を作成
        def _neural_mpu(pos):
            pred = self.neural_mpu(pos, out[1], out[2])
            return pred[self.depth_stop][0]
        # この neural_mpu 関数は主にテスト段階で使用され、最後の層の reg_voxs に基づいて
        # 任意の入力位置に対するSDF値を返します
        output['neural_mpu'] = _neural_mpu

        return output

    def extract_code(self, octree_in):
        doctree_in = DualOctree(octree_in)
        doctree_in.post_processing_for_docnn()

        convs = self.octree_encoder_step(octree_in, doctree_in) # conv的channel随着八叉树深度从6到2的变化为[32, 64, 128, 256, 512]
        code = self.KL_conv(convs[self.depth_stop])
        posterior = DiagonalGaussianDistribution(code)
        return posterior.sample(), doctree_in

    def decode_code(self, code, doctree_in, update_octree = True, pos = None):

        octree_in = doctree_in.octree
        # generate dual octrees
        if update_octree:
            octree_out = self.create_child_octree(octree_in)
            doctree_out = DualOctree(octree_out)
            doctree_out.post_processing_for_docnn()
        else:
            doctree_out = doctree_in

        # run decoder
        out = self.octree_decoder(code, doctree_out, update_octree=update_octree)
        output = {'logits': out[0], 'reg_voxs': out[1], 'octree_out': out[2]}

        if pos is not None:
            output['mpus'] = self.neural_mpu(pos, out[1], out[2])

        # create the mpu wrapper
        def _neural_mpu(pos):
            pred = self.neural_mpu(pos, out[1], out[2])
            return pred[self.depth_out][0]
        output['neural_mpu'] = _neural_mpu

        return output

def visualize_model(model, log_dir=None, sample_input=True):
    """Visualize the GraphVAE model structure using TensorBoard.
    
    Args:
        model: GraphVAE instance
        log_dir: TensorBoard log directory (if None, will create one in runs/graph_vae_[timestamp])
        sample_input: Whether to add a sample input to visualize the model graph
    """
    import torch
    from torch.utils.tensorboard import SummaryWriter
    import os
    import datetime
    
    if log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join('runs', f'graph_vae_{timestamp}')
    
    writer = SummaryWriter(log_dir)
    
    # Add model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f'parameters/{name}', param, global_step=0)
    
    # Add model layer information
    writer.add_text('model/architecture', str(model), global_step=0)
    writer.add_text('model/depth', f"depth: {model.depth}, full_depth: {model.full_depth}, depth_stop: {model.depth_stop}, depth_out: {model.depth_out}", global_step=0)
    
    # Add hyperparameters
    hparams = {
        'depth': model.depth,
        'full_depth': model.full_depth,
        'depth_stop': model.depth_stop,
        'depth_out': model.depth_out,
        'channel_in': model.channel_in,
        'nout': model.nout,
        'bottleneck': model.bottleneck,
        'code_channel': model.code_channel,
        'embed_dim': model.embed_dim
    }
    writer.add_hparams(hparams, {'hparam/model_initialized': 1})
    
    # Add model graph with a sample input if requested
    if sample_input:
        try:
            device = next(model.parameters()).device
            
            # Create a simple octree for visualization
            from ocnn.octree import Octree, Points
            import numpy as np
            
            # Create a simple sphere point cloud
            n_points = 1000
            points = []
            normals = []
            
            for _ in range(n_points):
                # Sample points on a sphere
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                
                # Convert to Cartesian coordinates
                x = 0.5 * np.sin(phi) * np.cos(theta)
                y = 0.5 * np.sin(phi) * np.sin(theta)
                z = 0.5 * np.cos(phi)
                
                points.append([x, y, z])
                # Normals point outward
                normals.append([x, y, z])
            
            # Create a Points object
            points_tensor = torch.tensor(points, dtype=torch.float32)
            normals_tensor = torch.tensor(normals, dtype=torch.float32)
            points_obj = Points(points=points_tensor, normals=normals_tensor)
            
            # Create an octree from the points
            octree = Octree(model.depth, model.full_depth)
            octree.build_octree(points_obj)
            octree = octree.to(device)
            
            # Create sample points for position input
            sample_pos = torch.rand(1000, 4, device=device)  # [N, 4] format with batch index
            
            # Generate a graph using torch.utils.tensorboard.SummaryWriter
            writer.add_graph(model, (octree, None, sample_pos))
            
            writer.add_text('model/input_example', 
                           f"Created a sample input with an octree of depth {model.depth} built from a sphere with {n_points} points.", 
                           global_step=0)
            
        except Exception as e:
            writer.add_text('model/graph_error', f"Failed to add model graph: {str(e)}", global_step=0)
            print(f"Warning: Failed to add model graph: {e}")
    
    writer.close()
    print(f"Model visualization saved to {log_dir}")
    print(f"View with: tensorboard --logdir={log_dir}")
    
    return log_dir

# model = GraphVAE(depth=6, channel_in=4, nout=4, full_depth=2, depth_stop=4, depth_out=6, code_channel=16, embed_dim=3)
# visualize_model(model,sample_input=True)
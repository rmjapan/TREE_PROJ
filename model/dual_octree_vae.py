
import torch.nn as nn
import sys 
sys.path.append("..")
import torch
from model.networks import modules
from model.networks.distributions import DiagonalGaussianDistribution


class DualOctreeVAE(nn.Module):
    def __init__(
        self,
        Encoder_input_depth=8,#対象とするOctreeの深さ（8)
        Encoder_output_depth=6,#Encoderの出力深度
        Decoder_input_depth=6,#Decoderの入力深度
        Decoder_output_depth=8,#Decoderの出力深度
        Encoder_in_channels=4,#入力特徴ベクトルのチャネル数
        Dual_Octree_full_depth=4,#Dual Octreeのfull depth
        embed_dim=3,#潜在変数の次元数
        ):
        super(DualOctreeVAE, self).__init__()

        self.Encoder_input_depth=Encoder_input_depth
        self.Encoder_output_depth=Encoder_output_depth
        self.Decoder_input_depth=Decoder_input_depth
        self.Decoder_output_depth=Decoder_output_depth
        self.Encoder_in_channels=Encoder_in_channels
        self.Dual_Octree_full_depth=Dual_Octree_full_depth
        self.Encoder=self.set_encoder_module()
        self.Decoder=self.set_decoder_module()
        self.channels=[]


        self.KL_conv=modules.Conv1x1(
            self.channels[self.Encoder_output_depth],
            2*self.embed_dim,#潜在変数の次元数の2倍(平均用と分散用)
            use_bias=True)
        self.post_KL_conv=modules.Conv1x1(
            self.embed_dim,#潜在変数の次元数
            self.channels[self.Encoder_output_depth],
            use_bias=True)
    def set_encoder_module(self):
        pass
    def set_decoder_module(self):
        pass
    def get_feature_vector(self,dual_octree_graph):
        pass
    def dual_octree_encoder_step(self,dual_octree_graph):
         #階層ごとで出力されるDualOctreeGraphを保存するための辞書
        convs=dict()
        data=self.get_feature_vector(dual_octree_graph)
        convs[self.Encoder_input_depth]=data
        #Encoderの操作は、depth=8からdepth=6まで行う.
        for i,d in enumerate(range(self.Encoder_input_depth,self.Encoder_output_depth-1,-1)):
            input=convs[d]
            #Encoderモジュールを適用する.
            Encoder_output=self.Encoder[i](input)
            convs[d]=Encoder_output
            if d>self.Encoder_output_depth:
                #Downsampleモジュールを適用する.
                downsample_output=self.Downsample[i](input)
                convs[d-1]=downsample_output
        #Encoderの出力を正規化する.
        convs[self.Encoder_output_depth]=self.Encoder_norm_out(convs[self.Encoder_output_depth])
        #Encoderの出力を活性化関数で活性化する.
        convs[self.Encoder_output_depth]=self.activation(convs[self.Encoder_output_depth])
        return convs
    def dual_octree_encoder(self,dual_octree_graph):
        convs=self.dual_octree_encoder_step(dual_octree_graph)
        
        code=self.KL_conv(convs[self.Encoder_output_depth])
        #事後分布を計算するクラス
        posterior=DiagonalGaussianDistribution(code)
        return posterior
    def decision_split_label(self,logit):
        pass
    def update_dual_octree_graph(self,dual_octree_graph,label,d):
        pass
    def dual_octree_decoder(self,code,dual_octree_graph,update_octree=False):

        code=self.post_KL_conv(code)
        
        #辞書の初期化
        octree_split_logits=dict()#logitsは確率になる前のスコア
        reg_voxel_values=dict()#分割予測のためのボクセルの座標
        deconvs=dict()
        
        deconvs[self.Decoder_input_depth]=code



        for i,d in enumerate(range(self.Decoder_input_depth,self.Decoder_output_depth+1)):
            if d>self.Decoder_input_depth:
                #upsampleモジュールを適用する.
                pass
            deconvs[d]=self.Decoder[i](deconvs[d])

            #分割予測を行う.
            logit=self.predict_split_label[i](deconvs[d])
            #存在する頂点個分のlogitを取得する.(logitの後ろからnode_num個分)
            node_num=dual_octree_graph.node_num[d]
            octree_split_logits[d]=logit[-node_num:]
            #推論時は、octree_split_logitsを使用して、octreeを分割する.(update_octree=True)
            if update_octree:
                label=self.decision_split_label(octree_split_logits[d])
                #octreeを分割し、DualOctreeGraphを更新する.
                dual_octree_graph=self.update_dual_octree_graph(dual_octree_graph,label,d)

            #ボクセル値（空白（ー１）、葉（０）、枝（０．5）,幹（１））を予測する.
            reg_voxel_value=self.reg_voxel_value[i](deconvs[d])
            #非葉ノードのnodeを取得する.
            node_mask=self.get_non_leaf_node_mask(dual_octree_graph)
            shape=(node_mask.shape[0],reg_voxel_value.shape[1])#ノード数×特徴量の次元数
            reg_voxel_value_pad=torch.zeros(shape,device=reg_voxel_value.device)
            reg_voxel_value_pad[node_mask]=reg_voxel_value
            reg_voxel_values[d]=reg_voxel_value_pad
            
            
            
    def forward(self,octree):
        #Encoderを適用して、事後分布を計算する.
        posterior=self.dual_octree_encoder(dual_octree_graph)
        #事後分布からサンプルを生成する.
        z=posterior.sample()
        #Decoderを適用する.
        Decoder_output=self.dual_octree_decoder(z,dual_octree_graph)
        return Decoder_output



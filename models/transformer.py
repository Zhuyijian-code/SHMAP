import copy # 导入copy模块，用于创建对象的深拷贝
from typing import Optional, Any # 导入类型提示相关的模块

import torch # 导入PyTorch库
from torch import Tensor # 导入PyTorch张量类型
import torch.nn.functional as F # 导入PyTorch函数式接口
from torch.nn import Module # 导入PyTorch基础模块类
from torch.nn import MultiheadAttention # 导入多头注意力机制
from torch.nn import ModuleList # 导入模块列表
from torch.nn.init import xavier_uniform_ # 导入Xavier均匀分布初始化
from torch.nn import Dropout # 导入Dropout层
from torch.nn import Linear # 导入线性层
from torch.nn import LayerNorm # 导入层归一化


class Transformer(Module):
    r"""一个Transformer模型。用户可以根据需要修改其属性。该架构基于论文"Attention Is All You Need"。
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. 用户可以通过相应的参数构建BERT模型。

    参数:
        d_model: 编码器/解码器输入中期望的特征数量（默认=512）
        nhead: 多头注意力模型中的头数（默认=8）
        num_encoder_layers: 编码器中的子编码器层数（默认=6）
        num_decoder_layers: 解码器中的子解码器层数（默认=6）
        dim_feedforward: 前馈网络模型的维度（默认=2048）
        dropout: dropout值（默认=0.1）
        activation: 编码器/解码器中间层的激活函数，relu或gelu（默认=relu）
        custom_encoder: 自定义编码器（默认=None）
        custom_decoder: 自定义解码器（默认=None）
        layer_norm_eps: 层归一化组件中的eps值（默认=1e-5）
        batch_first: 如果为True，则输入和输出张量以(batch, seq, feature)的形式提供。
            默认为False（seq, batch, feature）
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype} # 创建工厂参数字典
        super(Transformer, self).__init__() # 调用父类构造函数

        if custom_encoder is not None: # 如果提供了自定义编码器
            self.encoder = custom_encoder # 使用自定义编码器
        else: # 否则创建标准编码器
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first,
                                                    **factory_kwargs) # 创建编码器层
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) # 创建层归一化
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm) # 创建编码器

        if custom_decoder is not None: # 如果提供了自定义解码器
            self.decoder = custom_decoder # 使用自定义解码器
        else: # 否则创建标准解码器
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first,
                                                    **factory_kwargs) # 创建解码器层
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) # 创建层归一化
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm) # 创建解码器

        self._reset_parameters() # 重置参数

        self.d_model = d_model # 保存模型维度
        self.nhead = nhead # 保存注意力头数
        self.batch_first = batch_first # 保存批次维度位置标志

        # self.pos_embed = PositionEmbeddingSine(d_model, dropout)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None):
        r"""处理带掩码的源序列和目标序列。

        参数:
            src: 输入到编码器的序列（必需）
            tgt: 输入到解码器的序列（必需）
            src_mask: 源序列的加法掩码（可选）
            tgt_mask: 目标序列的加法掩码（可选）
            memory_mask: 编码器输出的加法掩码（可选）
            src_key_padding_mask: 每个批次的源键的ByteTensor掩码（可选）
            tgt_key_padding_mask: 每个批次的目标键的ByteTensor掩码（可选）
            memory_key_padding_mask: 每个批次的内存键的ByteTensor掩码（可选）
        """

        if not self.batch_first and src.size(1) != tgt.size(1): # 检查批次大小是否匹配
            raise RuntimeError("源序列和目标序列的批次数量必须相等")
        elif self.batch_first and src.size(0) != tgt.size(0):
            raise RuntimeError("源序列和目标序列的批次数量必须相等")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model: # 检查特征维度是否匹配
            raise RuntimeError("源序列和目标序列的特征数量必须等于d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask) # 编码器处理源序列
        # memory = self.pos_embed(memory)
        # import numpy as np
        # np.save('./after.npy', memory.squeeze(0).cpu().detach().numpy())

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, # 解码器处理目标序列
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output, memory # 返回输出和内存状态

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""生成序列的方形掩码。被掩码的位置填充float('-inf')，未掩码的位置填充float(0.0)。"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1) # 创建上三角矩阵
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)) # 填充掩码值
        return mask

    def _reset_parameters(self):
        r"""初始化transformer模型中的参数。"""
        for p in self.parameters(): # 遍历所有参数
            if p.dim() > 1: # 如果参数维度大于1
                xavier_uniform_(p) # 使用Xavier均匀分布初始化


class TransformerEncoder(Module):
    r"""TransformerEncoder是N个编码器层的堆栈

    参数:
        encoder_layer: TransformerEncoderLayer类的实例（必需）
        num_layers: 编码器中的子编码器层数（必需）
        norm: 层归一化组件（可选）
    """
    __constants__ = ['norm'] # 定义常量

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__() # 调用父类构造函数
        self.layers = _get_clones(encoder_layer, num_layers) # 创建编码器层的副本
        self.num_layers = num_layers # 保存层数
        self.norm = norm # 保存归一化层

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""依次通过编码器层传递输入。

        参数:
            src: 输入到编码器的序列（必需）
            mask: 源序列的掩码（可选）
            src_key_padding_mask: 每个批次的源键的掩码（可选）
        """
        output = src # 初始化输出

        for mod in self.layers: # 遍历所有编码器层
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask) # 通过每一层

        if self.norm is not None: # 如果存在归一化层
            output = self.norm(output) # 应用归一化

        return output # 返回输出


class TransformerDecoder(Module):
    r"""TransformerDecoder是N个解码器层的堆栈

    参数:
        decoder_layer: TransformerDecoderLayer类的实例（必需）
        num_layers: 解码器中的子解码器层数（必需）
        norm: 层归一化组件（可选）
    """
    __constants__ = ['norm'] # 定义常量

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__() # 调用父类构造函数
        self.layers = _get_clones(decoder_layer, num_layers) # 创建解码器层的副本
        self.num_layers = num_layers # 保存层数
        self.norm = norm # 保存归一化层

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""依次通过解码器层传递输入（和掩码）。

        参数:
            tgt: 输入到解码器的序列（必需）
            memory: 来自编码器最后一层的序列（必需）
            tgt_mask: 目标序列的掩码（可选）
            memory_mask: 内存序列的掩码（可选）
            tgt_key_padding_mask: 每个批次的目标键的掩码（可选）
            memory_key_padding_mask: 每个批次的内存键的掩码（可选）
        """
        output = tgt # 初始化输出

        for mod in self.layers: # 遍历所有解码器层
            output = mod(output, memory, tgt_mask=tgt_mask, # 通过每一层
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None: # 如果存在归一化层
            output = self.norm(output) # 应用归一化

        return output # 返回输出


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer由自注意力和前馈网络组成。
    这个标准编码器层基于论文"Attention Is All You Need"。
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010.

    参数:
        d_model: 输入中期望的特征数量（必需）
        nhead: 多头注意力模型中的头数（必需）
        dim_feedforward: 前馈网络模型的维度（默认=2048）
        dropout: dropout值（默认=0.1）
        activation: 中间层的激活函数，relu或gelu（默认=relu）
        layer_norm_eps: 层归一化组件中的eps值（默认=1e-5）
        batch_first: 如果为True，则输入和输出张量以(batch, seq, feature)的形式提供。
            默认为False
    """
    __constants__ = ['batch_first'] # 定义常量

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype} # 创建工厂参数字典
        super(TransformerEncoderLayer, self).__init__() # 调用父类构造函数
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs) # 创建自注意力层
        # 实现前馈网络
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs) # 第一个线性层
        self.dropout = Dropout(dropout) # dropout层
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs) # 第二个线性层

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) # 第一个层归一化
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) # 第二个层归一化
        self.dropout1 = Dropout(dropout) # 第一个dropout层
        self.dropout2 = Dropout(dropout) # 第二个dropout层

        self.activation = _get_activation_fn(activation) # 获取激活函数

    def __setstate__(self, state):
        if 'activation' not in state: # 如果状态中没有激活函数
            state['activation'] = F.relu # 设置默认激活函数为ReLU
        super(TransformerEncoderLayer, self).__setstate__(state) # 调用父类的__setstate__

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""通过编码器层传递输入。

        参数:
            src: 输入到编码器层的序列（必需）
            src_mask: 源序列的掩码（可选）
            src_key_padding_mask: 每个批次的源键的掩码（可选）
        """
        src2, att_weights = self.self_attn(src, src, src, attn_mask=src_mask, # 自注意力计算
                                           key_padding_mask=src_key_padding_mask)
        # print(att_weights)
        # raise SystemExit
        src = src + self.dropout1(src2) # 残差连接和dropout
        src = self.norm1(src) # 层归一化
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))) # 前馈网络
        src = src + self.dropout2(src2) # 残差连接和dropout
        src = self.norm2(src) # 层归一化
        return src # 返回输出


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer由自注意力、多头注意力和前馈网络组成。
    这个标准解码器层基于论文"Attention Is All You Need"。
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010.

    参数:
        d_model: 输入中期望的特征数量（必需）
        nhead: 多头注意力模型中的头数（必需）
        dim_feedforward: 前馈网络模型的维度（默认=2048）
        dropout: dropout值（默认=0.1）
        activation: 中间层的激活函数，relu或gelu（默认=relu）
        layer_norm_eps: 层归一化组件中的eps值（默认=1e-5）
        batch_first: 如果为True，则输入和输出张量以(batch, seq, feature)的形式提供。
            默认为False
    """
    __constants__ = ['batch_first'] # 定义常量

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype} # 创建工厂参数字典
        super(TransformerDecoderLayer, self).__init__() # 调用父类构造函数
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs) # 创建自注意力层
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs) # 创建多头注意力层
        # 实现前馈网络
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs) # 第一个线性层
        self.dropout = Dropout(dropout) # dropout层
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs) # 第二个线性层

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) # 第一个层归一化
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) # 第二个层归一化
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) # 第三个层归一化
        self.dropout1 = Dropout(dropout) # 第一个dropout层
        self.dropout2 = Dropout(dropout) # 第二个dropout层
        self.dropout3 = Dropout(dropout) # 第三个dropout层

        self.activation = _get_activation_fn(activation) # 获取激活函数

    def __setstate__(self, state):
        if 'activation' not in state: # 如果状态中没有激活函数
            state['activation'] = F.relu # 设置默认激活函数为ReLU
        super(TransformerDecoderLayer, self).__setstate__(state) # 调用父类的__setstate__

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""通过解码器层传递输入（和掩码）。

        参数:
            tgt: 输入到解码器层的序列（必需）
            memory: 来自编码器最后一层的序列（必需）
            tgt_mask: 目标序列的掩码（可选）
            memory_mask: 内存序列的掩码（可选）
            tgt_key_padding_mask: 每个批次的目标键的掩码（可选）
            memory_key_padding_mask: 每个批次的内存键的掩码（可选）
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, # 自注意力计算
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2) # 残差连接和dropout
        tgt = self.norm1(tgt) # 层归一化
        tgt2, att_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, # 多头注意力计算
                                                key_padding_mask=memory_key_padding_mask)
        # print(att_weights)
        # import pdb
        # pdb.set_trace()
        tgt = tgt + self.dropout2(tgt2) # 残差连接和dropout
        tgt = self.norm2(tgt) # 层归一化
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt)))) # 前馈网络
        tgt = tgt + self.dropout3(tgt2) # 残差连接和dropout
        tgt = self.norm3(tgt) # 层归一化
        return tgt # 返回输出


def _get_clones(module, N):
    """创建模块的N个深拷贝"""
    return ModuleList([copy.deepcopy(module) for i in range(N)]) # 返回模块列表


def _get_activation_fn(activation):
    """获取激活函数"""
    if activation == "relu": # 如果是ReLU
        return F.relu # 返回ReLU函数
    elif activation == "gelu": # 如果是GELU
        return F.gelu # 返回GELU函数

    raise RuntimeError("activation应该是relu/gelu，而不是{}".format(activation)) # 抛出错误

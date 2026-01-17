import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from einops import rearrange

from models.base_model import BaseModel, CustomGELU, BaseConvBlock
from models.sam import SAM

def score_to_grade_label(score, n_query, min_score=0.0, max_score=10.0):
    norm_score = (score - min_score) / (max_score - min_score)
    if hasattr(norm_score, 'clamp'):
        norm_score = norm_score.clamp(0, 1)
    else:
        norm_score = min(max(norm_score, 0), 1)
    if hasattr(norm_score, 'long'):
        grade = (norm_score * n_query).long()
    else:
        grade = int(norm_score * n_query)
    if hasattr(grade, 'clamp'):
        grade = grade.clamp(0, n_query - 1)
    else:
        grade = min(max(grade, 0), n_query - 1)
    return grade

def smart_fn(fn, feat_list, func_list=False):
    if func_list is False:
        return [fn(x) for x in feat_list]
    return [fn[idx](x) for idx, x in enumerate(feat_list)]

class Multi_Head_Attention(nn.Module):
    def __init__(self, num_heads, dim_model, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        assert dim_model % num_heads == 0
        self.dim_head = dim_model // self.num_heads
        self.linear_layers = nn.ModuleList([nn.Linear(dim_model, dim_model) for _ in range(3)])
        self.drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(dim_model)

    def self_attention(self, Q, K, V, mask=None, dropout=None):
        scores = -1 * torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(dim=1).repeat([1, self.num_heads, 1, 1])
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        context = torch.matmul(attn, V)
        return context, attn

    def forward(self, input_Q, input_K, input_V, mask=None):
        residual_Q, b = input_Q, input_V.size(0)
        Q, K, V = [l(x).view(b, -1, self.num_heads, self.dim_head).transpose(1, 2)
                   for l, x in zip(self.linear_layers, (input_Q, input_K, input_V))]
        x, attn = self.self_attention(Q, K, V, mask=mask, dropout=self.drop)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.num_heads * self.dim_head)
        return x

class MSA(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.fusion_s = Multi_Head_Attention(num_heads=num_heads, dim_model=dim, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, s, c):
        if s[0].shape[-1] != c.shape[-1]:
            s = [F.interpolate(x, size=c.shape[-1], mode='nearest') for x in s]
        q = rearrange(self.proj(c), 'b c t -> (b t) 1 c')
        s = torch.stack(s, dim=2)
        s = rearrange(s, 'b c n t -> (b t) n c')
        ns = self.fusion_s(q, s, s)
        ns = ns.mean(dim=1, keepdim=True)
        ns = rearrange(ns, '(b t) 1 c -> b c t', b=c.shape[0])
        return ns

class MCO(nn.Module):
    def __init__(self, dim, K,  num_heads=1,  gap=False, dropout=0.1, n_query_fixed=None):
        # gap: 是否使用全局平均池化，默认为False
        # dropout: dropout比率，默认为0.1
        # n_query_fixed: CDM/SAM模块的固定查询数量（序列长度）
        super().__init__()
        self.K = K
        self.proj1 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, self.K, 1),
        ) for _ in range(3)
        ])
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)
        ) for _ in range(self.K)
        ])
        if gap:
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif n_query_fixed is not None:
            self.pool = nn.AdaptiveAvgPool1d(n_query_fixed)
        else:
            self.pool = nn.AvgPool1d(2, 2)

        self.fusion_f = Multi_Head_Attention(num_heads=num_heads, dim_model=dim, dropout=dropout)
        self.proj2 = nn.Sequential(
            nn.Conv1d(2 * dim, dim, 1)
        )

        self.policy = nn.Sequential(
            nn.Conv1d(3*dim, K, 1)
        )

    def gen_mask(self, integration_path):
        b, k, t = integration_path.shape
        mask = torch.zeros_like(integration_path, device=integration_path.device)
        idx = torch.arange(0, k, device=integration_path.device).repeat([b, 1, 1]).permute([0, 2, 1]).repeat([1, 1, t])
        pos_mask = idx > torch.mul(integration_path, idx).sum(dim=1, keepdim=True)
        mask[pos_mask] = -1e9
        mask = mask + integration_path - integration_path.detach()
        return mask

    def gen_integration_path(self, f, tau):
        logit = self.pool(self.policy(torch.cat(f, dim=1)))
        integration_path = F.gumbel_softmax(logit, tau, hard=True, dim=1)
        mask = self.gen_mask(integration_path)
        return integration_path, mask

    def forward(self, f, c, s, tau):
        att = torch.softmax(torch.stack(smart_fn(self.proj1, f, True), dim=1), dim=1)
        att = [_ for _ in att.split(1, dim=2)]
        nf = [torch.mul(_, torch.stack(f, dim=1)).sum(dim=1) for _ in att]
        nf = torch.stack([self.pool(self.ffn[idx](_) + _) for idx, _ in enumerate(nf)], dim=2)
        nf = rearrange(nf, 'b c n t -> (b t) n c')
        
        if s.shape[-1] != c.shape[-1]:
            s = F.interpolate(s, size=c.shape[-1], mode='nearest')
        q = self.proj2(torch.cat([c, s], dim=1))
        q = F.adaptive_avg_pool1d(q, nf.shape[-1])
        q = rearrange(q, 'b c t -> (b t) 1 c')
        q = -1 * q
        integration_path, mask = self.gen_integration_path(f, tau)
        mask = rearrange(mask, 'b n t -> (b t) 1 n')
        nf = self.fusion_f(q, nf, nf, mask=mask)
        nf = torch.mean(nf, dim=1, keepdim=True)
        nf = rearrange(nf, '(b t) 1 c -> b c t', b=c.shape[0])
        return nf, integration_path

class ProgressiveUnit(nn.Module):
    def __init__(self, dim, K, ms_heads, cm_heads, gap=False, n_query_fixed=None):
        # ms_heads: 多尺度注意力头数
        # cm_heads: 跨模态注意力头数
        # gap: 是否使用全局平均池化
        # n_query_fixed: CDM/SAM模块的固定查询数量（序列长度）
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)
        )
        self.short_cut = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)
        )
        self.s_fusion = MSA(dim, ms_heads)
        self.f_fusion = MCO(dim, K, cm_heads, gap, n_query_fixed=n_query_fixed)
        self.cat_adapter = nn.Conv1d(dim*3, dim, 1)

    def forward(self, c, f, s, tau):
        c = self.conv(c) + self.short_cut(c)
        s = self.s_fusion(s, c)
        f, action = self.f_fusion(f, c, s, tau)
        if s.shape[-1] != c.shape[-1]:
            s = F.interpolate(s, size=c.shape[-1], mode='nearest')
        if f.shape[-1] != c.shape[-1]:
            f = F.interpolate(f, size=c.shape[-1], mode='nearest')
        c = torch.cat([c, s, f], dim=1)
        c = self.cat_adapter(c)
        return c, action

class SHMAP(nn.Module):
    def __init__(self,
                 model_dim, fc_drop, fc_r, feat_drop, K,
                 ms_heads, cm_heads,
                 ckpt_dir=None, rgb_ckpt_name=None, flow_ckpt_name=None, audio_ckpt_name=None,
                 dataset_name=None,
                 sam_n_head=8,
                 sam_n_encoder=3,
                 sam_n_decoder=3,
                 sam_n_query=4,
                 sam_dropout=0.1,
                 sam_activate_regular_restrictions=0
                 ):
        super().__init__()
        
        self.model_dim = model_dim
        
        def load_model_weights(model_path):
            import sys
            import models.shmap as shmap_module
            
            sys.modules['models.pamfn'] = shmap_module
            
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, nn.Module):
                    return checkpoint.state_dict()
                return checkpoint
            finally:
                if 'models.pamfn' in sys.modules:
                    del sys.modules['models.pamfn']

        self.model_r = BaseModel(in_dim=1024, model_dim=model_dim, drop_rate=0.5, modality="V")
        self.model_f = BaseModel(in_dim=1024, model_dim=model_dim, drop_rate=0.5, modality="V")
        self.model_a = BaseModel(in_dim=768, model_dim=model_dim, drop_rate=0.5, modality="A")
        
        if ckpt_dir is not None and dataset_name is not None:
            self.model_r.load_state_dict(load_model_weights(osp.join(ckpt_dir, f"{dataset_name}_rgb_{rgb_ckpt_name}.pth")))
            self.model_f.load_state_dict(load_model_weights(osp.join(ckpt_dir, f"{dataset_name}_flow_{flow_ckpt_name}.pth")))
            self.model_a.load_state_dict(load_model_weights(osp.join(ckpt_dir, f"{dataset_name}_audio_{audio_ckpt_name}.pth")))

        def replace_gelu(module):
            for name, child in module.named_children():
                if isinstance(child, nn.GELU):
                    setattr(module, name, CustomGELU())
                else:
                    replace_gelu(child)

        replace_gelu(self.model_r)
        replace_gelu(self.model_f)
        replace_gelu(self.model_a)

        self.shared_cdm = SAM(
            in_dim=model_dim,
            hidden_dim=model_dim,
            n_head=sam_n_head,
            n_encoder=sam_n_encoder,
            n_decoder=sam_n_decoder,
            n_query=sam_n_query,
            dropout=sam_dropout,
            activate_regular_restrictions=sam_activate_regular_restrictions
        )

        self.rgb_sam = SAM(
            in_dim=self.model_dim,
            hidden_dim=self.model_dim,
            n_head=4,
            n_encoder=2,
            n_decoder=2,
            n_query=5,
            dropout=0.1,
            activate_regular_restrictions=0
        )
        self.flow_sam = SAM(
            in_dim=self.model_dim,
            hidden_dim=self.model_dim,
            n_head=4,
            n_encoder=2,
            n_decoder=2,
            n_query=5,
            dropout=0.1,
            activate_regular_restrictions=0
        )
        self.audio_sam = SAM(
            in_dim=self.model_dim,
            hidden_dim=self.model_dim,
            n_head=4,
            n_encoder=2,
            n_decoder=2,
            n_query=5,
            dropout=0.1,
            activate_regular_restrictions=0
        )
        self.fusion_sam_proj = nn.Conv1d(self.model_dim * 3, 256, 1)
        self.fusion_sam = SAM(
            in_dim=256,
            hidden_dim=256,
            n_head=4,
            n_encoder=2,
            n_decoder=2,
            n_query=5,
            dropout=0.1,
            activate_regular_restrictions=0
        )

        self.c = nn.Parameter(torch.zeros([model_dim]), requires_grad=False)
        
        self.stage1 = ProgressiveUnit(model_dim, K, ms_heads, cm_heads, n_query_fixed=sam_n_query)
        self.stage2 = ProgressiveUnit(model_dim, K, ms_heads, cm_heads, n_query_fixed=sam_n_query)
        self.stage3 = ProgressiveUnit(model_dim, K, ms_heads, cm_heads, n_query_fixed=sam_n_query)
        
        self.lightweight_stage_attention = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=2, dropout=0.05, batch_first=True
        )
        
        self.simple_stage_weight = nn.Sequential(
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(True),
            nn.Linear(model_dim // 4, 3)
        )
        
        self.stage_pos_embedding = nn.Parameter(torch.randn(3, model_dim) * 0.2)
        
        self.feature_alignment = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )
        
        self.feature_temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        self.pool = nn.AvgPool1d(2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(feat_drop)

        self.fc = nn.Sequential(
            nn.Dropout(fc_drop),
            nn.Conv1d(model_dim, model_dim//fc_r, 1),
            nn.BatchNorm1d(model_dim//fc_r),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(model_dim//fc_r, 1, 1),
            nn.Sigmoid()
        )
        self.tau = nn.Parameter(torch.ones([]) * 10)
        self.mse = nn.MSELoss()

        self.rgb_adapter = nn.ModuleList([
            nn.Conv1d(2 * self.model_dim, self.model_dim, 1) for _ in range(3)
        ])
        self.flow_adapter = nn.ModuleList([
            nn.Conv1d(2 * self.model_dim, self.model_dim, 1) for _ in range(3)
        ])
        self.audio_adapter = nn.ModuleList([
            nn.Conv1d(2 * self.model_dim, self.model_dim, 1) for _ in range(3)
        ])

        self.final_fc = nn.Sequential(
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.ReLU(True),
            nn.Linear(self.model_dim, 1),
            nn.Sigmoid()
        )
        for m in self.final_fc:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.01, b=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.final_fc = nn.Sequential(
            nn.Linear(self.model_dim + 256, self.model_dim // 2),
            nn.ReLU(True),
            nn.Linear(self.model_dim // 2, 1),
            nn.Sigmoid()
        )



    def forward(self, input_feats):
        
        if 'rgb' not in input_feats and 'V' in input_feats:
            input_feats = {
                'rgb': input_feats['V'],
                'flow': input_feats['F'],
                'audio': input_feats['A']
            }
        
        base_model_input_rgb = {
            'V': input_feats['rgb'],
            'A': input_feats['audio']
        }
        
        base_model_input_flow = {
            'V': input_feats['flow'],
            'A': input_feats['audio']
        }
        
        base_model_input_audio = {
            'V': input_feats['rgb'],
            'A': input_feats['audio']
        }
        
        _, rgb_info = self.model_r(base_model_input_rgb)
        _, flow_info = self.model_f(base_model_input_flow)
        _, audio_info = self.model_a(base_model_input_audio)
        
        rgb_feats = rgb_info['feats'][:3]
        flow_feats = flow_info['feats'][:3]
        audio_feats = audio_info['feats'][:3]
        
        rgb_enhanced_feats = []
        for i, feat in enumerate(rgb_feats):
            B, D, T = feat.shape
            
            feat_transposed = feat.transpose(1, 2)
            
            sam_output = self.shared_cdm(feat_transposed)
            enhanced_feat = sam_output['embed']
            
            enhanced_feat = enhanced_feat.transpose(1, 2)
            
            if enhanced_feat.size(2) != T:
                enhanced_feat = F.adaptive_avg_pool1d(enhanced_feat, T)
            
            fused_feat = torch.cat([feat, enhanced_feat], dim=1)
            rgb_enhanced_feats.append(fused_feat)
        
        flow_enhanced_feats = []
        for i, feat in enumerate(flow_feats):
            B, D, T = feat.shape
            feat_transposed = feat.transpose(1, 2)
            sam_output = self.shared_cdm(feat_transposed)
            enhanced_feat = sam_output['embed']
            enhanced_feat = enhanced_feat.transpose(1, 2)
            
            if enhanced_feat.size(2) != T:
                enhanced_feat = F.adaptive_avg_pool1d(enhanced_feat, T)
            
            fused_feat = torch.cat([feat, enhanced_feat], dim=1)
            flow_enhanced_feats.append(fused_feat)
        
        audio_enhanced_feats = []
        for i, feat in enumerate(audio_feats):
            B, D, T = feat.shape
            feat_transposed = feat.transpose(1, 2)
            sam_output = self.shared_cdm(feat_transposed)
            enhanced_feat = sam_output['embed']
            enhanced_feat = enhanced_feat.transpose(1, 2)
            
            if enhanced_feat.size(2) != T:
                enhanced_feat = F.adaptive_avg_pool1d(enhanced_feat, T)
            
            fused_feat = torch.cat([feat, enhanced_feat], dim=1)
            audio_enhanced_feats.append(fused_feat)
        
        rgb_feats = [self.rgb_adapter[i](feat) for i, feat in enumerate(rgb_enhanced_feats)]
        flow_feats = [self.flow_adapter[i](feat) for i, feat in enumerate(flow_enhanced_feats)]
        audio_feats = [self.audio_adapter[i](feat) for i, feat in enumerate(audio_enhanced_feats)]
        
        feats = [
            [rgb_feats[0], flow_feats[0], audio_feats[0]],
            [rgb_feats[1], flow_feats[1], audio_feats[1]],
            [rgb_feats[2], flow_feats[2], audio_feats[2]],
        ]
        
        x = self.c.repeat([feats[0][0].shape[0], feats[0][0].shape[2], 1]).permute([0, 2, 1])

        x1, action1 = self.stage1(x, feats[0], feats[0], self.tau)
        x2, action2 = self.stage2(x1, feats[1], feats[1], self.tau)
        x3, action3 = self.stage3(x2, feats[2], feats[2], self.tau)

        x1_pooled = x1.mean(dim=2)
        x2_pooled = x2.mean(dim=2)
        x3_pooled = x3.mean(dim=2)
        stage_features = torch.stack([x1_pooled, x2_pooled, x3_pooled], dim=1)

        B = stage_features.size(0)
        pos_encoded_stages = stage_features + self.stage_pos_embedding.unsqueeze(0).expand(B, -1, -1)

        enhanced_stages, _ = self.lightweight_stage_attention(
            pos_encoded_stages, pos_encoded_stages, pos_encoded_stages
        )

        global_stage_feat = enhanced_stages.mean(dim=1)
        stage_weights = F.softmax(self.simple_stage_weight(global_stage_feat), dim=-1)

        original_stages = torch.stack([x1_pooled, x2_pooled, x3_pooled], dim=1)
        enhanced_avg = enhanced_stages
        
        mixed_stages = 0.7 * original_stages + 0.3 * enhanced_avg
        final_fused_feature = (mixed_stages * stage_weights.unsqueeze(-1)).sum(dim=1)

        x3_enhanced = final_fused_feature.unsqueeze(-1).expand_as(x3)
        x3 = x3 + 0.6 * x3_enhanced
        
        if not hasattr(self, '_weight_printed') or self._weight_printed < 10:
            avg_weights = stage_weights.mean(dim=0)
            print(f"[轻量ASIF] 阶段权重: S1={avg_weights[0]:.3f}, S2={avg_weights[1]:.3f}, S3={avg_weights[2]:.3f}")
            self._weight_printed = getattr(self, '_weight_printed', 0) + 1
        
        final_output_from_fc = self.fc(x3)
        
        score_unaggregated = final_output_from_fc.squeeze(dim=1)
        
        score = torch.mean(score_unaggregated, dim=1)
        
        rgb_clip_feat = rgb_feats[0].transpose(1, 2)
        rgb_sam_out = self.rgb_sam(rgb_clip_feat)
        fusion_clip_feat = torch.cat([
            rgb_feats[0], flow_feats[0], audio_feats[0]
        ], dim=1)
        fusion_clip_feat = self.fusion_sam_proj(fusion_clip_feat)
        fusion_clip_feat = fusion_clip_feat.transpose(1, 2)
        fusion_sam_out = self.fusion_sam(fusion_clip_feat)
        main_feat = x3_pooled
        rgb_grade_embed = rgb_sam_out['embed']
        sam_feat = rgb_grade_embed.mean(dim=1)
        
        aligned_sam_feat = self.feature_alignment(sam_feat)
        
        main_feat_scaled = main_feat * self.feature_temperature
        aligned_sam_feat_scaled = aligned_sam_feat * self.feature_temperature
        
        final_feat = torch.cat([main_feat_scaled, aligned_sam_feat_scaled], dim=1)
        score = self.final_fc(final_feat).squeeze(1)
        return score, {
            'action': [action1, action2, action3],
            'enhanced_features': {
                'rgb': rgb_enhanced_feats,
                'flow': flow_enhanced_feats,
                'audio': audio_enhanced_feats
            },
            'rgb_sam': rgb_sam_out,
            'fusion_sam': fusion_sam_out,
            'rgb_sam_grade_logits': rgb_sam_out['output']['int']
        }

    def update(self, **kwargs):
        return

    def call_loss(self, pred, label, **kwargs):
        loss_main = self.mse(pred.squeeze(), label.squeeze())
        base_lambda_sam = 0.001
        adaptive_factor = max(0.5, min(2.0, (loss_main.item() + 0.01) * 100))
        lambda_sam = base_lambda_sam * adaptive_factor
        loss_sam = 0.0
        if 'rgb_sam_grade_logits' in kwargs and 'label_grade' in kwargs:
            import torch.nn.functional as F
            loss_sam = F.cross_entropy(kwargs['rgb_sam_grade_logits'], kwargs['label_grade'])
        if (not hasattr(self, '_loss_printed')) or (self._loss_printed < 10):
            print(f"[call_loss] loss_main: {loss_main.item():.4f}  loss_sam: {loss_sam:.4f}  lambda_sam: {lambda_sam:.3f}  adapt_factor: {adaptive_factor:.2f}")
            if not hasattr(self, '_loss_printed'):
                self._loss_printed = 1
            else:
                self._loss_printed += 1
        loss = loss_main + lambda_sam * loss_sam
        return loss
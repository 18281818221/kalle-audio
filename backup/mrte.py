
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

MEL_CHANNELS =  80
SAMPLE_RATE = 16000
HOP_LENGTH = 256

class ConvBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, activation):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv(x)

        x = rearrange(x, "B D T -> B T D")
        x = self.norm(x)
        x = rearrange(x, "B T D -> B D T")
        return x

class ConvStack(nn.Module):
    def __init__(self, hidden_size, n_blocks, kernel_size, activation):
        super(ConvStack, self).__init__()

        blocks = []
        for i in range(n_blocks):
            blocks += [
                ConvBlock(
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    activation=activation,
                )
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class ResidualBlockStack(nn.Module):
    def __init__(self, hidden_size, n_stacks, n_blocks, kernel_size, activation):
        super(ResidualBlockStack, self).__init__()

        self.conv_stacks = []

        for i in range(n_stacks):
            self.conv_stacks += [
                ConvStack(
                    hidden_size=hidden_size,
                    n_blocks=n_blocks,
                    kernel_size=kernel_size,
                    activation=activation,
                )
            ]
        self.conv_stacks = nn.Sequential(*self.conv_stacks)

    def forward(self, x):
        for conv_stack in self.conv_stacks:
            x = x + conv_stack(x)
        return x

class ConvNetDoubleLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_stacks: int,
        n_blocks: int,
        middle_layer: nn.Module,
        kernel_size: int,
        activation: str,
    ):
        super(ConvNetDoubleLayer, self).__init__()
        self.conv_stack1 = ResidualBlockStack(
            hidden_size=hidden_size,
            n_stacks=n_stacks,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            activation=activation,
        )

        self.middle_layer = middle_layer

        self.conv_stack2 = ResidualBlockStack(
            hidden_size=hidden_size,
            n_stacks=n_stacks,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            activation=activation,
        )

    def forward(self, x):
        x = self.conv_stack1(x)
        x = self.middle_layer(x)
        x = self.conv_stack2(x)
        return x

class ConvNetDouble(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        n_layers: int,
        n_stacks: int,
        n_blocks: int,
        middle_layer: nn.Module,
        kernel_size: int,
        activation: str,
    ):
        super(ConvNetDouble, self).__init__()

        self.first_layer = first_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )

        self.layers = []
        for i in range(n_layers):
            self.layers += [
                ConvNetDoubleLayer(
                    hidden_size=hidden_size,
                    n_stacks=n_stacks,
                    n_blocks=n_blocks,
                    middle_layer=middle_layer,
                    kernel_size=kernel_size,
                    activation=activation,
                )
            ]

        self.layers = nn.Sequential(*self.layers)

        self.last_layer = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x):
        x = self.first_layer(x)

        x_out = self.layers[0](x)
        for layer in self.layers[1:]:
                x_out = x_out + layer(x)

        x = self.last_layer(x_out)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, qkv_dim,  n_heads=8, dropout=0.):
        super().__init__()

        assert qkv_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = qkv_dim // n_heads
        self.dropout = dropout
        self.qkv_dim = qkv_dim

        self.w_q = nn.Linear(qkv_dim, qkv_dim, bias=True)
        self.w_k = nn.Linear(qkv_dim, qkv_dim, bias=True)
        self.w_v = nn.Linear(qkv_dim, qkv_dim, bias=True)

        self.out_proj = nn.Sequential(
            nn.Linear(qkv_dim, qkv_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv=None, mask=None):
        # import pdb;pdb.set_trace()
        bsz, tgt_len, _ = q.size()
        src_len = kv.size(1) if kv is not None else tgt_len

        if kv is None:
            k = self.w_k(q)
            v = self.w_v(q)
            q = self.w_q(q)
        else:
            k = self.w_k(kv)
            v = self.w_v(kv)
            q = self.w_q(q)
        
        q = q.view(bsz, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.n_heads, self.head_dim).transpose(1, 2)
        # import pdb;pdb.set_trace()
        att = F.scaled_dot_product_attention(
            q, k, v, mask, self.dropout if self.training else 0.0, False)

        att = att.transpose(1, 2).contiguous().view(bsz, tgt_len, self.qkv_dim)

        return self.out_proj(att)

class MRTE(nn.Module):
    def __init__(
            self,
            mel_bins: int = MEL_CHANNELS,
            # mel_frames: int = HOP_LENGTH,
            mel_activation: str = 'ReLU',
            mel_kernel_size: int = 3,
            mel_stride: int = 16,
            mel_n_layer: int = 5,
            mel_n_stack: int = 5,
            mel_n_block: int = 2,
            # content_ff_dim: int = 1024,
            content_n_heads: int = 2,
            # content_n_layers: int = 8,
            hidden_size: int = 2048,
            # duration_token_ms: float = (
            #     HOP_LENGTH / SAMPLE_RATE * 1000),
            # phone_vocab_size: int = 320,
            dropout: float = 0.1,
            # sample_rate: int = SAMPLE_RATE,
    ):
        super(MRTE, self).__init__()

        self.n_heads = content_n_heads
        self.mel_bins = mel_bins
        self.hidden_size = hidden_size

        self.mel_encoder_middle_layer = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=mel_stride + 1,
            stride=mel_stride,
            padding=(mel_stride) // 2,
        )
        self.mel_encoder = ConvNetDouble(
            in_channels=mel_bins,
            out_channels=hidden_size,
            hidden_size=hidden_size,
            n_layers=mel_n_layer,
            n_stacks=mel_n_stack,
            n_blocks=mel_n_block,
            middle_layer=self.mel_encoder_middle_layer,
            kernel_size=mel_kernel_size,
            activation=mel_activation,
        )

        # self.adapter_up = nn.Linear(256, 512)
        self.adapter_down = nn.Linear(2048, hidden_size)
        self.adapter_cond_emb = nn.Linear(hidden_size, 2048)


        self.mha = MultiHeadAttention(
            qkv_dim=hidden_size,
            n_heads=1,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)
        # self.norm_256 = nn.LayerNorm(256)
        self.activation = nn.ReLU()

    # def tc_latent(
    #         self,
    #         token2embed: torch.Tensor,  # (B, T，2048)
    #         mel: torch.Tensor,  # (B, T, mel_bins)
    # ):
    #     mel = rearrange(mel, 'B T D -> B D T')
    #     mel_context = self.mel_encoder(mel)
    #     mel_context = rearrange(mel_context, 'B D T -> B T D')
    #     token2embed_context = self.adapter_down(token2embed)
    #     tc_latent = self.mha(token2embed_context, kv=mel_context)
    #     tc_latent = self.norm(tc_latent)
    #     tc_latent = self.activation(tc_latent)
    #     tc_latent = self.adapter_down(tc_latent)
    #     tc_latent = self.norm_256(tc_latent)
    #     tc_latent = self.activation(tc_latent)
    #     tc_latent = rearrange(tc_latent,'B T D -> B D T')
    #     mel_context = torch.mean(mel_context, dim=1) #去掉时间维度，得到global的mel表征
    #     mel_context = self.adapter_cond_emb(mel_context)

    #     return tc_latent.transpose(1,2), mel_context

    def forward(self,mel: torch.Tensor, phone_x: torch.Tensor, mask=None):
        # mel = rearrange(mel, 'B T D -> B D T')
        mel_context = self.mel_encoder(mel)
        mel_context = rearrange(mel_context, 'B D T -> B T D')
        # import pdb;pdb.set_trace()
        tc_latent = self.mha(phone_x, kv=mel_context)
        tc_latent = self.norm(tc_latent)
        tc_latent = self.activation(tc_latent)

        mean_mel_context = torch.mean(mel_context, dim=1) #去掉时间维度，得到global的mel表征
        mel_context = self.adapter_cond_emb(mean_mel_context) #256
        return mel_context,tc_latent

    # def forward(
    #         self,
    #         duration_tokens: torch.Tensor,  # (B, T)
    #         phone: torch.Tensor,  # (B, T)
    #         phone_lens: torch.Tensor,  # (B,)
    #         mel: torch.Tensor,  # (B, T, mel_bins)
    # ):
    #     tc_latent = self.tc_latent(phone, phone_lens, mel)
        
    #     # out = self.length_regulator(tc_latent, duration_tokens)
    #     return tc_latent

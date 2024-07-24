import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import numpy as np

from data_assimilation_with_generative_ML.neural_network_models import LinformerAttention, PositionalEmbedding

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = torch.swapaxes(x, 0, 1)

        x = x + self.pe[:x.size(0)]

        x = torch.swapaxes(x, 0, 1)
        return self.dropout(x)

# Define a module for Gaussian random features used to encode time steps.
class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        """
        Parameters:
        - embed_dim: Dimensionality of the embedding (output dimension)
        - scale: Scaling factor for random weights (frequencies)
        """
        super().__init__()

        # Randomly sample weights (frequencies) during initialization.
        # These weights (frequencies) are fixed during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        """
        Parameters:
        - x: Input tensor representing time steps
        """
        # Calculate the cosine and sine projections: Cosine(2 pi freq x), Sine(2 pi freq x)
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi

        # Concatenate the sine and cosine projections along the last dimension
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
# Define a module for a fully connected layer that reshapes outputs to feature maps.
class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Parameters:
        - input_dim: Dimensionality of the input features
        - output_dim: Dimensionality of the output features
        """
        super().__init__()

        # Define a fully connected layer
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Parameters:
        - x: Input tensor

        Returns:
        - Output tensor after passing through the fully connected layer
          and reshaping to a 4D tensor (feature map)
        """

        # Apply the fully connected layer and reshape the output to a 4D tensor
        return self.dense(x)[..., None, None]
        # This broadcasts the 2D tensor to a 4D tensor, adding the same value across space.

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=1):
        """
        Initialize the CrossAttention module.

        Parameters:
        - embed_dim: The dimensionality of the output embeddings.
        - hidden_dim: The dimensionality of the hidden representations.
        - context_dim: The dimensionality of the context representations (if not self attention).
        - num_heads: Number of attention heads (currently supports 1 head).

        Note: For simplicity reasons, the implementation assumes 1-head attention.
        Feel free to implement multi-head attention using fancy tensor manipulations.
        """
        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim

        # Linear layer for query projection
        self.query = nn.Linear(hidden_dim, embed_dim, bias=False)
        
        # Check if self-attention or cross-attention
        if context_dim is None:
            self.self_attn = True
            self.key = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.self_attn = False
            self.key = nn.Linear(context_dim, embed_dim, bias=False)
            self.value = nn.Linear(context_dim, hidden_dim, bias=False)

    def forward(self, tokens, context=None):
        """
        Forward pass of the CrossAttention module.

        Parameters:
        - tokens: Input tokens with shape [batch, sequence_len, hidden_dim].
        - context: Context information with shape [batch, context_seq_len, context_dim].
                   If self_attn is True, context is ignored.

        Returns:
        - ctx_vecs: Context vectors after attention with shape [batch, sequence_len, embed_dim].
        """

        if self.self_attn:
            # Self-attention case
            Q = self.query(tokens)
            K = self.key(tokens)
            V = self.value(tokens)
        else:
            # Cross-attention case
            Q = self.query(tokens)
            K = self.key(context)
            V = self.value(context)

        # Compute score matrices, attention matrices, and context vectors
        scoremats = torch.einsum("BTH,BSH->BTS", Q, K)  # Inner product of Q an d K, a tensor
        attnmats = F.softmax(scoremats / math.sqrt(self.embed_dim), dim=-1)  # Softmax of scoremats
        ctx_vecs = torch.einsum("BTS,BSH->BTH", attnmats, V)  # Weighted average value vectors by attnmats

        return ctx_vecs
    
class TransformerBlock(nn.Module):
    """The transformer block that combines self-attn, cross-attn, and feed forward neural net"""
    def __init__(self, hidden_dim, context_dim):
        """
        Initialize the TransformerBlock.

        Parameters:
        - hidden_dim: The dimensionality of the hidden state.
        - context_dim: The dimensionality of the context tensor.

        Note: For simplicity, the self-attn and cross-attn use the same hidden_dim.
        """

        super(TransformerBlock, self).__init__()

        # Self-attention module
        self.attn_self = CrossAttention(hidden_dim, hidden_dim)

        # Cross-attention module
        self.attn_cross = CrossAttention(hidden_dim, hidden_dim, context_dim)

        # Layer normalization modules
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Implement a 2-layer MLP with K * hidden_dim hidden units, and nn.GELU nonlinearity
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 3 * hidden_dim),
            nn.GELU(),
            nn.Linear(3 * hidden_dim, hidden_dim)
        )

    def forward(self, x, context=None):
        """
        Forward pass of the TransformerBlock.

        Parameters:
        - x: Input tensor with shape [batch, sequence_len, hidden_dim].
        - context: Context tensor with shape [batch, context_seq_len, context_dim].

        Returns:
        - x: Output tensor after passing through the TransformerBlock.
        """

        # Apply self-attention with layer normalization and residual connection
        x = self.attn_self(self.norm1(x)) + x

        if context is not None:
            # Apply cross-attention with layer normalization and residual connection
            x = self.attn_cross(self.norm2(x), context=context) + x

        # Apply feed forward neural network with layer normalization and residual connection
        # x = self.ffn(self.norm3(x)) + x

        return x

class SpatialLinformerAttention(nn.Module):
    def __init__(self, seq_len, dim, n_heads, k, bias=True):
        """
        Initialize the SpatialLinFormerBlock.

        Parameters:
        - hidden_dim: The dimensionality of the hidden state.
        - context_dim: The dimensionality of the context tensor.
        """
        super(SpatialLinformerAttention, self).__init__()

        self.norm = nn.GroupNorm(1, dim)
        
        self.attention = LinformerAttention(
            seq_len=seq_len,
            dim=dim,
            n_heads=n_heads,
            k=k,
            bias=bias
        )

        self.pos_embedding = PositionalEncoding(dim) #GaussianFourierProjection(dim)

    def forward(self, x, context=None):
        """
        Forward pass of the SpatialLinFormerBlock.

        Parameters:
        - x: Input tensor with shape [batch, channels, height, width].
        - context: Context tensor with shape [batch, context_seq_len, context_dim].

        Returns:
        - x: Output tensor after applying spatial transformation.
        """
        b, c, h, w = x.shape
        x_in = x

        # Apply norm
        x = self.norm(x)

        # Combine the spatial dimensions and move the channel dimension to the end
        x = rearrange(x, "b c h w -> b (h w) c")

        # Apply the positional embedding
        x = self.pos_embedding(x)


        # Apply the sequence transformer
        x = self.attention(x)

        # Reverse the process
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        # Residue connection
        return x + x_in


class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        """
        Initialize the SpatialTransformer.

        Parameters:
        - hidden_dim: The dimensionality of the hidden state.
        - context_dim: The dimensionality of the context tensor.
        """
        super(SpatialTransformer, self).__init__()
        
        # TransformerBlock for spatial transformation
        self.transformer = TransformerBlock(hidden_dim, context_dim)

    def forward(self, x, context=None):
        """
        Forward pass of the SpatialTransformer.

        Parameters:
        - x: Input tensor with shape [batch, channels, height, width].
        - context: Context tensor with shape [batch, context_seq_len, context_dim].

        Returns:
        - x: Output tensor after applying spatial transformation.
        """
        b, c, h, w = x.shape
        x_in = x

        # Combine the spatial dimensions and move the channel dimension to the end
        x = rearrange(x, "b c h w -> b (h w) c")

        # Apply the sequence transformer
        x = self.transformer(x, context)

        # Reverse the process
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        # Residue connection
        return x + x_in

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        """
        Initialize the ConvNextBlock.

        Parameters:
        - in_channels: The number of input channels.
        - out_channels: The number of output channels.
        - embed_dim: The dimensionality of the Gaussian random feature embeddings.
        """
        super(ConvNextBlock, self).__init__()

        # Convolutional layer with kernel size 3x3
        self.conv1 = nn.Conv2d(in_channels, in_channels, 7, stride=1, padding=3)

        # Gaussian random feature embedding layer
        self.dense = nn.Sequential(
            nn.Linear(embed_dim, in_channels)
        )

        self.conv_net = nn.Sequential(
            nn.GroupNorm(1, num_channels=in_channels),
            nn.Conv2d(in_channels, 3*in_channels, 3, stride=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, num_channels=3*in_channels),
            nn.Conv2d(3*in_channels, out_channels, 3, stride=1, padding=1),
        )

        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, stride=1)


    def forward(self, x, t):
        """
        Forward pass of the ConvNextBlock.

        Parameters:
        - x: Input tensor.
        - t: Time tensor.

        Returns:
        - h: Output tensor after passing through the ConvNextBlock.
        """
        
        # Obtain the Gaussian random feature embedding for t
        embed = self.dense(t)
        embed = embed[..., None, None]

        # Convolutional layer
        h = self.conv1(x) + embed

        # Convolutional neural network
        h = self.conv_net(h)

        # Skip connection
        h = h + self.skip_conv(x)

        return h



class UNet_Tranformer(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(
        self, 
        marginal_prob_std, 
        channels=[32, 64, 128, 256], 
        embed_dim=256,
        in_channels=1,
        imsize=64,
        text_dim=None, 
    ):
        """
        Initialize a time-dependent score-based network.

        Parameters:
        - marginal_prob_std: A function that takes time t and gives the standard deviation
          of the perturbation kernel p_{0t}(x(t) | x(0)).
        - channels: The number of channels for feature maps of each resolution.
        - embed_dim: The dimensionality of Gaussian random feature embeddings of time.
        - text_dim: The embedding dimension of text/digits.
        """
        super().__init__()

        self.in_channels = in_channels
        self.imsize = imsize


        imsizes = [self.imsize // 2**i for i in range(5)]
        seq_lens = [i**2 for i in imsizes]
        k_values = [i // 2 for i in seq_lens]
        heads = [1, 1, 2, 2, 4]

        # Gaussian random feature embedding layer for time
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        # Encoding layers where the resolution decreases
        self.enc_conv1 = ConvNextBlock(in_channels, channels[0], embed_dim)
        # self.enc_attn1 = SpatialLinformerAttention(
        #     seq_len=seq_lens[0],
        #     dim=channels[0],
        #     n_heads=heads[0],
        #     k=k_values[0]
        # )
        self.down1 = nn.Conv2d(channels[0], channels[0], 4, stride=2, padding=1)

        self.enc_conv2 = ConvNextBlock(channels[0], channels[1], embed_dim)
        # self.enc_attn2 = SpatialLinformerAttention(
        #     seq_len=seq_lens[1],
        #     dim=channels[1],
        #     n_heads=heads[1],
        #     k=k_values[1]
        # )
        self.down2 = nn.Conv2d(channels[1], channels[1], 4, stride=2, padding=1)

        self.enc_conv3 = ConvNextBlock(channels[1], channels[2], embed_dim)
        self.enc_attn3 = SpatialLinformerAttention(
            seq_len=seq_lens[2],
            dim=channels[2],
            n_heads=heads[2],
            k=k_values[2]
        )
        self.down3 = nn.Conv2d(channels[2], channels[2], 4, stride=2, padding=1)

        self.enc_conv4 = ConvNextBlock(channels[2], channels[3], embed_dim)
        self.enc_attn4 = SpatialLinformerAttention(
            seq_len=seq_lens[3],
            dim=channels[3],
            n_heads=heads[3],
            k=k_values[3]
        )
        self.down4 = nn.Conv2d(channels[3], channels[3], 4, stride=2, padding=1)


        # Bottle neck
        self.bottle_neck1 = ConvNextBlock(channels[3], 2*channels[3], embed_dim)

        self.bottle_neck_attn = SpatialLinformerAttention(
            seq_len=seq_lens[4],
            dim=2*channels[3],
            n_heads=heads[4],
            k=k_values[4]
        )
        
        # self.bottle_neck2 = ConvNextBlock(2*channels[3], channels[3], embed_dim)


        # Decoding layers where the resolution increases
        self.dec_conv1 = ConvNextBlock(2*channels[3], channels[3], embed_dim)
        self.dec_attn1 = SpatialLinformerAttention(
            seq_len=seq_lens[4],
            dim=channels[3],
            n_heads=heads[4],
            k=k_values[4]
        )
        self.up1 = nn.ConvTranspose2d(channels[3], channels[3], 4, stride=2, padding=1)

        
        self.dec_conv2 = ConvNextBlock(2*channels[3], channels[2], embed_dim)
        self.dec_attn2 = SpatialLinformerAttention(
            seq_len=seq_lens[3],
            dim=channels[2],
            n_heads=heads[3],
            k=k_values[3]
        )
        self.up2 = nn.ConvTranspose2d(channels[2], channels[2], 4, stride=2, padding=1)

        self.dec_conv3 = ConvNextBlock(2*channels[2], channels[1], embed_dim)
        # self.dec_attn3 = SpatialLinformerAttention(
        #     seq_len=seq_lens[2],
        #     dim=channels[1],
        #     n_heads=heads[2],
        #     k=k_values[2]
        # )
        self.up3 = nn.ConvTranspose2d(channels[1], channels[1], 4, stride=2, padding=1)

        self.dec_conv4 = ConvNextBlock(2*channels[1], channels[1], embed_dim)
        # self.dec_attn4 = SpatialLinformerAttention(
        #     seq_len=seq_lens[1],
        #     dim=channels[0],
        #     n_heads=heads[1],
        #     k=k_values[1]
        # )
        self.up4 = nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1)

        self.final_layer1 = nn.Conv2d(2*channels[0], channels[0], 3, stride=1, padding=1)
        self.final_layer2 = nn.Conv2d(channels[0], in_channels, 3, stride=1, padding=1, bias=False)

        # The swish activation function
        self.act = nn.GELU()
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        """
        Forward pass of the UNet_Transformer model.

        Parameters:
        - x: Input tensor.
        - t: Time tensor.
        - y: Target tensor.

        Returns:
        - h: Output tensor after passing through the UNet_Transformer architecture.
        """
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.time_embed(t))

        # Encoding path
        h1 = self.enc_conv1(x, embed)
        h = self.down1(h1)
        
        h2 = self.enc_conv2(h, embed)
        h = self.down2(h2)
        
        h3 = self.enc_conv3(h, embed)
        h3 = self.enc_attn3(h3)
        h = self.down3(h3)

        h4 = self.enc_conv4(h, embed)
        h4 = self.enc_attn4(h4)
        h = self.down4(h4)



        # Bottle neck
        h = self.bottle_neck1(h, embed)
        h = self.bottle_neck_attn(h)
        # h = self.bottle_neck2(h, embed)


        # Decoding path
        h = self.dec_conv1(h, embed)
        h = self.dec_attn1(h)
        h = self.up1(h)

    
        h = torch.cat([h, h4], dim=1)
        h = self.dec_conv2(h, embed)
        h = self.dec_attn2(h)
        h = self.up2(h)

        h = torch.cat([h, h3], dim=1)
        h = self.dec_conv3(h, embed)
        h = self.up3(h)
        
        h = torch.cat([h, h2], dim=1)
        h = self.dec_conv4(h, embed)
        h = self.up4(h)

        h = torch.cat([h, h1], dim=1)
        h = self.final_layer1(h)
        h = self.act(h)
        h = self.final_layer2(h)

        # Normalize output
        if self.marginal_prob_std is not None:
            h = h / self.marginal_prob_std(t)[:, None, None, None]
            
        return h
    

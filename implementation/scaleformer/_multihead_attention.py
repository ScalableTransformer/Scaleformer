import torch
import math
from typing import Optional, Union, Tuple


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int):
        """
        Parameters
        ----------
        projection_dim : int
            the dimension of the projection space for the feature vectors
        n_heads : int
            the number of different projection at each stage of the transformer
        linear_complexity : bool
            if True, a variant of the attention mechanism is used that has
            linear coplexity with the sequence length of tokens
        """
        super().__init__()
        self.n_heads = n_heads
        self.projection_dim = projection_dim
        dim = projection_dim*n_heads
        self.query = torch.nn.Linear(dim, dim, bias=False)
        self.key = torch.nn.Linear(dim, dim, bias=False)
        self.value = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, Y: torch.Tensor, X: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        Forward pass of the multihead attention module.
        Apply masked attention, followed by dropout, and batch normalization
        Parameters
        ----------
        Y : torch.Tensor
            the tensor to transform
            tensor of shape (N, Lq, D) with
            * N the number of sentences to treat
            * Lq the sequence length of the query
            * D the embedding dimension
        X : torch.Tensor
            the tensor to depend on
            tensor of shape (N, Lk, D) with
            * N the number of sentences to treat
            * Lk the sequence length of the key
            * D the embedding dimension
        mask : torch.Tensor or None
            the mask, tensor of booleans of shape (Lq, Lk), where attention
            is set to -infinity
        Returns
        -------
        torch.Tensor :
            tensor of shape (N, Lq, D)
        """
        N, Lq, _ = Y.shape
        N, Lk, _ = X.shape
        # project into 'n_heads' different subspaces
        q = self.query(Y).reshape(N, Lq, self.n_heads, self.projection_dim)
        k = self.key(X).reshape(N, Lk, self.n_heads, self.projection_dim)
        v = self.value(X).reshape(N, Lk, self.n_heads, self.projection_dim)
        # compute attention dot product
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attention, _ = self._scaled_dot_product_attention(q, k, v, mask)
        attention = attention.transpose(2, 1).reshape(N, Lq, -1)
        return attention

    def _multihead_attention(self, query: torch.Tensor, key: torch.Tensor,
                             mask: Union[Optional[torch.Tensor], bool]
                             ) -> torch.Tensor:
        """
        Apply multihead attention.
        Same inputs/outputs types/shapes as the forward pass
        """
        

    def _scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor,
                                      v: torch.Tensor,
                                      mask: Optional[torch.Tensor]
                                      ) -> Tuple[torch.Tensor]:
        """
        Apply scaled dot product attention to a batch of 'N' sentences pairs,
        with 'H' the number of heads, and 'D' the projection dimension.
        The query is a sequence of length 'Lq', and the key is
        a sequence of length 'Lk'.
        This is the original attention mechanism described in the 2017 paper:
            'Attention is all you need'
            https://arxiv.org/pdf/1706.03762.pdf
        Parameters
        ----------
        q : torch.Tensor
            query tensor of shape (N, H, Lq, D)
        k : torch.Tensor
            key tensor of shape (N, H, Lk, D)
        v : torch.Tensor
            value tensor of shape (N, H, Lk, D)
        mask : torch.Tensor or None
            tensor of booleans of shape (Lq, Lk)
        Returns
        -------
        tuple of torch.Tensors:
            a tuple of (attention, score)
        """
        Lk = k.shape[-2]
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Lk)
        if mask is not None:
            score = score.masked_fill(mask, -1.0E10)
        score = torch.softmax(score, dim=-1)
        attention = torch.matmul(score, v)
        return attention, score

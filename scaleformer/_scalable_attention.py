# -*- coding: utf-8 -*-
from typing import Optional
from typing import Tuple
import math
import torch
import torch.nn.functional as F
from ._functions import mask_chronological


class ScalableAttention(torch.nn.Module):

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
                RPE: Optional[torch.Tensor] = None, mask: bool = False):
        """
        Forward pass of the multihead attention module.
        Apply masked attention, followed by dropout, and batch normalization

        Parameters
        ----------
        Y : torch.Tensor
            tensor of shape (N, Lq, D) with
            * N the number of sentences to treat
            * Lq the sequence length of the query
            * D the embedding dimension
        X : torch.Tensor
            tensor of shape (N, Lk, D) with
            * N the number of sentences to treat
            * Lk the sequence length of the key
            * D the embedding dimension
        RPE : torch.Tensor or None
            tensor of shape (2R+1, d) of relative position  embedding, with:
            * R the radius of the horizon
            * d the projected dimension
        mask: bool
            If True returns unidirectional attention

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
        attention = self._normalized_scalable_attention(q, k, v, RPE, mask)
        attention = attention.transpose(2, 1).reshape(N, Lq, -1)
        return attention

    def _phi(self, X: torch.Tensor) -> torch.Tensor:
        """
        The kernel function for kernelized attention
        """
        return F.elu(X) + 1

    def _normalized_scalable_attention(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            RPE: Optional[torch.Tensor], masked: bool = False
            ) -> torch.Tensor:
        """
        This is an alternative attention function.
        'softmax(Q*Kt)*V' becomes 'phi(Q)*phi(K)*V'
        This allow to chose operation order for the 3 matrices multiplications
        and have linear (instead of square) complexity with sequence length or
        projection dim depending on the situation.


        This implementation reproduces this paper :
            'Efficient Attention: Attention with Linear Complexities'
            https://arxiv.org/pdf/1812.01243.pdf
        The masking for causal-attention is inspired from the paper :
            'Rethinking Attention with Performers', Annexe B.1
            https://arxiv.org/pdf/2009.14794.pdf

        Parameters
        ----------
        q : torch.Tensor
            query tensor of shape (N, H, Lq, D)
        k : torch.Tensor
            key tensor of shape (N, H, Lk, D)
        v : torch.Tensor
            value tensor of shape (N, H, Lk, D)
        RPE : torch.Tensor or None
            tensor of shape (2*R+1, D) with R the attention horizon
            for relative positional encoding
        masked : bool
            If True, returns unidirectional attention

        Returns
        -------
        tuple of (torch.tensor, None) :
            the attention score cannot always be calculated,
            so None is returned instead
        """
        pq = self._phi(q)
        pk = self._phi(k)
        _, _, _, D = pq.shape
        A_num = self._scalable_attention(pq, pk, v, RPE, masked)
        v_scale = torch.ones(*(list(v.shape)[:-1] + [1]), dtype=v.dtype,
                             device=v.device)
        v_scale = v_scale.expand(-1, -1, -1, D)
        A_denom = self._scalable_attention(pq, pk, v_scale, RPE, masked)
        return A_num / A_denom

    def _scalable_attention(
            self, pq: torch.Tensor, pk: torch.Tensor, v: torch.Tensor,
            RPE: Optional[torch.Tensor], masked: bool = False
            ) -> Tuple[torch.Tensor, None]:
        """
        Dynamically choose the algorithm to compute the attention/RPE

        Parameters
        ----------
        pq : torch.Tensor
            query tensor of shape (N, H, Lq, D)
        pk : torch.Tensor
            key tensor of shape (N, H, Lk, D)
        v : torch.Tensor
            value tensor of shape (N, H, Lk, D)
        RPE : torch.Tensor or None
            tensor of shape (2*R+1, D) with R the attention horizon
            for relative positional encoding
        masked : bool
            If True, returns unidirectional attention

        Returns
        -------
        tuple of (torch.tensor, None) :
            the attention score cannot always be calculated,
            so None is returned instead
        """
        N, H, Lq, D = pq.shape
        _, _, Lk, _ = pk.shape
        # kernelized attention
        naive_cost = Lq * Lk
        if masked:
            linear_cost = D**2 * max(Lq, Lk)
            if naive_cost <= linear_cost:
                A = self._naive_kernelized_attention(pq, pk, v, masked)
            else:
                A = self._linear_kernelized_attention(pq, pk, v, masked)
        else:
            linear_cost = D**2
            if naive_cost <= linear_cost:
                A = self._naive_kernelized_attention(pq, pk, v, masked)
            else:
                A = self._linear_kernelized_attention(pq, pk, v, masked)
        # relative positional encoding
        if RPE is not None:
            E, _ = RPE.shape
            R = E//2
            naive_cost = Lq*Lk
            linear_cost = Lq*(2*R + 1 + 4)
            if naive_cost <= linear_cost:
                rpe = self._naive_RPE(pq, RPE, v, masked)
            else:
                rpe = self._linear_RPE(pq, RPE, v, masked)
            A = A+rpe
        return A

    def _naive_kernelized_attention(self, pq: torch.Tensor, pk: torch.Tensor,
                                    v: torch.Tensor, masked: bool
                                    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pq : torch.Tensor
            query tensor of shape (N, H, Lq, D)
        pk : torch.Tensor
            key tensor of shape (N, H, Lk, D)
        v : torch.Tensor
            value tensor of shape (N, H, Lk, D)
        masked : bool
            If True, returns unidirectional attention
        """
        N, H, Lq, D = pq.shape
        _, _, Lk, _ = pk.shape
        left = torch.matmul(pq, pk.transpose(-2, -1))
        if masked:
            mask = mask_chronological(Lq, Lk, left.device)
            left = left.masked_fill(mask, 0.)
        return torch.matmul(left, v)

    def _linear_kernelized_attention(self, pq: torch.Tensor, pk: torch.Tensor,
                                     v: torch.Tensor, masked: bool
                                     ) -> torch.Tensor:
        """
        Parameters
        ----------
        pq : torch.Tensor
            query tensor of shape (N, H, Lq, D)
        pk : torch.Tensor
            key tensor of shape (N, H, Lk, D)
        v : torch.Tensor
            value tensor of shape (N, H, Lk, D)
        masked : bool
            If True, returns unidirectional attention
        """
        if masked:
            N, H, Lq, D = pq.shape
            _, _, Lk, _ = pk.shape
            unrolled_left = pq - torch.cat([torch.zeros(N, H, 1, D),
                                           pq[..., :-1, :]], dim=-2)
            unrolled_right = pk.unsqueeze(-2)*v.unsqueeze(-1)
            result = torch.cumsum(torch.matmul(unrolled_left.unsqueeze(-2),
                                               unrolled_right),
                                  dim=-2).squeeze(-2)
        elif masked:
            N, H, Lq, D = pq.shape
            _, _, Lk, _ = pk.shape
            # unrolled_right_1 = k.unsqueeze(-2)*v.unsqueeze(-1)
            # right = torch.cumsum(unrolled_right, dim=-3)
            unrolled_right = torch.einsum("nhli, nhlj -> nhijl", pk, v)
            right = torch.cumsum(unrolled_right, dim=-1).permute(0, 1, 4, 2, 3)
            if Lq > Lk:
                right = torch.cat([right, right[..., -1:, :, :].expand(
                    N, H, Lq-Lk, D, D)], dim=-3)
            elif Lk > Lq:
                right = right[..., :Lq, :, :]
            result = torch.matmul(pq.unsqueeze(-2), right).squeeze(-2)
        else:
            right = torch.matmul(pk.transpose(-2, -1), v)
            result = torch.matmul(pq, right)
        return result

    def _linear_RPE(self, pq: torch.Tensor, RPE_matrix: torch.Tensor,
                    v: torch.Tensor, masked: bool = False) -> torch.Tensor:
        """
        Performs the 'Relative Positional Encoding' part of the attention
        as described in:

        'Self-Attention with Relative Position Representations'
        https://arxiv.org/pdf/1803.02155.pdf


        implementation is adapted from the second strategy described in:
        'Translational Equivariance in Kernelizable Attention'
        https://arxiv.org/pdf/2102.07680.pdf

        Parameters
        ----------
        pq : torch.Tensor
            tensor of shape (N, H, Lq, D)
        RPE_matrix : torch.Tensor
            tensor of shape (2*R+1, D)
        v : torch.Tensor
            tensor of shape (N, H, Lk, D)

        Returns
        -------
        torch.Tensor :
            tensor of shape (N, H, Lq, 2*R+1)
        """
        N, H, Lq, D = pq.shape
        E, _ = RPE_matrix.shape
        N, H, Lk, _ = v.shape
        R = (E-1)//2
        assert E % 2 == 1 and R > 0
        # compute the dot product between each query and each RPE embedding
        # 'scores' is of shape (N, H, Lq, 2*R+1)
        scores = torch.einsum("nhij, kj -> nhik", pq, RPE_matrix)
        # split the scores
        before = scores[..., 0:1]  # the score of distance -R and lower
        after = scores[..., -1:]  # the score of the distance R and upper
        if masked:
            # The score of the distances -(R-1) to 0
            window = scores[..., 1:R+1].unsqueeze(-2)
        else:
            # The scores of the distances -(R-1) to (R-1)
            window = scores[..., 1:-1].unsqueeze(-2)
        # sum of the value vectors on the left of the RPE horizon
        head = torch.zeros((N, H, min(R, Lq), D), device=pq.device)
        core = v[..., :min(max(0, Lq-R), Lk), :].cumsum(dim=-2)
        content = [head, core]
        if Lq-Lk-R > 0:
            bottom = core[..., -1:, :].repeat(1, 1, max(0, Lq-Lk-R), 1)
            content.append(bottom)
        left = torch.cat(content, dim=-2)
        if masked:
            # weighted sum of the values in the first half of the window
            slider = torch.cat([torch.zeros((N, H, max(0, R-1), D),
                                            device=pq.device),
                                v, torch.zeros((N, H, max(0, Lq-Lk), D),
                                               device=pq.device)],
                               dim=-2)
            L = slider.shape[-2]
            slider = slider.as_strided((N, H, Lq, R, D),
                                       (H*L*D, L*D, D, D, 1))
            center = torch.matmul(window, slider).squeeze(-2)
            # returns the weighted sum of value vectors
            return before*left + center
        else:
            # sum of the value vectors on the right of the RPE horizon
            stop = min(Lq+R, Lk)
            core = (v[..., R-1:stop, :].sum(dim=-2).unsqueeze(-2)
                    - v[..., R-1:stop-1, :].cumsum(dim=-2))
            bottom = torch.zeros((N, H, max(0, Lq-max(0, Lk-R)), D),
                                 device=pq.device)
            right = torch.cat([core, bottom], dim=-2)
            # weighted sum of the values in the horizon
            slider = torch.cat([torch.zeros((N, H, max(0, R-1), D),
                                            device=pq.device),
                                v, torch.zeros((N, H, max(0, Lq-(Lk-R)), D),
                                               device=pq.device)],
                               dim=-2)
            L = slider.shape[-2]
            slider = slider.as_strided((N, H, Lq, E-2, D),
                                       (H*L*D, L*D, D, D, 1))
            center = torch.matmul(window, slider).squeeze(-2)
            # returns the weighted sum of value vectors
            return before*left + center + after*right

    def _naive_RPE(self, pq: torch.Tensor, RPE_matrix: torch.Tensor,
                   v: torch.Tensor, masked: bool = False) -> torch.Tensor:
        """
        naive implementation of _linear_RPE
        for testing only
        """
        N, H, Lq, D = pq.shape
        E, D = RPE_matrix.shape
        N, H, Lk, D = v.shape
        R = (E-1)//2
        assert R >= 0
        # compute the dot product between each query and each RPE embedding
        # 'scores' is of shape (N, H, Lq, 2*R+1)
        scores = torch.einsum("nhij, kj -> nhik", pq, RPE_matrix)
        # expands score matrix of shape (N, H, Lq, Lk)
        indexes = torch.tensor([[R-i+j for j in range(Lk)] for i in range(Lq)],
                               dtype=torch.long, device=pq.device).clip(0, 2*R)
        indexes = indexes.reshape(1, 1, Lq, Lk).expand(N, H, Lq, Lk)
        scores = torch.gather(scores, -1, indexes)
        # masks the top right corner of the matrix if needed
        if masked:
            mask = mask_chronological(Lq, Lk, pq.device)
            scores = scores.masked_fill(mask, 0.)
        # returns the weighted sum of value vectors
        return torch.matmul(scores, v)

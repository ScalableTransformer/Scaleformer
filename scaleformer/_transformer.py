# -*- coding: utf-8 -*-
from os import error
from typing import Union
from typing import Optional
from typing import Tuple
import math
import torch
from ._tokenizer import BytePairEncoder
from ._functions import positional_encoding
from ._functions import mask_chronological
from ._functions import strings_to_tensor
from ._multihead_attention import MultiHeadAttention
from ._scalable_attention import ScalableAttention


class TransformerEncoderStage(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None,
                 activation: str = "relu", scalable: bool = True):
        super().__init__()
        Attention = ScalableAttention if scalable else MultiHeadAttention
        dim = projection_dim * n_heads
        self.activation = getattr(torch, activation)
        self.self_attention = Attention(projection_dim, n_heads)
        self.intermediate_norm = torch.nn.BatchNorm1d(dim)
        self.intermediate_dropout = torch.nn.Dropout(dropout)
        self.expand = torch.nn.Linear(dim, dim*4)
        self.contract = torch.nn.Linear(dim*4, dim)
        self.out_dropout = torch.nn.Dropout(dropout)
        self.out_norm = torch.nn.BatchNorm1d(dim)

    def forward(self, X, mask: Optional[torch.Tensor] = None):
        """
        Parameter
        ---------
        X : torch.Tensor
            Tensor of shape (N, L, F) with
            * N sentences count
            * L sequence length
            * F number of features
        mask : torch.Tensor or None
            mask to apply for the attention
        Returns
        -------
        torch.Tensor
            tensor of shape (N, L, F)
        """
        if isinstance(self.self_attention, ScalableAttention):
            mask = (mask is not None)
        X = X.to(self.device)
        N, L, _ = X.shape
        input = X.reshape(N*L, -1)
        X = self.self_attention(X, X, mask=mask).reshape(N*L, -1)
        X = self.intermediate_norm(self.intermediate_dropout(X) + input)
        input = X
        X = self.activation(self.expand(X))
        X = self.out_dropout(self.contract(X))
        X = self.out_norm(X + input)
        return X.reshape(N, L, -1)

    @property
    def device(self) -> torch.device:
        return self.self_attention.key.weight.device


class TransformerDecoderStage(torch.nn.Module):

    def __init__(self, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu",
                 scalable: bool = True):
        super().__init__()
        Attention = ScalableAttention if scalable else MultiHeadAttention
        dim = projection_dim * n_heads
        self.activation = getattr(torch, activation)
        self.masked_attention = Attention(projection_dim, n_heads)
        self.first_dropout = torch.nn.Dropout(dropout)
        self.first_norm = torch.nn.BatchNorm1d(dim)
        self.attention = Attention(projection_dim, n_heads)
        self.second_dropout = torch.nn.Dropout(dropout)
        self.second_norm = torch.nn.BatchNorm1d(dim)
        self.expand = torch.nn.Linear(dim, 4*dim)
        self.contract = torch.nn.Linear(4*dim, dim)
        self.out_dropout = torch.nn.Dropout(dropout)
        self.out_norm = torch.nn.BatchNorm1d(dim)

    def forward(self, encoded, Y, mask: Optional[torch.Tensor] = None):
        """
        Parameter
        ---------
        encoded : torch.Tensor
            Tensor of shape (N, L, F)
        Y : torch.Tensor
            Tensor of shape (N, L, F)
        mask : torch.Tensor or None
            mask to apply for the attention
        Returns
        -------
        torch.Tensor
            tensor of shape (N, L, F)
        """
        if isinstance(self.attention, ScalableAttention):
            mask = (mask is not None)
        encoded = encoded.to(self.device)
        Y = Y.to(self.device)
        N, L, _ = Y.shape
        input = Y.reshape(N*L, -1)
        Y = self.masked_attention(Y, Y, mask=mask).reshape(N*L, -1)
        Y = self.first_norm(self.first_dropout(Y) + input).reshape(N, L, -1)
        input = Y.reshape(N*L, -1)
        Y = self.attention(Y, encoded, mask=None).reshape(N*L, -1)
        Y = self.second_norm(self.second_dropout(Y) + input)
        input = Y
        Y = self.out_dropout(self.contract(self.activation(self.expand(Y))))
        Y = self.out_norm(Y + input)
        return Y.reshape(N, L, -1)

    @property
    def device(self) -> torch.device:
        return self.masked_attention.key.weight.device


class TransformerEncoder(torch.nn.Module):

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu",
                 scalable: bool = True):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        for stage in range(n_stages):
            self.stages.append(TransformerEncoderStage(projection_dim, n_heads,
                               dropout=dropout, activation=activation,
                               scalable=scalable))

    def forward(self, X, mask=None):
        """
        same as TransformerEncoderStage.forward
        """
        for stage in self.stages:
            X = stage(X, mask=mask)
        return X


class TransformerDecoder(torch.nn.Module):

    def __init__(self, n_stages: int, projection_dim: int, n_heads: int,
                 dropout: Optional[float] = None, activation: str = "relu",
                 scalable: bool = True):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        for stage in range(n_stages):
            self.stages.append(TransformerDecoderStage(projection_dim, n_heads,
                               dropout=dropout, activation=activation,
                               scalable=scalable))

    def forward(self, encoded, Y, mask: Optional[torch.Tensor] = None):
        """
        same as TransformerDecoderStage.forward
        """
        for stage in self.stages:
            Y = stage(encoded, Y, mask=mask)
        return Y


class Transformer(torch.nn.Module):

    def __init__(self, tokenizer_in: BytePairEncoder,
                 tokenizer_out: BytePairEncoder,
                 n_stages: int, projection_dim: int, n_heads: int,
                 dropout: float = 0., activation: str = "relu",
                 scalable: bool = True):
        """
        Parameters
        ----------
        tokenizer_in : BytePairEncoder
            input sentences tokenizer
        tokenizer_out : BytePairEncoder
            target sentences tokenizer
        n_stages : int
            number of stages of the transformer
        projection_dim : int
            dimension of the Q, K, V projected spaces
        n_heads : int
            number of heads
        dropout : float
            dropout probability
        activation : str
            name of the activation function of the feed forwards
        scalable : bool
            if True, use ScalableAttention
        """
        super().__init__()
        self.tokenizer_in = tokenizer_in
        self.tokenizer_out = tokenizer_out
        vocab_in = tokenizer_in.vocabulary
        vocab_out = tokenizer_out.vocabulary
        # embedding in
        self.embedding_in = torch.nn.Embedding(len(vocab_in),
                                               projection_dim*n_heads)
        self.dropout_in = torch.nn.Dropout(dropout)
        # encoder
        self.encoder = TransformerEncoder(n_stages, projection_dim, n_heads,
                                          dropout=dropout,
                                          activation=activation,
                                          scalable=scalable)
        # embedding out
        self.embedding_out = torch.nn.Embedding(len(vocab_out),
                                                projection_dim*n_heads)
        self.dropout_out = torch.nn.Dropout(dropout)
        # decoder
        self.decoder = TransformerDecoder(n_stages, projection_dim, n_heads,
                                          dropout=dropout,
                                          activation=activation,
                                          scalable=scalable)
        # output
        self.output = torch.nn.Linear(projection_dim*n_heads, len(vocab_out))

    def predict(self, string: str, max_tokens=100):
        """
        predict an output of the model

        Parameters
        ----------
        X : torch.Tensor
            tensor of longs of tokens indexes of shape (N, L)
        max_tokens : int
            max number of tokens generated
        """
        self.eval()
        START = self.tokenizer_out.START
        END = self.tokenizer_out.END
        device = self.embedding_in.weight.device
        X = strings_to_tensor([string], self.tokenizer_in).to(device)
        with torch.no_grad():
            encoded = self(X)
            # Y is initialized as a single 'start of sentence' character
            Y = torch.full([len(X), 1], START,
                           dtype=torch.long, device=X.device)
            for _ in range(max_tokens):
                res = self.decode(encoded, Y)
                res = torch.argmax(res, dim=-1)
                index = res[:, -1:]
                Y = torch.cat([Y, index], dim=-1)
                if (index == END).any() or Y.shape[1] > max_tokens:
                    break
            else:
                Y = torch.cat([Y, index], dim=-1)
        Y = Y.detach().cpu().numpy().reshape(-1)
        vocab = self.tokenizer_out.vocabulary
        result = [vocab[i] for i in Y]
        return b"".join(b for b in result if isinstance(b, bytes)
                        ).decode("utf-8", errors="ignore")

    def forward(self, X):
        return self.encode(X)

    def encode(self, X):
        """
        performs the encoding part of the network
        Parameters
        ----------
        X : torch.Tensor
            tensor of embedded input tokens
            tensor of longs of shape (N, L) with:
            * N : number of sentences
            * L : tokens per sentence
        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, L, D)
        """
        device = self.embedding_in.weight.device
        X = X.to(device)
        X = self.dropout_in(positional_encoding(self.embedding_in(X)))
        return self.encoder(X)

    def decode(self, encoded, Y):
        """
        performs the decoding part of the network:
        for each of the already predicted tokens, predict the next token.
        Parameters
        ----------
        encoded : torch.Tensor
            tensor of encoded inputs
            tensor of floats of shape (N, Lx, D) with:
            * N : number of sentences
            * Lx : tokens per sentence in the input language
            * D : embedding dim
        Y : torch.Tensor
            tensor of the already predicted tokens
            tensor of long of shape (N, Ly) with:
            * N : number of sentences
            * Ly : tokens per sentence in the output language
        Returns
        -------
        torch.Tensor :
            tensor of floats of shape (N, Ly, D)
        """
        Y = Y.to(self.embedding_out.weight.device)
        _, Lq = Y.shape
        mask = mask_chronological(Lq, Lq, device=Y.device)
        Y = self.dropout_out(positional_encoding(self.embedding_out(Y)))
        Y = self.decoder(encoded, Y, mask=mask)
        return self.output(Y)

    def loss(self, X: torch.Tensor, Y: torch.Tensor,
             class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        compute the loss of the model
        """
        encoded = self.encode(X)
        y_pred = self.decode(encoded, Y[:, :-1])
        vocab = self.tokenizer_out.vocabulary
        PAD = self.tokenizer_out.PAD
        class_weights = [0 if t == PAD else 1 for t in vocab]
        class_weights = torch.tensor(class_weights, dtype=torch.float32,
                                     device=y_pred.device)
        Y = Y.to(y_pred.device)
        return torch.nn.functional.cross_entropy(y_pred.transpose(1, 2),
                                                 Y[:, 1:], class_weights)

    def to_multi_GPU(self):
        """
        scatter the model across multiple GPUs
        """
        n_GPUs = torch.cuda.device_count()
        n_layers = len(self.encoder.stages) + len(self.decoder.stages)
        encoder_devices = [torch.device(f"cuda:{i}") for i in
                           [(i*n_GPUs)//n_layers
                            for i in range(len(self.encoder.stages))]]
        decoder_devices = [torch.device(f"cuda:{i}") for i in
                           [((i+len(self.encoder.stages))*n_GPUs)//n_layers
                            for i in range(len(self.decoder.stages))]]
        # embedding in
        device = encoder_devices[0]
        self.embedding_in.to(device)
        self.dropout_in.to(device)
        # encoder
        for stage, device in zip(self.encoder.stages, encoder_devices):
            stage.to(device)
        # embedding out
        device = decoder_devices[0]
        self.embedding_out.to(device)
        self.dropout_out.to(device)
        # decoder
        for stage, device in zip(self.decoder.stages, decoder_devices):
            stage.to(device)
        # output
        device = decoder_devices[-1]
        self.output.to(device)

    def to_CPU(self):
        """
        gather the model to CPU memory
        """
        self.to(torch.device("cpu"))

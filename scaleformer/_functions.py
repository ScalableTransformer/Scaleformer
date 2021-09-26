# -*- coding: utf-8 -*-
from typing import Optional
from typing import Tuple
from typing import List
import copy
import torch
import matplotlib.pyplot as plt
from ._tokenizer import BytePairEncoder


def positional_encoding(X: torch.Tensor) -> torch.Tensor:
    """
    Performs positional encoding on the input, in the
    "Attention is all you need" paper fashion.

    Parameters
    ----------
    X : torch.Tensor
        tensor of shape (..., D), with D the embedding dimension

    Returns
    -------
    torch.Tensor:
        tensor of shape (..., D)
    """
    shape = X.shape
    X = X.reshape(-1, shape[-1])
    N, D = X.shape
    pe = torch.zeros(N, D, dtype=torch.float, device=X.device)
    position = torch.arange(0, D, dtype=torch.float).unsqueeze(0)
    angle = position / 10000**(2*(position//2)/D)
    pe[:, 0::2] = torch.cos(angle[:, 0::2])
    pe[:, 1::2] = torch.sin(angle[:, 1::2])
    X = (X + pe).reshape(shape)
    return X


def mask_chronological(Lq: int, Lk: int, device: torch.device) -> torch.Tensor:
    """
    A mask for transformers training.
    """
    mask = torch.ones(Lq, Lk, dtype=bool, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


def batchify(*args, batch_size: Optional[int] = None,
             n_batches: Optional[int] = None):
    """
    yields 'n_batches' (or less) batches of size 'batch_size'.
    Each batch is a tuple (x, y, ...) subset of (X, Y, ...)
    """
    n = len(args[0])
    if batch_size is None:
        batch_size = n
    if n_batches is None:
        n_batches = n // batch_size
    indexes = torch.randperm(n, device=args[0].device)
    for i in range(n_batches):
        subset = indexes[i*batch_size:(i+1)*batch_size]
        yield tuple(x[subset] for x in args)


def train_loop(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
               train_data: Tuple[torch.Tensor, ...],
               val_data: Tuple[torch.Tensor, ...],
               n_epochs: int,
               batch_size: Optional[int] = None,
               n_batches: Optional[int] = None,
               patience: Optional[int] = None):
    """
    training loop
    """
    train_losses = []
    val_losses = []
    best_loss = float("inf")
    best_epoch = 0
    model_chkpt = copy.deepcopy(model.state_dict())
    opt_chkpt = copy.deepcopy(optimizer.state_dict())
    try:
        for epoch in range(n_epochs):
            # step of the optimization
            optimizer.step()
            optimizer.zero_grad()
            # training loss
            model.train()
            train_loss = []
            for data in batchify(*train_data, batch_size=batch_size,
                                 n_batches=n_batches):
                loss = model.loss(*data)
                loss.backward()
                train_loss.append(loss.item())
            train_loss = sum(train_loss) / len(train_loss)
            train_losses.append(train_loss)
            # validation loss
            model.eval()
            val_loss = []
            for data in batchify(*val_data, batch_size=batch_size,
                                 n_batches=n_batches):
                with torch.no_grad():
                    loss = model.loss(*data)
                val_loss.append(loss.item())
            val_loss = sum(val_loss) / len(val_loss)
            val_losses.append(val_loss)
            # printing
            print(f"Epoch {epoch}: val loss = {val_loss:.3g}, "
                  f"train loss = {train_loss:.3g}", flush=True)
            # model checkpointing
            if val_loss < best_loss:
                best_epoch = epoch
                best_loss = val_loss
                model_chkpt = copy.deepcopy(model.state_dict())
                opt_chkpt = copy.deepcopy(optimizer.state_dict())
            # early stoping
            elif patience is not None and epoch - best_epoch > patience:
                break
    except KeyboardInterrupt:
        pass
    model.load_state_dict(model_chkpt)
    optimizer.load_state_dict(opt_chkpt)
    return train_losses, val_losses, best_epoch


def plot_loss(train_loss, val_loss, best_epoch):
    """
    display losses of the training
    """
    f, ax = plt.subplots()
    ax.plot(range(len(train_loss)), train_loss, label="training")
    ax.plot(range(len(val_loss)), val_loss, label="validation")
    ax.axvline(best_epoch)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    plt.show()


def strings_to_tensor(strings: List[str], tokenizer: BytePairEncoder):
    """
    converts strings into a tensor
    """
    encoded = [tokenizer.encode(s) for s in strings]
    L_padding = max(len(e) for e in encoded)
    padded = [[tokenizer.START] + e + [tokenizer.END]
              + [tokenizer.PAD]*(L_padding - len(e)) for e in encoded]
    return torch.tensor(padded, dtype=torch.long)

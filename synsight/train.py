from argparse import ArgumentParser
import numpy as np
from time import time
import torch
from torch.optim import Adam

from .model import Encoder
from .source import Source


def parse_flags():
    x = ArgumentParser()
    x.add_argument('--samples', type=str, required=True)
    x.add_argument('--size', type=int, required=True)
    x.add_argument('--val_frac', type=float, default=0.1)
    x.add_argument('--tqdm', type=int, default=1)
    x.add_argument('--num_epochs', type=int, default=100)
    x.add_argument('--rounds_per_epoch', type=int, default=20)
    x.add_argument('--train_per_round', type=int, default=5)
    x.add_argument('--test_per_round', type=int, default=1)
    x.add_argument('--batch_size', type=int, default=1024)
    x.add_argument('--device', type=str, default='cuda:0')
    x.add_argument('--body_channels', type=int, default=64)
    x.add_argument('--embed_dim', type=int, default=64)
    return x.parse_args()


def is_match(a, b):
    return (a * b).mean()


def get_loss(y):
    sim_loss = is_match(y[:, 0, :], y[:, 1, :])
    dis_loss = is_match(y[:, 0, :], y[:, 2, :])
    return dis_loss - sim_loss


def mean(x):
    return sum(x) / len(x)


def main(flags):
    t0 = time()
    device = torch.device(flags.device)
    t = time() - t0
    print('Init device: %.3f sec' % t)

    t0 = time()
    samples = np.fromfile(flags.samples, np.uint8)
    t = time() - t0
    print('Dataset to RAM: %.3f sec' % t)

    t0 = time()
    source = Source(samples, flags.size, flags.val_frac, flags.batch_size,
                    device)
    t = time() - t0
    print('Init source: %.3f sec' % t)

    t0 = time()
    encoder = Encoder(source.x.shape[1], flags.body_channels, flags.embed_dim)
    encoder.to(device)
    optimizer = Adam(encoder.parameters())
    t = time() - t0
    print('Init models: %.3f sec' % t) 

    for epoch_id in range(flags.num_epochs):
        t_losses = []
        v_losses = []
        for round_id in range(flags.rounds_per_epoch):
            encoder.train()
            for train_id in range(flags.train_per_round):
                optimizer.zero_grad()
                x = source(True)
                n, triple, c, h, w = x.shape
                x_flat = x.view(-1, c, h, w)
                y_flat = encoder(x_flat)
                y = y_flat.view(n, triple, -1)
                loss = get_loss(y)
                loss.backward()
                optimizer.step()
                t_losses.append(loss.item())
            encoder.eval()
            with torch.no_grad():
                for test_id in range(flags.test_per_round):
                    x = source(False)
                    n, triple, c, h, w = x.shape
                    x_flat = x.view(-1, c, h, w)
                    y_flat = encoder(x_flat)
                    y = y_flat.view(n, triple, -1)
                    loss = get_loss(y)
                    v_losses.append(loss.item())
        t_loss = mean(t_losses)
        v_loss = mean(v_losses)
        print('%4d %7.3f %7.3f' % (epoch_id, t_loss, v_loss))


if __name__ == '__main__':
    main(parse_flags())

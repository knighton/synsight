import numpy as np
import torch


def run_len_encode(x):
    n = len(x)
    is_eq = np.array(x[1:] != x[:-1])
    idx = np.append(np.where(is_eq), n - 1)
    run = np.diff(np.append(-1, idx))
    pos = np.cumsum(np.append(0, run))[:-1]
    return pos, run


class Dataset(object):
    def __init__(self, samples, side, val_frac, batch_size, device):
        bytes_per_sample = 3 * side * side + 4
        samples = samples.reshape(-1, bytes_per_sample)

        b = samples[:, -4:].tobytes()
        y = np.frombuffer(b, np.int32)

        samples = samples[:, :-4]
        samples = samples.reshape(-1, 3, side, side)

        self.x = samples
        self.y = y
        self.side = side
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.device = device

        num_classes = int(y.max()) + 1
        classes = np.arange(num_classes, dtype=np.int32)
        np.random.shuffle(classes)
        num_v_classes = int(num_classes * val_frac)
        self.t_classes = classes[num_v_classes:]
        self.v_classes = classes[:num_v_classes]

        self.class_offsets, self.samples_per_class = run_len_encode(y)
        assert (y[self.class_offsets] == np.arange(num_classes)).all()

    def __call__(self, train):
        split_classes = self.t_classes if train else self.v_classes
        y = np.random.choice(split_classes, (self.batch_size, 2))
        repeat_first = y[:, :1]
        y = np.concatenate([repeat_first, y], 1)
        r = np.random.choice(1 << 32, y.shape)
        ids = self.class_offsets[y] + r % self.samples_per_class[y]
        x = self.x[ids]
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return x / 255

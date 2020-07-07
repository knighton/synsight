from argparse import ArgumentParser
from glob import glob
import numpy as np
import os
from PIL import Image
import struct
from tqdm import tqdm


def parse_flags():
    x = ArgumentParser()

    x.add_argument('--in', type=str, required=True)
    x.add_argument('--splits', type=str, required=True)

    x.add_argument('--out', type=str, required=True)
    x.add_argument('--size', type=int, required=True)

    x.add_argument('--h_min', type=int, default=64)
    x.add_argument('--h_max', type=int, default=1024)
    x.add_argument('--w_min', type=int, default=64)
    x.add_argument('--w_max', type=int, default=1024)
    x.add_argument('--ar_max', type=int, default=1.25)

    x.add_argument('--tqdm', type=int, default=1)

    return x.parse_args()


def collect_files(in_root, splits):
    ff = []
    for split in splits:
        pattern = '%s/%s/*/*.jpg' % (in_root, split)
        gg = glob(pattern)
        assert gg
        ff += gg
    rr = []
    for f in ff:
        split, class_name, image_name = f.split('/')[-3:]
        r = class_name, image_name, split
        rr.append(r)
    return sorted(rr)


class ShapeFilter(object):
    def __init__(self, h_min, h_max, w_min, w_max, ar_max):
        self.h_min = h_min
        self.h_max = h_max
        self.w_min = w_min
        self.w_max = w_max
        self.ar_max = ar_max

    def accepts(self, h, w):
        if not (self.h_min <= h <= self.h_max):
            return False

        if not (self.w_min <= w <= self.w_max):
            return False

        if w <= h:
            ar = h / w
        else:
            ar = w / h
        if not (ar <= self.ar_max):
            return False

        return True


def rescale_image(im, size):
    w, h = im.size
    trim = abs(w - h) // 2
    min_wh = min(w, h)
    if w < h:
        crop = 0, trim, min_wh, trim + min_wh
    else:
        crop = trim, 0, trim + min_wh, min_wh
    im = im.crop(crop)
    return im.resize((size, size))


def main(flags):
    assert not os.path.exists(flags.out)
    os.makedirs(flags.out)

    in_root = getattr(flags, 'in')
    splits = flags.splits.split(',')
    samples = collect_files(in_root, splits)

    shape_filter = ShapeFilter(flags.h_min, flags.h_max, flags.w_min,
                               flags.w_max, flags.ar_max)

    class_names = []
    class_name2class_id = {}
    class_counts = []
    out_f = '%s/samples.npy' % flags.out
    with open(out_f, 'wb') as out:
        for class_name, image_name, split in tqdm(samples, leave=False):
            class_id = class_name2class_id.get(class_name)
            if class_id is None:
                class_id = len(class_names)
                class_names.append(class_name)
                class_name2class_id[class_name] = class_id
                class_counts.append(0)
            class_counts[class_id] += 1

            f = '%s/%s/%s/%s' % (in_root, split, class_name, image_name)
            im = Image.open(f)
            w, h = im.size
            if not shape_filter.accepts(h, w):
                continue

            im = rescale_image(im, flags.size)
            a = np.asarray(im)
            a = a.transpose(2, 0, 1)
            b = a.tobytes()
            out.write(b)

            b = struct.pack('i', class_id)
            out.write(b)

    out_f = '%s/class_names.txt' % flags.out
    with open(out_f, 'w') as out:
        for s in class_names:
            out.write(s + '\n')

    class_counts = np.array(class_counts, np.int32)
    out_f = '%s/class_counts.npy' % flags.out
    class_counts.tofile(out_f)


if __name__ == '__main__':
    main(parse_flags())

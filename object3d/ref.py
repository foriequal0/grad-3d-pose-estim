from os import path
from collections import namedtuple
import h5py

from .opts import opts

labels = ["train", "valid"]


Reference = namedtuple("Reference",
                       ["nsamples", "iters", "batch"])

def load_annots(label: str):
    names_filename = path.join(opts.data_dir, "pascal3d", "annot",
                               "{}_images.txt".format(label))

    images = []
    image_to_indices = {}

    with open(names_filename) as names_file:
        lines = names_file.read().splitlines()
        for index, line in enumerate(lines):
            line = line.rstrip()
            images.append(line)

            indices = image_to_indices.setdefault(line, [])
            indices.append(index)

    annot = h5py.File(path.join(opts.data_dir, "pascal3d", "annot",
                            "{}.h5".format(label)))

    return {
        "dataset": {
            "images": images,
            "part": annot["part"],
            "center": annot["center"],
            "scale": annot["scale"],
        },
        "ref": {
            "n_parts": annot["part"].shape[1],
            "image_to_indices": image_to_indices,
            "nsamples": annot["part"].shape[0],
        },
    }


annots = { label: load_annots(label) for label in labels }

ref_train = Reference(
    nsamples=annots["train"]["nsamples"],
    iters=opts.train_iters,
    batch=opts.train_batch,
)
ref_valid = Reference(
    nsamples=annots["valid"]["nsamples"],
    iters=opts.valid_iters,
    batch=opts.valid_batch,
)
ref_predict = Reference(
    nsamples=annots['valid']["nsamples"],
    iters=annots['valid']["nsamples"],
    batch=1
)

n_parts = annots["train"]["part"].shape[1]
data_dim = (3, opts.input_res, opts.input_res)
label_dim = (n_parts, opts.output_res, opts.output_res)

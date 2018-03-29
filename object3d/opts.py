import argparse

__all__ = ["parser", "opts"]

parser = argparse.ArgumentParser()

_general = parser.add_argument_group("General options")
_general.add_argument("--mode", default="train", choices=["train", "pred", "dummy"])
_general.add_argument("--exp-id", default="default")
_general.add_argument("--data-dir", default="./data")
_general.add_argument("--model-dir", default="./model/{exp-id}")
_general.add_argument("--log-dir", default="./log/{exp-id}")
_general.add_argument("--manual-seed", default=-1, type=int)

_training = parser.add_argument_group("Training options")
_training.add_argument("--optimizer", choices=["tf", "torch"])
_training.add_argument("--n-epochs", default=100, type=int)
_training.add_argument("--train-iters", default=4000, type=int)
_training.add_argument("--train-batch", default=4, type=int)
_training.add_argument("--valid-iters", default=2958, type=int)
_training.add_argument("--valid-batch", default=4, type=int)

_data = parser.add_argument_group("Data Options")
_data.add_argument("--input-res", default=256, type=int)
_data.add_argument("--output-res", default=64, type=int)
_data.add_argument("--scale-factor", default=0.25, type=float)
_data.add_argument("--rot-factor", default=30.0, type=float)

opts, _unknown = parser.parse_known_args()
if "{exp-id}" in opts.model_dir:
    opts.model_dir = opts.model_dir.format(**{"exp-id": opts.exp_id})
if "{exp-id}" in opts.log_dir:
    opts.log_dir = opts.log_dir.format(**{"exp-id": opts.exp_id})
""" ProLIP
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
"""
import os
import yaml

import munch

import torch.nn as nn

from prolip import create_model_and_transforms, get_tokenizer, get_input_dtype
from training.precision import get_autocast
from training.file_utils import pt_load


def load_model(checkpoint_dir, is_dp=False, batch_size=128):
    if os.path.exists(f"{checkpoint_dir}/checkpoints/last.pt"):
        checkpoint = pt_load(f"{checkpoint_dir}/checkpoints/last.pt", map_location="cpu")
    elif os.path.exists(f"{checkpoint_dir}/checkpoints/epoch_32.pt"):
        checkpoint = pt_load(f"{checkpoint_dir}/checkpoints/epoch_32.pt", map_location="cpu")
    elif os.path.exists(f"{checkpoint_dir}/checkpoints/epoch_128.pt"):
        checkpoint = pt_load(f"{checkpoint_dir}/checkpoints/epoch_128.pt", map_location="cpu")
    else:
        fnames = [_fname for _fname in os.listdir(f"{checkpoint_dir}/checkpoints") if _fname.endswith(".pt")]
        # Assume there is only one checkpoint (delete-prev-checkpoint)
        checkpoint = pt_load(f"{checkpoint_dir}/checkpoints/{fnames[0]}", map_location="cpu")
        if len(fnames) > 1:
            print(f"# candidates = {len(fnames)}")
            print(fnames)
            print(f"{fnames[0]} is selected")

    with open(f"{checkpoint_dir}/params.txt") as fin:
        load_str = ""
        for line in fin:
            if "\t" not in line:
                load_str += line
                args = yaml.safe_load(load_str)
                new_args = args.copy()
                for k, v in args.items():
                    if v == 'None':
                        new_args[k] = None
                    elif v == 'False':
                        new_args[k] = False
                    elif v == 'True':
                        new_args[k] = True
        args = new_args
        args = munch.munchify(args)
        args.train_data = None

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device="cuda",
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,
        pretrained_image=args.pretrained_image,
        output_dict=True,
    )
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint["state_dict"].items()})

    args.precision = "amp"
    args.batch_size = batch_size
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    if is_dp:
        model = nn.DataParallel(model)
    model = model.eval()

    tokenizer = get_tokenizer(args.model)

    return {
        "args": args,
        "model": model,
        "tokenizer": tokenizer,
        "autocast": autocast,
        "input_dtype": input_dtype,
        "preprocess_train": preprocess_train,
        "preprocess_val": preprocess_val,
    }

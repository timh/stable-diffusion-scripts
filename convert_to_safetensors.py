#!/usr/bin/env python3

import torch
import safetensors, safetensors.torch
import sys
from pathlib import Path

for arg in sys.argv[1:]:
    path = Path(arg)
    if path.is_dir() and path.suffix != ".safetensors":
        if not Path(path, "model_index.json").exists():
            print(f"skipping {path}, no model index")
            continue
        newpath = path.with_suffix(".safetensors")
        if newpath.exists():
            print(f"skip {newpath}, already exists")
            continue
        newpath.mkdir(exist_ok=True)
        for subdir in path.iterdir():
            newsubdir = Path(newpath, subdir.name)
            if subdir.is_dir():
                newsubdir.mkdir(exist_ok=True, parents=True)
                for subpath in subdir.iterdir():
                    if subpath.suffix == ".bin":
                        newfile = Path(newsubdir, subpath.name).with_suffix(".safetensors")
                        weights = torch.load(subpath)
                        safetensors.torch.save_file(weights, newfile)
                        print(f"wrote {newfile}")
                    else:
                        newfile = Path(newsubdir, subpath.name)
                        with open(newfile, "w") as out:
                            out.write(open(subpath, "r").read())
                        print(f"write {newfile}")
            else:
                with open(newsubdir, "wb") as out:
                    out.write(open(subdir, "rb").read())
                print(f"write {newsubdir}")


    elif path.suffix == ".ckpt":
        newpath = path.with_suffix(".safetensors")
        weights = torch.load(path)
        safetensors.torch.save_file(weights['state_dict'], newpath)
        print(f"wrote {newpath}")

import transformers
import sys
from typing import Dict
from pathlib import Path
import torch

tokenizer: transformers.CLIPTokenizer = None
text_encoder: transformers.CLIPTextModel = None

def get_embedding(input_ids: torch.Tensor) -> torch.Tensor:
    return text_encoder(input_ids)[0]

def get_input_ids(token: str) -> torch.Tensor:
    return tokenizer(token, truncation=True, return_tensors="pt").input_ids

def get_mean(embedding: torch.Tensor) -> torch.Tensor:
    return torch.mean(embedding, 1, keepdim=True)

def print_token_stats(token: str, unk_mean: torch.Tensor):
    input_ids = get_input_ids(token)
    embedding = get_embedding(input_ids)
    print(f"  input_ids = {input_ids}")
    print(f"  embedding = {embedding}")
    mean = get_mean(embedding)
    # print(f"  mean = {mean}")
    dist = torch.dist(mean, unk_mean)
    print(f"  dist from unknown: {dist:.3}")


if __name__ == "__main__":
    model_name = "/home/tim/models/f222v"
    args = []
    for arg in sys.argv[1:]:
        if Path(arg).exists() and Path(arg).is_dir():
            model_name = arg
            continue
        args.append(arg)

    tokenizer = transformers.CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = transformers.CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")

    unk_token = "<|endoftext|>"
    # unk_token = "<unk>"
    unk_input_ids = get_input_ids(unk_token)
    unk_embedding = get_embedding(unk_input_ids)
    unk_mean = get_mean(unk_embedding)
    print(f"unk_token {unk_token}:")
    print(f"  input_ids {unk_input_ids}")
    # print(f"  embedding {unk_embedding}")
    print()

    special_tokens = list(tokenizer.all_special_tokens)
    for token in special_tokens:
        print(f"special {token}:")
        print_token_stats(token, unk_mean)
    print()

    for arg in args:
        print(f"{arg}:")
        print_token_stats(arg, unk_mean)

    for i in range(len(args)):
        one = args[i]
        one_mean = get_mean(get_embedding(get_input_ids(one)))
        # one_input_ids = get_input_ids(one)
        # one_embedding = get_embedding(one_input_ids)
        # one_mean = get_mean(one_embedding)
        for j in range(i + 1, len(args)):
            two = args[j]
            two_mean = get_mean(get_embedding(get_input_ids(two)))
            dist = torch.dist(one_mean, two_mean)
            print(f"{one} -> {two}: {dist}")

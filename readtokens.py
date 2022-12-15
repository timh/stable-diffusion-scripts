import transformers
import sys
from pathlib import Path
from torch import Tensor

if __name__ == "__main__":
    model_name = "/home/tim/models/f222v"
    args = []
    for arg in sys.argv[1:]:
        if Path(arg).exists() and Path(arg).is_dir():
            model_name = arg
            continue
        args.append(arg)

    tokenizer = transformers.CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

    specials = set(tokenizer.all_special_ids)
    print(f"specials = {specials}")


    def filter_tensor(t: Tensor):
        return [v for v in t if t not in specials]

    for arg in args:
        text_inputs = tokenizer(
            arg,
            # padding="max_length",
            # max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # input_ids = list(map(filter_tensor, text_inputs.input_ids[0]))
        input_ids = text_inputs.input_ids
        filtered = [val for val in input_ids[0] if int(val) not in specials]
        print(f"{arg} = {filtered}")
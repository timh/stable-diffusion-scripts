#!/usr/bin/env python3

import sys, os, os.path
import re
from collections import namedtuple, defaultdict

Picture = namedtuple("Picture", "model_name,prompt,sampler,seed,filename")

def generate_css():
    contents_css_generated = ""
    for prompt in all_prompts:
        for model_name in all_model_names:
            for sampler in all_samplers:
                grid_col = gridcol_for(model_name, prompt, sampler)
                css_cls = f"col_{grid_col}"
                contents_css_generated += f".{css_cls} {{ display: block; grid-column: {grid_col}; }}\n"
                contents_css_generated += f"#checkbox_{css_cls}:checked ~ .{css_cls} {{ visibility: hidden; }}\n"

    for seed in all_seeds:
        grid_row = seed_to_gridrow[seed]
        css_cls = f"row_{grid_row}"
        contents_css_generated += f".{css_cls} {{ display: block; grid-row: {grid_row}; }}\n"
        contents_css_generated += f"#checkbox_{css_cls}:checked ~ .{css_cls} {{ visibility: hidden; }}\n"

    def print_all_objs(prefix, all_objs):
        res = ""
        for idx, obj in enumerate(all_objs):
            css_cls = f"{prefix}_{idx}"
            checkbox_id = f"checkbox_{css_cls}"
            res += f"#{checkbox_id}:checked ~ .{css_cls} {{ visibility: hidden; }}\n"
        return res
    contents_css_generated += print_all_objs("prompt", all_prompts)
    contents_css_generated += print_all_objs("model", all_model_names)
    contents_css_generated += print_all_objs("sampler", all_samplers)

    return contents_css_generated

def find_pngs():
    #alex34_06500-photo of alexhin person, full body, pencil sketch-k_heun_50
    re_dirname = re.compile(r"(.+[\d_]+)-(.+)-([\w_]+_\d+)")
    re_png_invokeai = re.compile(r"(\d+).(\d+)\.png")

    skip_substrs = sys.argv[1:]

    all_pics = list()
    for dirname in os.listdir("."):
        if not os.path.isdir(dirname):
            continue

        match = re_dirname.match(dirname)
        if not match:
            continue

        if any([skip in dirname for skip in skip_substrs]):
            continue

        model_name = match.group(1)
        prompt = match.group(2)
        sampler = match.group(3)
        
        #prompt_safe = prompt.replace(" ", "_").replace(",", "-").replace("+","P")
        print(f"model_name {model_name}, prompt '{prompt}', sampler {sampler}", file=sys.stderr)

        for short_filename in os.listdir(dirname):
            filename = f"{dirname}/{short_filename}"
            if os.path.isdir(filename):
                continue
            if not filename.endswith("png"):
                continue

            match = re_png_invokeai.match(short_filename)
            if not match:
                continue
            seed = int(match.group(2))

            all_pics.append(Picture(model_name, prompt, sampler, seed, filename))
    
    return all_pics

if __name__ == "__main__":
    all_pics = find_pngs()
    columnid_to_modelname = dict()
    columnid_to_prompt = dict()
    columnid_to_sampler = dict()

    all_seeds = sorted(list(set([int(pic.seed) for pic in all_pics])))
    all_model_names = sorted(list(set([pic.model_name for pic in all_pics])))
    all_prompts = sorted(list(set([pic.prompt for pic in all_pics])))
    all_samplers = sorted(list(set([pic.sampler for pic in all_pics])))
    seed_to_gridrow = {seed: idx + 8 for idx, seed in enumerate(all_seeds)}
    max_pics = len(all_prompts) * len(all_model_names) * len(all_samplers)

    idx_for_model_name = {model_name: idx for idx, model_name in enumerate(all_model_names)}
    idx_for_prompt = {prompt: idx for idx, prompt in enumerate(all_prompts)}
    idx_for_sampler = {sampler: idx for idx, sampler in enumerate(all_samplers)}

    def gridcol_for(model_name, prompt, sampler):
        res = idx_for_prompt[prompt] * len(all_model_names) * len(all_samplers)
        res += idx_for_model_name[model_name] * len(all_samplers)
        res += idx_for_sampler[sampler]
        return res + 2

    contents_css_static = open(os.path.dirname(__file__) + "/imagegrid.css").read(10 * 1024)
    contents_css_generated = generate_css()
    print(f"""
    <head>
    <style type="text/css">
    {contents_css_static}
    {contents_css_generated}
    </style>
    </head>""")

    print("<body>")
    print("<div class=\"grid_container\">")

    def print_all_objs(prefix, all_objs, grid_row):
        for idx, obj in enumerate(all_objs):
            css_cls = f"{prefix}_{idx}"
            checkbox_id = f"checkbox_{css_cls}"
            style = f"grid-row: {grid_row}"
            print(f"""<label for="{checkbox_id}" style="{style}">{obj}</label>""")
            print(f"""<input type="checkbox" id="{checkbox_id}" style="{style}"/>""")
    print_all_objs("prompt", all_prompts, 1)
    print_all_objs("model", all_model_names, 2)
    print_all_objs("sampler", all_samplers, 3)

    for seed in all_seeds:
        grid_row = seed_to_gridrow[seed]
        css_cls = f"row_{grid_row}"
        checkbox_id = f"checkbox_{css_cls}"
        print(f"""
    <label for="{checkbox_id}" style="grid-row: {grid_row}; grid-column: 1">{seed}</label>
    <input type="checkbox" id="{checkbox_id}" style="grid-row: {grid_row}; grid-column: 1"/>""")

    for prompt_idx, prompt in enumerate(all_prompts):
        width = len(all_model_names) * len(all_samplers)
        prompt_start = prompt_idx * width + 2
        prompt_end = prompt_start + width
        prompt_css_cls = f"prompt_{prompt_idx}"
        print(f"""<span class="header-prompt {prompt_css_cls}" style="grid-column-start: {prompt_start}; grid-column-end: {prompt_end}">{prompt}</span>""")

        for model_idx, model_name in enumerate(all_model_names):
            model_start = prompt_start + model_idx * len(all_samplers)
            model_end = model_start + len(all_samplers)
            model_css_cls = f"model_{model_idx}"
            print(f"""<span class="header-model-name {prompt_css_cls} {model_css_cls}" style="grid-column-start: {model_start}; grid-column-end: {model_end}">{model_name}</span>""")

            for sampler_idx, sampler in enumerate(all_samplers):
                sampler_col = model_start + sampler_idx
                sampler_css_cls = f"sampler_{sampler_idx}"
                print(f"""<span class="header-sampler {prompt_css_cls} {model_css_cls} {sampler_css_cls}" style="grid-column: {sampler_col}">{sampler}</span>""")

                css_cls = f"col_{sampler_col}"
                checkbox_id = f"checkbox_{css_cls}"
                print(f"""<input type="checkbox" id="{checkbox_id}" style="grid-column: {sampler_col}" class="header-checkbox {prompt_css_cls} {model_css_cls} {sampler_css_cls}"/>""")

    for idx, pic in enumerate(all_pics):
        row = seed_to_gridrow[pic.seed]
        grid_col = gridcol_for(pic.model_name, pic.prompt, pic.sampler)
        grid_row = seed_to_gridrow[pic.seed]
        css_cls = f"col_{grid_col} row_{grid_row}"

        css_cls += f" prompt_{idx_for_prompt[pic.prompt]}"
        css_cls += f" model_{idx_for_model_name[pic.model_name]}"
        css_cls += f" sampler_{idx_for_sampler[pic.sampler]}"

        print(f"""
    <span class="tooltip {css_cls}"">
    <span class="tooltiptext">
        <ul>
            <li>model_name {pic.model_name}</li>
            <li>prompt {pic.prompt}</li>
            <li>sampler {pic.sampler}</li>
            <li>seed {pic.seed}</li>
        </ul>
        <img src="{pic.filename}" />
    </span>
    <img src="{pic.filename}" />
    </span>""")

    print("</div>")
    print("</body>")

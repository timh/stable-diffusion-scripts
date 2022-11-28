#!/usr/bin/env python3

import sys, os, os.path
import re
from collections import namedtuple, defaultdict

class Picture:
    model_str: str
    model_name: str
    model_seed: str
    model_steps: int
    prompt: str
    sampler: str
    seed: int
    filename: str

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
    
def generate_css():
    contents_css_generated = ""
    for prompt in all_prompts:
        for model_str in all_model_strs:
            for sampler in all_samplers:
                grid_col = gridcol_for(model_str, prompt, sampler)
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
    contents_css_generated += print_all_objs("model", all_model_strs)
    contents_css_generated += print_all_objs("sampler", all_samplers)

    return contents_css_generated

def find_pngs():
    #alex34_06500-photo of alexhin person, full body, pencil sketch-k_heun_50
    re_dirname = re.compile(r"(.+[\d_]+)--(.+)--([\w\+\d_,]+)")
    re_dirname_old = re.compile(r"(.+[\d_]+)-(.+)-([\w\+_]+_\d+( [c\d]+)?)")
    re_png_invokeai = re.compile(r"(\d+).(\d+)\.png")

    re_sampler_cfg = re.compile(r"([\w\+_]+)_(\d+),c(\d+)")

    #alexhin20_f222_5e7_r7_05500
    re_model_str = re.compile(r"([\w\d\._-]+)_r(\d+)_(\d+)")

    all_pics = list()
    for dirname in os.listdir("."):
        if not os.path.isdir(dirname):
            continue

        match = re_dirname.match(dirname)
        if not match:
            match = re_dirname_old.match(dirname)
            if not match:
                continue
    
        if len(sys.argv) > 1:
            all_include_strs = [arg[1:] for arg in sys.argv[1:] if arg[0] == '+']
            all_exclude_strs = [arg[1:] for arg in sys.argv[1:] if arg[0] == '-']

            if len(all_include_strs) > 0:
                include_any_true = any([arg in dirname for arg in all_include_strs])
            else:
                include_any_true = True
            exclude_any_true = any([arg in dirname for arg in all_exclude_strs])

            if not include_any_true or exclude_any_true:
                continue

        model_str = match.group(1)
        prompt = match.group(2)
        sampler = match.group(3)
        match_sampler = re_sampler_cfg.match(sampler)
        if match_sampler:
            sname, ssteps, cfg = match_sampler.group(1), int(match_sampler.group(2)), int(match_sampler.group(3))
            sampler = f"{sname} {ssteps:02}, c{cfg:02}"
        
        #prompt_safe = prompt.replace(" ", "_").replace(",", "-").replace("+","P")

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

            # normalize 'alexhin person' and 'alexhin woman' to 'alexhin'
            prompt = prompt.replace("alexhin person", "alexhin")
            prompt = prompt.replace("alexhin woman", "alexhin")

            match = re_model_str.match(model_str)
            if match:
                model_name = match.group(1)
                model_seed = int(match.group(2))
                model_steps = int(match.group(3))
                model_str_pretty = f"{model_name} seed {model_seed} steps {model_steps:04}"
            else:
                model_name = model_str
                model_seed = None
                model_steps = None
                model_str_pretty = model_str

            print(f"model_str {model_str} (name {model_name}, seed {model_seed}, steps {model_steps}), prompt '{prompt}', sampler {sampler}", file=sys.stderr)
            all_pics.append(Picture(model_str=model_str, model_str_pretty=model_str_pretty, model_name=model_name, model_seed=model_seed, model_steps=model_steps, prompt=prompt, sampler=sampler, seed=seed, filename=filename))
    
    return all_pics

if __name__ == "__main__":
    all_pics = find_pngs()
    columnid_to_modelname = dict()
    columnid_to_prompt = dict()
    columnid_to_sampler = dict()

    all_seeds = sorted(list(set([int(pic.seed) for pic in all_pics])))
    all_model_strs = sorted(list(set([pic.model_str for pic in all_pics])))
    all_model_strs_pretty = sorted(list(set([pic.model_str_pretty for pic in all_pics])))
    all_prompts = sorted(list(set([pic.prompt for pic in all_pics])))
    all_samplers = sorted(list(set([pic.sampler for pic in all_pics])))
    seed_to_gridrow = {seed: idx + 8 for idx, seed in enumerate(all_seeds)}
    max_pics = len(all_prompts) * len(all_model_strs) * len(all_samplers)

    idx_for_model_str = {model_str: idx for idx, model_str in enumerate(all_model_strs)}
    idx_for_prompt = {prompt: idx for idx, prompt in enumerate(all_prompts)}
    idx_for_sampler = {sampler: idx for idx, sampler in enumerate(all_samplers)}

    def gridcol_for(model_str, prompt, sampler):
        res = idx_for_prompt[prompt] * len(all_model_strs) * len(all_samplers)
        res += idx_for_model_str[model_str] * len(all_samplers)
        res += idx_for_sampler[sampler]
        return res + 2

    contents_css_static = open(os.path.dirname(__file__) + "/imagegrid.css").read()
    contents_js_static = open(os.path.dirname(__file__) + "/imagegrid.js").read()
    contents_css_generated = generate_css()
    print(f"""
    <head>
    <script type="text/javascript">
    {contents_js_static}
    </script>
    <style type="text/css">
    {contents_css_static}
    {contents_css_generated}
    </style>
    </head>""")

    print("<body>")
    print("<div class=\"grid_container\">")

    def print_all_objs(prefix, all_objs, grid_row):
        style = f"grid-row: {grid_row}; grid-column-start: 1; grid-column-end: {max_pics + 2}"
        inside_span = f"""<span style="{style}">"""
        outside_span = ""
        for idx, obj in enumerate(all_objs):
            css_cls = f"{prefix}_{idx}"
            checkbox_id = f"checkbox_{css_cls}"
            inside_span += f"""  <label for="{checkbox_id}" style="margin-right: 20px">{obj}</label>"""
            outside_span += f"""  <input type="checkbox" id="{checkbox_id}" style="{style}"/>"""
        inside_span += "</span>"
        print(inside_span)
        print(outside_span)

    print_all_objs("prompt", all_prompts, 1)
    print_all_objs("model", all_model_strs_pretty, 2)
    print_all_objs("sampler", all_samplers, 3)

    for seed in all_seeds:
        grid_row = seed_to_gridrow[seed]
        css_cls = f"row_{grid_row}"
        checkbox_id = f"checkbox_{css_cls}"
        print(f"""
    <label for="{checkbox_id}" style="grid-row: {grid_row}; grid-column: 1">{seed}</label>
    <input type="checkbox" id="{checkbox_id}" style="grid-row: {grid_row}; grid-column: 1"/>""")

    for prompt_idx, prompt in enumerate(all_prompts):
        width = len(all_model_strs) * len(all_samplers)
        prompt_start = prompt_idx * width + 2
        prompt_end = prompt_start + width
        prompt_css_cls = f"prompt_{prompt_idx}"
        print(f"""<span class="header-prompt {prompt_css_cls}" style="grid-column-start: {prompt_start}; grid-column-end: {prompt_end}">{prompt}</span>""")

        for model_idx, model_str in enumerate(all_model_strs):
            model_start = prompt_start + model_idx * len(all_samplers)
            model_end = model_start + len(all_samplers)
            model_css_cls = f"model_{model_idx}"
            model_str_pretty = all_model_strs_pretty[model_idx]
            print(f"""<span class="header-model-name {prompt_css_cls} {model_css_cls}" style="grid-column-start: {model_start}; grid-column-end: {model_end}">{model_str_pretty}</span>""")

            for sampler_idx, sampler in enumerate(all_samplers):
                sampler_col = model_start + sampler_idx
                sampler_css_cls = f"sampler_{sampler_idx}"
                print(f"""<span class="header-sampler {prompt_css_cls} {model_css_cls} {sampler_css_cls}" style="grid-column: {sampler_col}">{sampler}</span>""")

                css_cls = f"col_{sampler_col}"
                checkbox_id = f"checkbox_{css_cls}"
                print(f"""<input type="checkbox" id="{checkbox_id}" style="grid-column: {sampler_col}" class="header-checkbox {prompt_css_cls} {model_css_cls} {sampler_css_cls}"/>""")

    for idx, pic in enumerate(all_pics):
        row = seed_to_gridrow[pic.seed]
        grid_col = gridcol_for(pic.model_str, pic.prompt, pic.sampler)
        grid_row = seed_to_gridrow[pic.seed]
        css_cls = f"col_{grid_col} row_{grid_row}"

        css_cls += f" prompt_{idx_for_prompt[pic.prompt]}"
        css_cls += f" model_{idx_for_model_str[pic.model_str]}"
        css_cls += f" sampler_{idx_for_sampler[pic.sampler]}"

        check_id = f"check_{idx}"
        args = f"'{pic.prompt}', '{pic.model_str}', '{pic.sampler}', '{check_id}'"

        print(f"""
    <span class="tooltip {css_cls}"">
    <span class="tooltiptext">
        <ul>
            <li>model_str {pic.model_str}</li>
            <li>prompt {pic.prompt}</li>
            <li>sampler {pic.sampler}</li>
            <li>seed {pic.seed}</li>
        </ul>
        <img src="{pic.filename}" />
    </span>
    <span id="{check_id}" class="img_unselected">check</span>
    <img src="{pic.filename}" onClick="mark({args})" />
    </span>""")

    print("</div>")
    print("<div id=\"results\"></div>")
    print("</body>")

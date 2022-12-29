
from pathlib import Path
from typing import List
import sys
import math

print("<style>")
sourcedir = Path(__file__).parent
print(open(sourcedir.joinpath("simplegrid.css"), "r").read())
print("</style>")

def get_image_paths(dir: Path) -> List[Path]:
    res: List[Path] = list()
    for path in dir.iterdir():
        if path.is_dir():
            child_res = get_image_paths(path)
            res.extend(child_res)
            continue

        if path.suffix in [".jpg", "jpeg", ".png"]:
            res.append(path)
    return res

# figure out args.
input_dirs: List[Path]
if len(sys.argv) > 1:
    input_dirs = [Path(arg) for arg in sys.argv[1:]]
else:
    input_dirs = [Path(".")]

# load 'em
paths: List[Path] = []
for input in input_dirs:
    paths.extend(get_image_paths(input))

num_paths = len(paths)
side = math.sqrt(num_paths)

#width = int(side * 16 / 9)
width = 12

print("""<div id="images">""")
for idx, path in enumerate(paths):
    column = idx % width + 1
    row = int(idx / width) + 1
    print(f"""
<span class="image" style="grid-row: {row}; grid-column: {column}">
  <img src="{path}" class="thumbnail" />
  <img src="{path}" class="fullsize" />
</span>
    """)
print("""</div>""")
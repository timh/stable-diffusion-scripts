import sys, os, re

if len(sys.argv) <= 1:
    raise Exception("need prefix")
prefix = sys.argv[1]

re_filename = re.compile("(DSC|_DSC|IMG|[\dA-F]).+(jpg|jpeg)")
index = 1

for filename in os.listdir("."):
    if not re_filename.match(filename):
        print(f"skip {filename}")
        continue
    new_filename = ""
    while new_filename == "" or os.path.exists(new_filename):
        new_filename = f"{prefix} ({index}).jpg"
        index = index + 1
    os.rename(filename, new_filename)
    print(f"{filename} => {new_filename}")



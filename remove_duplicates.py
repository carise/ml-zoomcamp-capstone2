"""
This is a quick and dirty script to remove duplicate images.
It is not perfect and may or may not work without some tweaking.

This project is using python 3.11, and this script uses tkinter,
which isn't automatically compiled with python 3.11.
"""
import hashlib
import os
import psutil
import tkinter as tk

from collections import defaultdict
from PIL import Image, ImageTk
from pathlib import Path

seen_files = defaultdict(list)

print("Finding duplicates in data/...")
for dirpath, dirnames, filenames in os.walk('data'):
    for f in filenames:
        filepath = os.path.join(dirpath, f)
        with open(filepath, 'rb') as fp:
            file_digest = hashlib.file_digest(fp, 'md5').hexdigest()
            seen_files[file_digest].append(filepath)


dupes = [files for _, files in seen_files.items() if len(files) > 1]
print(f"There are {len(dupes)} groups of duplicates")

# Present the files in the duplicate group and allow user to
# select which one to keep. User can enter "-1" if the images
# should all be moved (e.g. they don't match any of the categories).
# Duplicate images in a group in the same directory will automatically
# get moved, except for the first image in the group.
for dupe_group in dupes:
    i = 0

    group_path = None
    has_same_path = True
    for f in dupe_group:
        if group_path is None:
            group_path = os.path.dirname(f)
        else:
            has_same_path = has_same_path and (group_path == os.path.dirname(f))
        print(f"[{i}] {f}")
        i += 1

    if has_same_path:
        idx = 0
    else:
        root = tk.Tk()
        img = ImageTk.PhotoImage(Image.open(dupe_group[0]))
        label = tk.Label(root, image=img).pack()
        root.after(3000, lambda: root.destroy())
        root.mainloop()

        idx = int(input("Keep which? "))

    i = 0
    for f in dupe_group:
        i += 1
        if i-1 == idx and idx > -1:
            continue
        # preserve the path under data/duplicates in case it gets messed up
        new_dirname = os.path.join('data/duplicates', os.path.dirname(f))
        Path(new_dirname).mkdir(parents=True, exist_ok=True)
        new_loc = os.path.join(new_dirname, os.path.basename(f))
        print(f"Move {f} to {new_loc}")
        os.rename(f, new_loc)
    print('\n')


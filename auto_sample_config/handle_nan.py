import os
import shutil

root_path = r'Z:\dufeilong\datasets\new'
folders = os.listdir(root_path)

for folder in folders:
    if len(folder.split(r'.'))>1:
        continue
    full_path = os.path.join(root_path, folder)
    items = os.listdir(full_path)
    if len(items) < 34:
        shutil.rmtree(full_path)
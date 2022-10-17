import os

from tqdm import tqdm

path = r"DeepFish/VOC2007/JPEGImages"

# num = 1

for file in tqdm(os.listdir(path)):
    name = os.path.splitext(file)
    # 如果后缀是.dat
    if name[1] == ".png":
        # 重新组合文件名和后缀名
        newname = name[0] + ".jpg"
        old_name = os.path.join(path, file)
        new_name = os.path.join(path, newname)
        os.rename(old_name, new_name)
    # old_name = os.path.join(path, file)
    # new_name = os.path.join(path)

    # num = num + 1
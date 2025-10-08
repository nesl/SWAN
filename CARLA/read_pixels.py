from PIL import Image
import numpy as np

data = np.array(Image.open('/home/jason/Desktop/Dyn_MMOT/CARLA/data/00030_sem.png')) # H, W, 3

id_set = {}
h, w, c = data.shape
for i in range(h):
    for j in range(w):
        if data[i][j][0] == 12:
            str_id = str(data[i][j][1]) + str(data[i][j][2])
            if str_id not in id_set:
                id_set[str_id] = 0
            else:
                id_set[str_id] += 1
print(id_set)

with open('dump.txt', 'r') as handle:
    data = handle.readlines()

for line in data:
    id = int(line.strip())
    G = str((id & 0x00ff) >> 0)
    B = str((id & 0xff00) >> 8)
    if (G + B) in id_set:
        print('G', G)
        print('B', B)
        print("FOUND", id)

import pdb; pdb.set_trace()

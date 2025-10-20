import cv2
from constants import const
import json

output_video = 'rendered.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4
fps = 20  # frames per second
video = cv2.VideoWriter(output_video, fourcc, fps, (const.IMAGE_WIDTH, const.IMAGE_HEIGHT))
for idx in range(10782, 20206):
    try:
        str_idx = str(idx).zfill(8)
        img = cv2.imread(f'data/{str_idx}_rgb.png')
        with open(f'outputs/{str_idx}_filtered_gt.json', 'r') as handle:
            entries = json.load(handle)
        for entry in entries:
            x, y, w, h = entry['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            pt1, pt2 = (x, y), (x+w, y)
            # Top edge
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            # Left Edge
            pt1, pt2 = (x, y), (x, y+h)
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            # Right Edge
            pt1, pt2 = (x+w, y), (x+w, y+h)
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            # Bottom Edge
            pt1, pt2 = (x, y+h), (x+w, y+h)
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        video.write(img)
    except Exception as e:
        print("Corresponding files not found for idx", idx)
        print(e)
video.release()
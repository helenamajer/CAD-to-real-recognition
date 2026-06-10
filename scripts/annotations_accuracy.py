import cv2
import numpy as np
from pathlib import Path

# grab one image and its label from the training set
img_path   = "/Users/helenamajer/Git-Repos/Instance_Recognition_App/Instance-Recognition/data/yolo_dataset/images/val/Lamp_Bracket_53739782_0310.png"
label_path = "/Users/helenamajer/Git-Repos/Instance_Recognition_App/Instance-Recognition/data/yolo_dataset/labels/val/Lamp_Bracket_53739782_0310.txt"

img   = cv2.imread(str(img_path))
h, w  = img.shape[:2]
line  = open(label_path).readline().strip().split()
coords = [float(x) for x in line[1:]]

# convert normalised coords back to pixels
points = [(int(coords[i] * w), int(coords[i+1] * h)) 
          for i in range(0, len(coords), 2)]

# draw polygon on image
pts = np.array(points, np.int32).reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, (0, 255, 0), 2)

cv2.imwrite("test_photos/annotation_check.png", img)
print("Saved to test_photos/annotation_check.png")
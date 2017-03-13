import cv2
cv2.__version__
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

## Template matching
# http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html
paths = "/home/animesh/Documents/Kaggle/Fisheries/train/ALB"
file2 = os.path.join(paths, "img_07917.jpg")
img_ppl = cv2.imread(file2, 0)
print(img_ppl.shape, img_ppl.size, img_ppl.dtype)

template = img_ppl[170:330, 550:900]
cv2.imshow('image', template)
cv2.waitKey(0)


test_img = os.path.join(paths, "img_00029.jpg") # img_00176,img_02758, img_01512
img = cv2.imread(test_img, 0)
img2 = img
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED']

for meth in methods:
    img = img2
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')  # ,aspect='auto'
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()


## Test csv markers file

test_img = os.path.join(paths, "img_00012.jpg")
img_ppl = cv2.imread(test_img, 0)

template = img_ppl[470:622, 494:700]
cv2.imshow('image', template)
cv2.waitKey(0)

template = img_ppl[494:700, 470:622]
cv2.imshow('image', template)
cv2.waitKey(0)

#Fish location [y-20:y+20, x-20:x+20]

import json
import csv
os.chdir("/home/animesh/Documents/Kaggle/Fisheries/labels")

with open('alb_labels.json', 'r') as f:
    data = json.load(f)

print(data['filename'])

data = []
with open('alb_labels.json') as f:
    for line in f:
        data.append(json.loads(line))

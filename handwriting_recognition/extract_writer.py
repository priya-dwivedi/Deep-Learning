import os

file_loc = 'iam_dataset/forms_for_parsing.txt'


d = {}
with open('iam_dataset/forms_for_parsing.txt') as f:
    for line in f:
        key = line.split(' ')[0]
        writer = line.split(' ')[1]
        d[key] = writer


print(d)
print(len(d))

#make a counter to count num of writers

select_writer_list = []

from collections import Counter
num_writers = Counter(d.values())
for k,v in num_writers.most_common():
    if v >4:
        select_writer_list.append(k)

print(select_writer_list)
print(len(select_writer_list)) #301 writers have more than 1 form. Lets just work of them

#Select 50 most common writers for this analysis
select_50_writer = []
from collections import Counter
num_writers = Counter(d.values())
for k,v in num_writers.most_common(50):
    select_50_writer.append(k)

print(select_50_writer)
print(len(select_50_writer))

## Now I need list of forms related to these writers
select_forms = []
for k,v in d.items():
    if v in select_50_writer:
        select_forms.append(k)

print(select_forms)  #Only has forms associated with top 50 writers
print(len(select_forms))

##Copy all associated files to a new directory
import os
import glob
import shutil

cwd = os.getcwd()
new_path = os.path.join('iam_dataset','sentences','*.png')

for filename in sorted(glob.glob(new_path)):
    image_name = filename.split('/')[-1]
    file, ext = os.path.splitext(image_name)
    parts = file.split('-')
    form = parts[0]+'-'+parts[1]
    if form in select_forms:
        dst = os.path.join('iam_dataset','data_subset',image_name)
        shutil.copy2(filename,dst)

width_list = []
height_list = []
from PIL import Image
new_path = os.path.join('iam_dataset','data_subset','*.png')
for filename in sorted(glob.glob(new_path)):
    im = Image.open(filename)
    width = im.size[0]
    height = im.size[1]
    width_list.append(width)
    height_list.append(height)

import matplotlib.pyplot as plt
plt.plot(width_list)
plt.ylabel('width')
plt.show()

plt.plot(height_list)
plt.ylabel('height')
plt.show()

print("Avg width: ", sum(width_list)/len(width_list))
print("Avg height: ", sum(height_list)/len(height_list))
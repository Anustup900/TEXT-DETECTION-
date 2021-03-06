

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

import os
from os.path import exists, join, basename, splitext

git_repo_url = 'https://github.com/argman/EAST.git'
project_name = splitext(basename(git_repo_url))[0]
if not exists(project_name):
  # clone and install
  !git clone -q $git_repo_url
  #!cd $project_name && pip install -q -r requirements.txt
  # patch for the Python 3.7
  !sed -i 's/python3-config/python3.7-config/g' EAST/lanms/Makefile
  
import sys
sys.path.append(project_name)
import time
import matplotlib
import matplotlib.pylab as plt
plt.rcParams["axes.grid"] = False

def download_from_google_drive(file_id, file_name):
  # download a file from the Google Drive link
  !rm -f ./cookie
  !curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=$file_id" > /dev/null
  confirm_text = !awk '/download/ {print $NF}' ./cookie
  confirm_text = confirm_text[0]
  !curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=$confirm_text&id=$file_id" -o $file_name
  

pretrained_model = 'east_icdar2015_resnet_v1_50_rbox'
if not exists(pretrained_model):
  # download the pretrained model
  pretrained_model_file_name = 'east_icdar2015_resnet_v1_50_rbox.zip'
  download_from_google_drive('0B3APw5BZJ67ETHNPaU9xUkVoV0U', pretrained_model_file_name)
  !unzip $pretrained_model_file_name

IMAGE_URL = 'https://raw.githubusercontent.com/tugstugi/dl-colab-notebooks/master/resources/billboard.jpg'


image_file_name = basename(IMAGE_URL)
download_dir = '/content/images'
!mkdir -p $download_dir && rm -rf $download_dir/*
!wget -q -P $download_dir $IMAGE_URL
  

plt.imshow(matplotlib.image.imread(join(download_dir, image_file_name)))

!mkdir -p output
!cd $project_name && python eval.py --test_data_path=$download_dir --gpu_list=0 --checkpoint_path=../$pretrained_model/ --output_dir=..

plt.figure(figsize=(20, 26))
plt.imshow(matplotlib.image.imread(image_file_name))

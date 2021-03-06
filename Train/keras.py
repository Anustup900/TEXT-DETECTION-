# Commented out IPython magic to ensure Python compatibility.
try:
  # %tensorflow_version only exists in Colab.
#   %tensorflow_version 2.x
except Exception:
  pass

!pip install keras-ocr

import matplotlib.pyplot as plt
import keras_ocr
import os

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

!mkdir uploads

"""## Upload files (images with text) form your local filesystem"""

# IMPORTANT! If the execution of this cell fails, JUST EXECUTE IT AGAIN!
from google.colab import files

uploaded = files.upload()
uploaded_files = list(uploaded.keys())
for uploaded_file in uploaded_files:
  print(uploaded_file)
  !mv $uploaded_file uploads/$uploaded_file

uploads_dir = "/content/uploads"
custom_images = []

for filename in os.listdir(uploads_dir):
    print(os.path.join(uploads_dir, filename))
    custom_images.append(os.path.join(uploads_dir, filename))

images = [ keras_ocr.tools.read(path) for path in custom_images]

# Commented out IPython magic to ensure Python compatibility.
# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
# %time predictions = pipeline.recognize(images)

fig, axs = plt.subplots(nrows=len(images), figsize=(10, 10))
if(len(custom_images) == 1):
  for image, prediction in zip(images, predictions):
    keras_ocr.tools.drawAnnotations(image=image, predictions=prediction, ax=axs)
else:
  for ax, image, prediction in zip(axs, images, predictions):
    keras_ocr.tools.drawAnnotations(image=image, predictions=prediction, ax=ax)

with open('results.txt', 'a+') as f:
  for idx, prediction in enumerate(predictions):
    if(idx != 0):
      print("\n")
      f.write("\n\n")
    print("Results for the file: " + os.path.basename(custom_images[idx]))
    f.write("Results for the file: " + os.path.basename(custom_images[idx]) + ":\n\n")
    for word, array in prediction:
      if word == "\n":
        print("\n")
        f.write("\n")
      else:
        print(word,  end = ' ')
        f.write(word + " ")

files.download("results.txt")


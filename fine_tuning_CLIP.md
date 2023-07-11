## Fine-tuning CLIP

With the dataset of pairs of images and book titles with predicted categories, we built the pipeline to transform and feed data into a pre-trained CLIP model to fine-tune for the task of relating book cover images and book titles with categories. Since CLIP is a large model and works with visual data, we utilize [PySpark’s TorchDistributor package](https://spark.apache.org/docs/latest//api/python/reference/api/pyspark.ml.torch.distributor.TorchDistributor.html) to create a series of TorchDistriubutor objects and parallelize the fine-tuning (training) process of the CLIP model. It allows us to save computational resources and fine-tune the model on large-scale data.

![test_book](images/test_book.png)

![model_results](images/table_results_718.png)

In the figure above, we use the example book cover and generate a few book title prompts to test for similarity with the book cover. From the table above, we can see that the closest title is estimated to be ”The Power of Nowww” which almost says the same thing as the actual book cover, and the model is able to understand the similarity. In the future, if we add a predicted category to each prompt and fine-tune the model again, we will be able to increase the similarity scores.

The full code is [here](code/finetuning_part.py).

### The code to fine-tune CLIP with book covers data

Import PySpark:
```
%%bash
pip install pyspark &> /dev/null
```

Import required libraries and create spark and sparkcontext objects:
```
from pyspark.sql import SparkSession
from pyspark.sql import Row
import numpy as np
import pandas as pd
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
```
```
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import re
```

Import the training data for the CLIP.
In the future, I plan to use the classification model from part 1 (Naive Bayes) to predict the categories.
The dataset I use here has categories, but I wanted to use the predicted ones and see whether I can improve the fine-tuned CLIP model with that data.
```
datasource = pd.read_csv('/content/drive/MyDrive/train_images/Mind-Body-Spirit/Mind-Body-Spirit.csv', delimiter=',')
```

Import the validation data I use to test the CLIP model and try out some prompts and see how well fine-tuned CLIP will be able to recognize them.
```
datasource_val = pd.read_csv('/content/drive/MyDrive/train_images/Medical/Medical.csv', delimiter=',')
```
```
image	name	author	format	book_depository_stars	price	currency	old_price	isbn	category	img_paths
0	https://d1w7fb2mkkr3kw.cloudfront.net/assets/i...	Milk and Honey	Rupi Kaur	Paperback	4.0	8.18	$	15.0	9781449474256	Mind-Body-Spirit	dataset/Mind-Body-Spirit/0000001.jpg
1	https://d1w7fb2mkkr3kw.cloudfront.net/assets/i...	The Power of Now	Eckhart Tolle	Paperback	4.0	8.20	$	13.9	9780340733509	Mind-Body-Spirit	dataset/Mind-Body-Spirit/0000002.jpg
2	https://d1w7fb2mkkr3kw.cloudfront.net/assets/i...	The Happiness Trap	Russ Harris	Paperback	4.0	8.34	$	13.9	9781845298258	Mind-Body-Spirit	dataset/Mind-Body-Spirit/0000003.jpg
3	https://d1w7fb2mkkr3kw.cloudfront.net/assets/i...	Gifts Of Imperfection, The:	Brene Brown	Paperback	4.0	11.80	$	14.9	9781592858491	Mind-Body-Spirit	dataset/Mind-Body-Spirit/0000004.jpg
4	https://d1w7fb2mkkr3kw.cloudfront.net/assets/i...	Man's Search For Meaning	Viktor E. Frankl	Paperback	4.5	9.66	$	NaN	9781846041242	Mind-Body-Spirit	dataset/Mind-Body-Spirit/0000005.jpg
```

Here, I prepare the training data for fine-tuning CLIP on.
In the future I plan to predict the categories for each book title first
```
data_med = datasource.loc[datasource['category'] == 'Mind-Body-Spirit']

# Since I'm using data from the drive, I want to change the image path for the CLIP to be able to read the image files for training
data_med['img_paths'] = data_med['img_paths'].str.replace('dataset/Mind-Body-Spirit/', '')

df = data_med[['img_paths', 'name', 'author']]

# I want to put all the image paths in a separate list
image_paths = ["/content/drive/MyDrive/train_images/Mind-Body-Spirit/"+i for i in df['img_paths']]

# Image captions are also put in a separate list 
# In the future I will include the predicted categories within each caption 
captions = [i for i in df['name']]

# this is final training dataset that i will be feeding into a training dataloader function
train_data = pd.DataFrame(list(zip(captions, image_paths)),columns=['caption','image_url'])
train_data = train_data.applymap(str)
train_data
```
```
	caption	image_url
0	Milk and Honey	/content/drive/MyDrive/train_images/Mind-Body-...
1	The Power of Now	/content/drive/MyDrive/train_images/Mind-Body-...
2	The Happiness Trap	/content/drive/MyDrive/train_images/Mind-Body-...
3	Gifts Of Imperfection, The:	/content/drive/MyDrive/train_images/Mind-Body-...
4	Man's Search For Meaning	/content/drive/MyDrive/train_images/Mind-Body-...
...	...	...
984	Love, Medicine And Miracles	/content/drive/MyDrive/train_images/Mind-Body-...
985	Inner Child Cards	/content/drive/MyDrive/train_images/Mind-Body-...
986	Warrior Goddess Training	/content/drive/MyDrive/train_images/Mind-Body-...
987	The Salt Path	/content/drive/MyDrive/train_images/Mind-Body-...
988	The Wild Edge Of Sorrow	/content/drive/MyDrive/train_images/Mind-Body-...
```

I perform all the same steps for the validation data. Look full code here for reference.


### Fine-tuning CLIP with Torch distriubutor

Install libraries and tools:
```
!pip install sparktorch
!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git
!pip install transformers
```
```
from pkg_resources import packaging
from sparktorch import serialize_torch_obj, SparkTorch
import torch
import torch.nn as nn
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import Pipeline

import requests
import clip
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

from tqdm.autonotebook import tqdm
```

Define the CUDA device and make sure GPUs are available for fine-tuning.
Also import, the 32-bit pre-trained CLIP model to match the image size of your data.
```
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
```

### Creating a dataloader object from the data

Import Torch distributor:
```
from pyspark.ml.torch.distributor import TorchDistributor
```
The class below is written with help from: https://github.com/openai/CLIP/issues/83.
It accepts training data as input and performs text and image vectorization using internal CLIP functions to transform both caption and image columns in the training data
```
BATCH_SIZE = 32

class image_title_dataset(Dataset):
    def __init__(self, list_image_path,list_txt):

        self.image_path = list_image_path
        self.title  = clip.tokenize(list_txt) 

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx])) # Image from PIL module
        title = self.title[idx]
        return image,title

# i define the function that uses the class to create a training dataloader
def load_features(list_image_path, list_txt):
  dataset = image_title_dataset(list_image_path,list_txt)
  train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE) 
  return train_dataloader
```

Define torch distributor object for the dataloader
```
train_dataloader = TorchDistributor(num_processes=1, 
local_mode=True, 
use_gpu=False).run(load_features,
train_data["image_url"].values, train_data["caption"].values)
```

Function to convert the model to f32:
```
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float()
```

Define parameters for the training:
```
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params from CLIP paper
```


### Training the CLIP model

The train function is also written with the help of: https://github.com/openai/CLIP/issues/83
It takes the images and captions from the training dataloader and creates logits, which are probabilities of similarities.
```
def train_clip(EPOCH, train_dataloader):
  for epoch in range(EPOCH):
    for batch in train_dataloader:
        optimizer.zero_grad()

        images,texts = batch 
      
        images= images.to(device)
        texts = texts.to(device)
      
        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss.backward()
        if device == "cpu":
          optimizer.step()

        else : 
          convert_models_to_fp32(model)
          optimizer.step()
          clip.model.convert_weights(model)
```

Define the torch distributor object for the training step.
This will allow me to run the fine-tuning process in parallel in case I have multiple nodes I can use.
```
TorchDistributor(num_processes=1, 
local_mode=True, 
use_gpu=False).run(train_clip,
11, train_dataloader)
```

Save the model.
I save the model's checkpoint in order to be able to continue fine-tuning process in the future with the updated data (predicted categories included in the caption for example)
```
torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, f"/content/drive/MyDrive/model_test2.pt")
```

Load the model at the checkpoint
```
checkpoint = torch.load("/content/drive/MyDrive/model_test2.pt", map_location=torch.device('cpu')) 

model.load_state_dict(checkpoint['model_state_dict'])
```


### Model testing and evaluation:

### Method 1:

Get an example data from the validation set to check the model:
```
image_path = val_data["image_url"][0]
Image.open(image_path)
```

Here, I test the capability of the fine-tuned model
I input the example image from the validation set as well as a set of captions and I want to see how each caption is similar to the example image.
```
image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(c) for c in testdata.caption]).to(device)
#text_inputs = torch.cat(["Random Book","The Power of Cringe", "Cat Stories"]).to(device)
```

Calculate features:
```
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
```

Pick the top 5 most similar labels for the image
```
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)
```

Print the results.
Since the first title prompt refers to the title of the example image it has the highest accuracy.
```
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{index}: {100 * value.item():.2f}%")
```
```
Top predictions:

8: 97.22%
0: 1.32%
1: 0.70%
7: 0.44%
2: 0.17%
```


### Method 2:

```
Image.open(testdata["image_url"][0])
```
![example_book](images/test_book.png)

Here, I want to give CLIP a set of random prompts as well as one closely related to the example book.
I want to see if CLIP can recognize the closest title to the book cover.
```
image = preprocess(Image.open(val_data["image_url"][0])).unsqueeze(0).to(device)
text = clip.tokenize(['Lord of Rings','The Power of Nowww', 'random book', 'Health Power']).to(device)
#text = clip.tokenize([c for c in testdata.caption]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
```
```
Label probs: [[0.1239  0.86    0.00792 0.00817]]
```

The results show that the second title prompit I gave the model is the closest and others are pretty close to 0, which makes sense. This shows that the model is able to associate the title prompt and give it the best book cover estimate.
In the future, I plan to add a predicted category that can be used with the title prompt and it may increase the accuracy of the fine-tuned CLIP.

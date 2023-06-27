#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os, glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input,LSTM
from tensorflow.keras.applications import ResNet50
import tensorflow as tf 
from tensorflow.keras.backend import is_keras_tensor
from tensorflow.python.keras.backend import is_keras_tensor as is_keras_tensor_tf
from keras_resnet.models import ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications. resnet50 import preprocess_input
from sklearn.metrics import classification_report


# In[4]:


file_path = 'C:\\Users\\DELL\\Desktop\\train\\'


# In[5]:


import os


# In[6]:


os.getcwd()


# In[7]:


name_class =  os.listdir(file_path)
name_class


# In[8]:


filepaths = list (glob.glob(file_path+'/**/*.*'))


# In[9]:


filepaths


# In[10]:


labels = list (map(lambda x: os. path.split(os.path.split(x) [0])[1], filepaths))
labels


# In[11]:


filepath = pd.Series (filepaths, name='Filepath').astype (str)
labels = pd.Series(labels, name='Label')
data= pd.concat([filepath, labels], axis=1)
data = data.sample (frac=1).reset_index (drop=True)
data.head (5)


# In[12]:


counts=data.Label.value_counts()
sns.barplot(x=counts.index,y=counts)
plt.xlabel('Type')
plt.xticks(rotation=90);


# In[13]:


train,test=train_test_split(data,test_size=0.25,random_state=42)


# In[14]:


fig, axes=plt.subplots(nrows=5, ncols=3, figsize=(10,8), subplot_kw={'xticks':[],'yticks':[]})
for i,ax in enumerate(axes.flat):
   ax.imshow(plt.imread(data.Filepath[i]))
   ax.set_title(data.Label[i])
plt.tight_layout()
plt.show()


# In[15]:


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# In[16]:


train_gen = train_datagen.flow_from_dataframe(
   dataframe=train,
   x_col='Filepath',
   y_col='Label',
   target_size=(100,100),
   class_mode='categorical',
   batch_size=32,
   shuffle=True,
   seed=42
)
valid_gen = train_datagen.flow_from_dataframe(
  dataframe=test,
  x_col='Filepath',
  y_col='Label',
  target_size=(100,100),
  class_mode='categorical',
  batch_size=32,
  shuffle=False,
  seed=42
)
test_gen = test_datagen.flow_from_dataframe(
   dataframe=test,
   x_col='Filepath',
   y_col='Label',
   target_size=(100,100),
   class_mode='categorical',
   batch_size=32,
   shuffle=True,
   
)


# In[17]:


pretrained_model=tf.keras.applications.ResNet50(input_shape=(100,100, 3),
  include_top=False,
  weights='imagenet',
  pooling='avg'
) 
pretrained_model.trainable=False


# In[18]:


inputs = pretrained_model.input

x = Dense(128, activation='relu')(pretrained_model.output)
x = Dense(128, activation='relu')(x)

outputs= Dense(2, activation='softmax')(x)
model=Model(inputs=inputs, outputs=outputs)


# In[19]:


model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)


# In[20]:


my_callbacks=[EarlyStopping(monitor='val_accuracy',
min_delta=0,
patience=2,
mode='auto')]


# In[21]:


history=model.fit(
train_gen,
validation_data=valid_gen,
epochs=100
)


# In[22]:


model.save('model_resnet50.h5')


# In[23]:


pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
plt.title('Accuracy')
plt.show()


# In[24]:


pd.DataFrame(history.history)[['loss','val_loss']].plot()
plt.title("Loss")
plt.show()


# In[25]:


results=model.evaluate(test_gen,verbose=0) 
print("     Test Loss:{:.5f}".format(results[0]))
print("Test Accuracy:{:2f}%".format(results[1]*100))


# In[26]:


pred = model.predict(test_gen)
pred = np.argmax(pred,axis=1)
labels=(train_gen.class_indices)
labels=dict((v,k) for k,v in labels.items())
pred=[labels[k] for k in pred]


# In[27]:


y_test=list(test.Label) 
print(classification_report(y_test, pred))


# In[28]:


fig, axes=plt.subplots(nrows=4, ncols=2, figsize=(12, 8),
                       subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat): 
    ax.imshow(plt.imread(test.Filepath.iloc[i]))
    ax.set_title(f"True: {test. Label.iloc[i]}\nPredicted: {pred[i]}")

plt.tight_layout()
plt.show()


# In[29]:


import cv2
print(cv2.__version__)


# In[30]:


from tensorflow.keras.models import load_model 
loaded_model_imageNet=load_model("model_resnet50.h5")
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# In[31]:


import matplotlib.pyplot as plt


# In[40]:


import cv2
import numpy as np
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input

#ing-image, load ingling exing.ing to array(img)
img_path ='12.jpg'
img = cv2.imread('C:\\Users\\DELL\\Desktop\\train\\malignant\\23.jpg')
img= cv2.resize(img,(100, 100))

x=np.expand_dims (img, axis=0) 
x= preprocess_input(x)
result =loaded_model_imageNet.predict(x)
print((result*100).astype('int'))
plt.imshow(img)


# In[41]:


p=list((result*100).astype('int'))
pp=list(p[0]) 
print(pp)


# In[42]:


print("Largest element is:", max(pp))


# In[43]:


index=pp.index(max(pp))


# In[44]:


name_class=['benign','malignant']


# In[45]:


name_class[index]


# In[46]:


plt.title(name_class[index])
plt.imshow(img)


# In[ ]:





# In[ ]:





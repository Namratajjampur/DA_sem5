#!/usr/bin/env python
# coding: utf-8

# In[165]:


#Assignment5
#Namrata R PES1201700921
#C Diya PES201700246
#Chiranth J PES1201701438

#modules to be imported
#! pip install librosa
from keras import layers, models, optimizers 
import pandas as pd 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
from matplotlib.pyplot import imread
from keras.utils import to_categorical, np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import librosa
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import librosa.display


# In[114]:


path1='styles.csv'
audio_path = r'C:\Users\diyas\Downloads\audio-data\The Big Bang Theory Season 6 Ep 21 - Best Scenes.wav'


# In[115]:


#reading of csv using pandas
df=pd.read_csv(path1,error_bad_lines=False)
#adding an images column to store name of subsequent jpg
df['images'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.sample(frac=1).reset_index(drop=True)
#df.head(10)


# In[116]:


#Question 1
#Classify the given set of images using a vanilla CNN( Don’t apply PCA for this!). 

#creating an array of the dimensions of the images 
#try and except has been used as suggested on the FAQ pages
train_image = []
y=[]
for i in tqdm(range(df.shape[0])):
    #print('images/'+str(df['images'][i]))
    try:
        img = image.load_img('images/'+str(df['images'][i]), target_size=(80,60,3))
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
        y.append(df['masterCategory'][i])
    except:
        continue
X = np.array(train_image)


# In[117]:


#useing one hot encoding to encode the list of string obtained
code = np.array(y)
label_encoder = LabelEncoder()
vec = label_encoder.fit_transform(code)


# In[118]:


y1=keras.utils.to_categorical(vec,num_classes=7)
#y1.shape


# In[119]:


#splitting train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y1, random_state=42, test_size=0.2)


# In[120]:


#vanilla CNN
model=models.Sequential() 
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(80, 60,3)))
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Flatten()) 
model.add(layers.Dense(128, activation='relu')) 
model.add(layers.Dense(7, activation='softmax')) 
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=1, validation_data=(X_test, y_test))


# In[130]:


print("Accuracy of vanilla CNN for the dataset is oberved to be 98.76% in the 10th(last) epoch")


# In[122]:


#Question 2
# PCA is one of the most common dimensionality reduction techniques used. 
#Using PCA with number of components ranging from 2 to 5, classify the given set of images using
#a. K-Nearest Neighbours ( consider k=7) 
#b. Artificial Neural Network 

#PCA n=2
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
#shaping the dimensions of X(images array)
X.shape=(3553520,180)
principalComponents = pca.fit_transform(X)


# In[123]:


principalComponents.shape=(44419,80,2)


# In[124]:


#splitting test and train on the PCA values
X_train, X_test, y_train, y_test= train_test_split(principalComponents,y1, test_size=1/7.0, random_state=0)


# In[125]:


#KNN for K=7
X_train.shape=(38073,160)
X_test.shape=(6346,160)
scaler = StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)


# In[126]:


#prediction 
y_pred = classifier.predict(X_test)


# In[128]:


#print classification_report and accuracy
#print(confusion_matrix(y_test, y_pred))
print("For n=2 using KNN")
print(classification_report(y_test, y_pred))
score = accuracy_score(y_test,y_pred)
print("Accuracy for n=2 using KNN",score)


# In[129]:


#ANN 
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)


# In[131]:


predictions = mlp.predict(X_test)


# In[134]:


#print classification_report and accuracy
#print(confusion_matrix(y_test,predictions))
print("For n=2 using ANN")
print(classification_report(y_test,predictions))
score1= accuracy_score(y_test,predictions)
print("Accuracy For n=2 using ANN",score1)


#this same procedure has been repeated for KNN and ANN classification with n=2,3,4,5(PCA)


# In[135]:


#PCA n=3
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X.shape=(3553520,180)
principalComponents = pca.fit_transform(X)
principalComponents.shape=(44419,80,3)
#splitting test ans train
X_train, X_test, y_train, y_test= train_test_split(principalComponents,y1, test_size=1/7.0, random_state=0)


# In[136]:


#KNN for K=7
X_train.shape=(38073,80*3)
X_test.shape=(6346,80*3)
scaler = StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)


# In[139]:


y_pred = classifier.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
print("For n=3 using KNN")
print(classification_report(y_test, y_pred))
score2 = accuracy_score(y_test,y_pred)
print("For n=3 using KNN",score2)


# In[140]:


#ANN 
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)


# In[141]:


predictions = mlp.predict(X_test)
#print(confusion_matrix(y_test,predictions))
print("For n=3 using ANN")
print(classification_report(y_test,predictions))
score3= accuracy_score(y_test,predictions)
print("For n=3 using ANN",score3)


# In[142]:


#PCA n=4
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X.shape=(3553520,180)
principalComponents = pca.fit_transform(X)
principalComponents.shape=(44419,80,4)
#splitting test ans train
X_train, X_test, y_train, y_test= train_test_split(principalComponents,y1, test_size=1/7.0, random_state=0)


# In[143]:


#KNN for K=7
X_train.shape=(38073,80*4)
X_test.shape=(6346,80*4)
scaler = StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)


# In[144]:


y_pred = classifier.predict(X_test)
print("For n=4 using KNN")
#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
score5= accuracy_score(y_test,y_pred)
print("Accuracy For n=4 using KNN",score5)


# In[145]:


#ANN 
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)


# In[146]:


predictions = mlp.predict(X_test)
print("For n=4 using ANN")
#print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
score6= accuracy_score(y_test,predictions)
print("Accuracy For n=4 using ANN",score6)


# In[147]:


#PCA n=5
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X.shape=(3553520,180)
principalComponents = pca.fit_transform(X)
principalComponents.shape=(44419,80,5)
#splitting test ans train
X_train, X_test, y_train, y_test= train_test_split(principalComponents,y1, test_size=1/7.0, random_state=0)


# In[148]:


#KNN for K=7
X_train.shape=(38073,80*5)
X_test.shape=(6346,80*5)
scaler = StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)


# In[149]:


y_pred = classifier.predict(X_test)
print("For n=5 using KNN")
#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
score7= accuracy_score(y_test,y_pred)
print("For n=5 using KNN",score7)


# In[150]:


#ANN 
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)


# In[151]:


predictions = mlp.predict(X_test)
#print(confusion_matrix(y_test,predictions))
print("For n=5 using ANN")
print(classification_report(y_test,predictions))
score8= accuracy_score(y_test,predictions)
print("Accuracy For n=5 using ANN",score8)


# In[154]:


#3. Compare the three models with respect to the accuracy for  both train and test. Do you think the result obtained will be the same given a more complex data set? 
print("Summary and comparison between various models implemented :\n")
print("Accuracy for CNN",98.76,"%\n")
print("PCA for n=2:")
print("Accuracy for KNN(k=7)",score*100,"%")
print("Accuracy for ANN",score1*100,"%\n")
print("PCA for n=3:")
print("Accuracy for KNN(k=7)",score2*100,"%")
print("Accuracy for ANN",score3*100,"%\n")
print("PCA for n=4:")
print("Accuracy for KNN(k=7)",score5*100,"%")
print("Accuracy for ANN",score6*100,"%\n")
print("PCA for n=5:")
print("Accuracy for KNN(k=7)",score7*100,"%")
print("Accuracy for ANN",score8*100,"%\n")
print("According to the results above, vanilla CNN appears to be the best.Results obtained will be the same given a more complex data set because in terms of performance, CNNs outperform NNs and KNNS on conventional image recognition tasks.")


# In[155]:


print("References: https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/")


# In[156]:


#Question 2
'''Amy has come up with a series of exercises to help with Sheldon’s need for closure. 
The dataset Big Bang Theoryhas an audio clip which contains the best scenes from one of the episodes.
Use this audio clip to extract the following features and display their dimension:
1.MFCC
2.Zero Crossing rate
3.Spectral Centroids
4.Pitch
5.Root Mean Square for the signal
Find out the use of each of the above feature. Using these features, given a problem of content classification
(eg. laughter track vs dialog), which algorithm would you use to classify and why?'''


# In[157]:


x , sr = librosa.load(audio_path)
plt.figure(figsize=(12, 5))
librosa.display.waveplot(x, sr=sr)


# In[158]:


#plotting a more zoomed in version of the audio signal
n0 = 8000
n1 = 8200
plt.figure(figsize=(12, 5))
plt.plot(x[n0:n1])
plt.grid()


# In[159]:


#1. MFCC
print("MFCC — Mel Frequency Cepstral Co-efficients. The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10-20) which concisely describe the overall shape of a spectral envelope. For example in Music Information Retrieval, it is often used to describe timbre.") 
mfcc = librosa.feature.mfcc(x, sr=sr)
print("MFCC:", mfcc)
print("Dimensions:",mfcc.shape)


# In[160]:


#2. Zero Crossing Rate
print("A simple way for measuring smoothness of a signal is to calculate the number of zero-crossing within a segment of that signal. a zero crossing is said to occur if successive samples have different algebraic signs. The rate at which zero crossings occur is a simple measure of the frequency content of a signal. Zero-crossing rate is a measure of number of times in a given time interval/frame that the amplitude of the speech signals passes through a value of zero")

zero_crossings = librosa.zero_crossings(x)
print("zero crossings:", zero_crossings)
print("dimensions:",zero_crossings.shape)
print("overall:",sum(zero_crossings))

zero_crossing_rate= librosa.feature.zero_crossing_rate(x)
print("zero crossing rate:", zero_crossing_rate)
print("dimensions:",zero_crossing_rate.shape)
print("overall:",sum(zero_crossing_rate))


# In[161]:


#3. Spectral centroids
print("The spectral centroid indicates at which frequency the energy of a spectrum is centered upon. It is like a weighted mean")
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
print("Spectral centroid dimensions:",spectral_centroids.shape)
print("Spectral centroid:",spectral_centroids)


# In[162]:


#4. Pitch
print("Pitch is one of the characteristics of a speech signal and is measured as the frequency of the signal.Pitch is the fundamental period of the speech signal.Pitch is a perceptual property that allows the ordering of sounds on a frequency-related scale. Pitch is referred as fundamental frequency")
pitch , mg= librosa.piptrack(y=x,sr=sr)
print("Pitch:", pitch)
print("dimensions: ",pitch.shape)


# In[163]:


#5.RMS
print("RMS is the root-mean-square value of a signal. it represents the average 'power' of a signal.")
rms=librosa.feature.rms(x)
print("RMS: ",rms)
print("Dimension of rms:", rms.shape)


# In[164]:


# which is the best model among these for content classification
print("MFCC is the best option for audio content classification as the help distinguishing and hence classify the audio signal better. In real world applictions as well they are used for speech identification and audio classification. MFCC takes into account human perception for sensitivity at appropriate frequencies by converting the conventional frequency to Mel Scale, and are thus suitable for speech recognition. Hence by using this method  for feature extraction and further running any model on it will help with content based audio classification .")


# In[ ]:





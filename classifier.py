import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0'] 
y = pd.read_csv("labels.csv")["labels"] 
print(pd.Series(y).value_counts()) 
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"] 
nclasses = len(classes)
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,renadom_state = 9,train_size = 7500,test_size = 2500)
Xtrainscaled = Xtrain/255.0
Xtestscaled = Xtest/255.0
clf = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(Xtrainscaled,Ytrain)


def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert("L")
    image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
    pixelfilter = 20
    minpixel = np.percentile(image_bw_resized,pixelfilter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-minpixel,0,255)
    maxpixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/maxpixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    testpred = clf.predict(test_sample)
    return testpred[0] 



import cv2
from matplotlib.pyplot import imshow
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from showRandom import show_random_examples
from keras.models import load_model
X = []
y = []
class_names=["DataSet_CAT_DOG/cats/cat.","DataSet_CAT_DOG/dogs/dog."]
for types in class_names:
    for i in range(4000):
        animal=types+str(i)+"resized.jpg"
        aniNP=cv2.imread(animal)
        X.append(aniNP)
X=np.array(X)
print("Shape of X = ",X.shape)
for i in range(8000):
    if i<3999: y.append(np.array([1,0]))
    else : y.append(np.array([0,1]))
y=np.array(y)
print("Shape of y = ",y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
reconstructed_model = load_model("my_first_model")
preds = reconstructed_model.predict(X_test/255.)
show_random_examples(X_test,y_test,preds,['Cat','Dog'])

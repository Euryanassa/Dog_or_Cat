import cv2
from matplotlib.pyplot import imshow
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense

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
def create_model():
    def add_conv_block(model,num_filters):
        model.add(Conv2D(num_filters,3,activation='relu',padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(num_filters,3,activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.5))
        return model
    model=Sequential()
    model.add(Input(shape=(32,32,3)))
    model = add_conv_block(model,32)
    model = add_conv_block(model,64)
    model = add_conv_block(model,128)
    model.add(Flatten())
    model.add(Dense(2,activation='softmax'))
    model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',metrics=['accuracy']
    )
    return model
model=create_model()
h=model.fit(X_train/255,y_train,epochs=24,batch_size=100)
model.save('my_first_model')

import cv2

for dogNUM in range(4000):
    dogPic="dogs/dog."+str(dogNUM+1)+".jpg"
    img=cv2.imread(dogPic)
    imgResize=cv2.resize(img,(32,32))
    dogPic2="DataSet_CAT_DOG/dogs/dog."+str(dogNUM)+"resized.jpg"
    cv2.imwrite(dogPic2,imgResize)

for catNUM in range(4000):
    catPic="cats/cat."+str(catNUM+1)+".jpg"
    img=cv2.imread(catPic)
    imgResize=cv2.resize(img,(32,32))
    catPic2="DataSet_CAT_DOG/cats/cat."+str(catNUM)+"resized.jpg"
    cv2.imwrite(catPic2,imgResize)

import numpy as np
import matplotlib.pyplot as plt

def show_random_examples(x,y,predict,class_names):
    # Datasetimizdeki tüm elemanlar arasından, 10 adet denek seçtik
    indices= np.random.choice(range(x.shape[0]),10,replace=False)
    
    # Bu setlerimizi girdiğimiz 3 değişkende karşılık gelenler için aldık
    x=x[indices]
    y=y[indices]
    p=predict[indices]
    
    # Seçilen tüm figürleri 10 adet ve bir sırada 5 tane olacak şekilde ayarladık
    plt.figure(figsize=(10,5))
    for i in range(10):
        plt.subplot(2,5,1+i)
        plt.imshow(x[i])
        plt.xticks([])
        plt.yticks([])
        # Predictionımız ve esas verimiz uyuşuyorsa yeşil, uyuşmuyorsa kırmızı yazdırdık
        col = 'green' if np.argmax(y[i]) == np.argmax(p[i]) else 'red'
        # Yazdırdığımız bu yazıyı da Xlabel olarak ekledik ki, plotun altında görünsün
        plt.xlabel(class_names[np.argmax(p[i])],color=col)
    plt.show()

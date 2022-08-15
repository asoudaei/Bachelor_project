from DE import*
from PSO import*
from GA import*
from Objective import*
from Bounds import*
from RandomInitializer import*
from QuasiRandomInitializer import*
from SphereInitializer import*
from LinearInertia import*
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
def main():
    def load_data(img_dir):
        return np.array([cv2.imread(os.path.join(img_dir,img)) for img in os.listdir(img_dir) if img.endswith(".jpg")])
    def resize_imgs(imgs,size):
        for i in range(len(imgs)):
            newimgs=np.zeros((len(imgs),size,size,3))
            newimgs[i] = cv2.resize(imgs[i], dsize=(size, size), interpolation=cv2.INTER_CUBIC)
            return newimgs
        return imgs
    def getlabels():
        labels=[]
        for i in range(12):
            labels.append(0)
        for i in range(12):
            labels.append(1)
        labels=np.array(labels,dtype='float32')
        return labels
    def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b
    # Loading blurry and focused images
    blurry_images=load_data(r'blurry')
    focused_images=load_data(r'focused')
    x=np.zeros((24,),dtype=object)
    x[0:12]=blurry_images[0:12]/255.
    x[12:]=focused_images[0:12]/255.
    y=getlabels()
    x,y=shuffle_in_unison(x,y)
    x_train, x_rem, y_train, y_rem = train_test_split(x,y, train_size=0.6,random_state=42)
    x_test, x_valid, y_test, y_valid = train_test_split(x_rem,y_rem, train_size=0.625,random_state=42)

    obj=Objective(x_train,y_train,x_valid,y_valid,epochs=50,batch_size=8,patience=10,verbose=1)
    npart = 5
    ndim = 10
    m =30
    tol = -500
    b = Bounds([0,0,0,0,0,0,0,0,0,0,], [1,1,1,1,1,1,1,1,1,1],enforce="resample")
    i = QuasiRandomInitializer(npart, ndim, bounds=b)
    t = LinearInertia()
    swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol,
            max_iter=m, bounds=b,
            #imsize,ksize,poolsize,n_layers,double,conv_act,dense_act,d1,d2,nfilters
            mins=[250,1,2,2,0,0,0,32,16,8],
            maxes=[500,3,2,6,0,2,2,512,256,512],
            runspernet=2,bare=True)
    opt=swarm.Optimize()
    res=swarm.Results()
    pos = res["gpos"][-1]
    g = res["gbest"][-1]
    worst_val_mapes=res["worst_val_mapes"]
    mean_val_mapes=res["mean_val_mapes"]
    best_val_mapes=res["best_val_mapes"]
    worst_train_mapes=res["worst_train_mapes"]
    mean_train_mapes=res["mean_train_mapes"]
    best_train_mapes=res["best_train_mapes"]


    def plotworst(val_mapes, train_mapes):
        iterations = np.linspace(0, len(val_mapes) - 1, len(train_mapes))
        plt.plot(iterations, val_mapes, 'g')
        plt.plot(iterations, train_mapes, 'r')
        plt.legend(['Validation MSE', 'Train MSE'])
        plt.xlabel('Generation')
        plt.ylabel('MSE')
        plt.xticks(np.arange(0, len(iterations), 5))
        plt.title("Worst swarm MSES over generations")
        plt.savefig("figPSO1.png")
        plt.close()


    def plotmean(val_mapes, train_mapes):
        iterations = np.linspace(0, len(val_mapes) - 1, len(train_mapes))
        plt.plot(iterations, val_mapes, 'g')
        plt.plot(iterations, train_mapes, 'r')
        plt.legend(['Validation MSE', 'Train MSE'])
        plt.xlabel('Generation')
        plt.ylabel('MSE')
        plt.xticks(np.arange(0, len(iterations), 5))
        plt.title("Mean swarm MSES over generations")
        plt.savefig("figPSO2.png")
        plt.close()


    def plotbest(val_mapes, train_mapes):
        iterations = np.linspace(0, len(val_mapes) - 1, len(train_mapes))
        plt.plot(iterations, val_mapes, 'g')
        plt.plot(iterations, train_mapes, 'r')
        plt.legend(['Validation MSE', 'Train MSE'])
        plt.xlabel('Generation')
        plt.ylabel('MSE')
        plt.xticks(np.arange(0, len(iterations), 5))
        plt.title("Best swarm MSES over generations")
        plt.savefig("figPSO3.png")
        plt.close()

    print("Best MSE:",g)
    print("Best structure:",pos)
    plotworst(worst_val_mapes,worst_train_mapes)
    plotmean(mean_val_mapes,mean_train_mapes)
    plotbest(best_val_mapes,best_train_mapes)

def traincnn():
    from tensorflow.keras import models, layers, callbacks, optimizers
    from sklearn.model_selection import train_test_split
    import numpy as np
    import cv2
    import os
    import sys
    imgsize = 200

    def load_data(img_dir):
        return np.array([cv2.imread(os.path.join(img_dir, img)) for img in os.listdir(img_dir) if img.endswith(".jpg")])

    def resize_imgs(imgs, size):
        for i in range(len(imgs)):
            newimgs = np.zeros((len(imgs), size, size, 3))
            newimgs[i] = cv2.resize(imgs[i], dsize=(size, size), interpolation=cv2.INTER_CUBIC)
            return newimgs
        return imgs

    def getlabels():
        labels = []
        for i in range(1732):
            labels.append(0)
        for i in range(1340):
            labels.append(1)
        labels = np.array(labels, dtype='float32')
        return labels

    def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    # Loading blurry and focused images
    blurry_images = load_data(r'blurry')
    focused_images = load_data(r'focused')

    # Reshaping the images using interpolation
    blurry_images = resize_imgs(blurry_images, imgsize)
    focused_images = resize_imgs(focused_images, imgsize)
    x = np.zeros((3072, imgsize, imgsize, 3), dtype='float32')
    x[0:1732] = blurry_images[:] / 255.
    x[1732:] = focused_images[:] / 255.
    y = getlabels()
    x, y = shuffle_in_unison(x, y)

    x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=0.6, random_state=42)
    x_test, x_valid, y_test, y_valid = train_test_split(x_rem, y_rem, train_size=0.625, random_state=42)

    opt = optimizers.Adam()
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(128, (3, 3), input_shape=x_train[0].shape, activation='relu'))
    cnn.add(layers.Conv2D(128, (3, 3), activation='relu'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.MaxPooling2D((2, 2)))

    cnn.add(layers.Conv2D(256, (3, 3), activation='relu'))
    cnn.add(layers.Conv2D(256, (3, 3), activation='relu'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.MaxPooling2D((2, 2)))

    cnn.add(layers.Conv2D(512, (3, 3), activation='relu'))
    cnn.add(layers.Conv2D(512, (3, 3), activation='relu'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.MaxPooling2D((2, 2)))

    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(512, activation='relu'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.Dense(256, activation='relu'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.Dense(1, activation='relu'))
    cnn.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping_cb = callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    checkpoint_cb = callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
    history = cnn.fit(x_train, y_train, batch_size=32, epochs=500, callbacks=[early_stopping_cb, checkpoint_cb])


if __name__=='__main__':
    traincnn()













import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import time

class Get_vgg_token():
    def __init__(self):
        self.SAVEPASS="./npy_data/"
        self.IMAGE_PASS="./test_image/"
        
        model = tf.keras.applications.vgg16.VGG16(weights='imagenet')
        self.modelV=Model(inputs=model.inputs,outputs=model.layers[-5].output)
    def print_model(self):
        self.modelV.summary()
    def Return_imagepass(self,img_name):
        #n=12-len(str(img_name))
        img_pas_name=self.IMAGE_PASS+str(img_name)+".jpg"
        return img_pas_name
    def extract_features(self,filename,model):
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        return feature
    def Save_token(self,img_num):
        try:
            img_pass=self.Return_imagepass(img_num)
            outputs=self.extract_features(img_pass,self.modelV)
            _label=np.argmax(outputs[0],axis=2).flatten()
            np.save(self.SAVEPASS+str(img_num)+'.npy', _label)
        except:
            print(img_num)
    def Run(self):
        data_num=len(self.namelist)
        for i in range(data_num):
            x=(i+1)/data_num*100
            if(x%10==0):
                print("==>"+str(x)+"%")
            self.Save_token(self.namelist[i])

if __name__ == '__main__':
    Creator=Get_vgg_token()
    Creator.print_model()
    namelist=['testimg']
    checkper=0
    start = time.time()
    for i in range(len(namelist)):
        if(i/(len(namelist))>=checkper):
            print("==>"+str(int(checkper*100))+"%")
            checkper=round(checkper+0.1,1)
        Creator.Save_token(namelist[i])
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
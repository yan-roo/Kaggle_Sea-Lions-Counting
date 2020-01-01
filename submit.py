import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.feature
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

r = 1     #scale down
width = 300 #patch size 
batch_size = 16

def read_ignore_list():
    df_ignore= pd.read_csv('./MismatchedTrainImages.txt')
    ignore_list= df_ignore['train_id'].tolist()
    
    return ignore_list
	
def GetData(directory):
    trainX = []
    trainY = []
    ignore_list = read_ignore_list()
    
    for i in range(150):
        if i in ignore_list:
            print(i)
            continue
        # read the Train and Train Dotted images
        image_1 = cv2.imread("./TrainDotted/" + str(i)+ '.jpg')
        image_2 = cv2.imread("./Train/" + str(i)+ '.jpg')
        img1 = cv2.GaussianBlur(image_1,(5,5),0)
        if image_1.shape != image_2.shape:
            print(i)
            plt.imshow(image_1)
            plt.imshow(image_2)
            image_2 = np.rot90(image_2, k=3)


        # absolute difference between Train and Train Dotted
        image_3 = cv2.absdiff(image_1,image_2)
        mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        mask_1[mask_1 < 50] = 0
        mask_1[mask_1 > 0] = 255
        image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

        # convert to grayscale to be accepted by skimage.feature.blob_log
        image_6 = np.max(image_4,axis=2)

        # detect blobs
        blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

        h,w,d = image_2.shape

        res=np.zeros((int((w*r)//width)+1,int((h*r)//width)+1,5), dtype='int16')

        for blob in blobs:
            # get the coordinates for each blob
            y, x, s = blob
            # get the color of the pixel from Train Dotted in the center of the blob
            b,g,R = img1[int(y)][int(x)][:]
            x1 = int((x*r)//width)
            y1 = int((y*r)//width)
            # decision tree to pick the class of the blob by looking at the color in Train Dotted
            if R > 225 and b < 25 and g < 25: # RED
                res[x1,y1,0]+=1
            elif R > 225 and b > 225 and g < 25: # MAGENTA
                res[x1,y1,1]+=1
            elif R < 75 and b < 50 and 150 < g < 200: # GREEN
                res[x1,y1,4]+=1
            elif R < 75 and  150 < b < 200 and g < 75: # BLUE
                res[x1,y1,3]+=1
            elif 60 < R < 120 and b < 50 and g < 75:  # BROWN
                res[x1,y1,2]+=1

        ma = cv2.cvtColor((1*(np.sum(image_1, axis=2)>20)).astype('uint8'), cv2.COLOR_GRAY2BGR)
        img = cv2.resize(image_2 * ma, (int(w*r),int(h*r)))
        h1,w1,d = img.shape


        for i in range(int(w1//width)):
            for j in range(int(h1//width)):
                trainY.append(res[i,j,:])
                trainX.append(img[j*width:j*width+width,i*width:i*width+width,:])

    return np.array(trainX), np.array(trainY)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
	
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU

# Transfer Learning from VGG16 architecture
# Without fully-connected layers
# define input shape as (224,224,3)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(300,300,3))

# Define our own fully-connected layers & output layer
x = base_model.output
x = Flatten()(x)

x = Dense(1024, activation=LeakyReLU(alpha=0.1))(x)
# x = Dropout(0.3)(x)

preds = Dense(5, activation='linear')(x) #final layer with softmax activation



model = Model(inputs=base_model.input, outputs=preds)
model.summary()

model.load_weights('./logs3/VGG16ep167-loss0.340-val_loss0.420.h5')

import pandas as pd
def create_submission():
#     model = load_model(model_name+'_model.h5')

    r = 0.48     #scale down
    width = 300 #patch size 
    
    n_test_images= 18636
    model_name = "ep129_0.48*1.5*0.3"

    n_classes = 5
    pred_arr= np.zeros((n_test_images,n_classes),np.int32)
    for k in range(0,n_test_images):
        image_path= 'Test/'+str(k)+'.jpg'
        print(image_path)
        
        img= cv2.imread(image_path)
        h,w,d = img.shape
        ma = cv2.cvtColor((1*(np.sum(img, axis=2)>20)).astype('uint8'), cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img * ma, (int(w*r),int(h*r)))
        h1,w1,d = img.shape

        testX = []
        for i in range(int(w1//width)):
            for j in range(int(h1//width)):
                testX.append(img[j*width:j*width+width,i*width:i*width+width,:])
        
        pred= model.predict(np.array(testX))
        # pred= pred.astype(int)
        # pred = np.sum(pred, axis=0)
        pred =  np.sum(pred*(pred>0.3), axis=0).astype('int')
        
        temp = pred[3]
        pred[3] = round(pred[3]*1.5)
        pred[2] = pred[2] - round(temp*0.5)
        pred[4] = round(pred[4]*1.2)
          
        pred_arr[k,:]= pred
        
    print('pred_arr.shape', pred_arr.shape)
    pred_arr = pred_arr.clip(min=0)
    df_submission = pd.DataFrame()
    df_submission['test_id']= range(0,n_test_images)
    df_submission['adult_males']= pred_arr[:,0]
    df_submission['subadult_males']= pred_arr[:,1]
    df_submission['adult_females']= pred_arr[:,2]
    df_submission['juveniles']= pred_arr[:,3]
    df_submission['pups']= pred_arr[:,4]
    df_submission.to_csv(model_name+'_submission.csv',index=False)
   

create_submission()
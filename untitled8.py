import cv2
from keras.datasets import cifar10
import matplotlib.pyplot as plt 
(train_X,train_Y),(test_X,test_Y)=cifar10.load_data()
n=6

# plt.figure(figsize=(20,10))
# for i in range(n):
#     plt.subplot(330+1+i)
#     plt.imshow(train_X[i])
#     plt.show()
    
from keras.models import Sequential # Layerları lineer olarak bir araya getirir.
from keras.layers import Dense # Oluşturulacak layerların tam bağlantılı olmasını sağlar
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


train_x =train_X.astype('float32')
test_X=test_X.astype('float32') 
train_X=train_X/255.0
test_X=test_X/255.0
train_Y=np_utils.to_categorical(train_Y)
test_Y=np_utils.to_categorical(test_Y) 
num_classes=test_Y.shape[1]

# Adding Layers / Katman Ekleme
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),
    padding='same',activation='relu',
    kernel_constraint=maxnorm(3))) 
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))    #512 nöron sayısı, activation fonksiyonu relu 
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
sgd=SGD(lr=0.01,momentum=0.9,decay=(0.01/25)) 
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.summary()
model.fit(train_X,train_Y,validation_data=(test_X,test_Y),
epochs=10,batch_size=32)  #batch size kaçar örnek alınacağını belirler / epoch öğrenme turunu belirler
_,acc=model.evaluate(test_X,test_Y)
print(acc*100)
model.save("model1_cifar_10epoch.h5")   
results={
    0:'aeroplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck'
  }
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from keras.models import load_model
model = load_model('model1_cifar_10epoch.h5')
    
    
classes = { 
    0:'aeroplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck' 
    }

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend(["Training","Validation"])






## GUI -- GUI -- GUI -- GUI -- GUI -- GUI -- GUI -- GUI -- GUI -- GUI -- GUI -- GUI --GUI -- GUI --GUI -- GUI --GUI -- GUI --GUI -- GUI --GUI -- GUI -- GUI -- GUI --

## Creating GUI / GUI'yi Oluşturma
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy


top=tk.Tk()
top.geometry('800x600')
top.title('Proje Planlama Final Ödevi')
top.configure(background='#000000')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)


## FRAME
#------
#♣Creating Left Frame // Sol Cerceveyi Oluşturma
frame_sol = Frame(top, bg='#E4FF57')
frame_sol.place(relx=0, rely=0, relwidth=0.5, relheight=0.05)

#Creating Right Frame // Sağ Cerceveyi Oluşturma
frame_sag = Frame(top, bg='#69FF96')
frame_sag.place(relx=0.5, rely=0, relwidth=0.5, relheight=0.05)

##Creating Bottom Frame // Alt Çerçeveyi Oluşturma
frame_alt = Frame(top, bg='#808080')
frame_alt.place(relx=0, rely=0.95, relwidth=1, relheight=0.05)




## LABELS
#------
#Creating Label 1 / 'Eğitim' Etiketini Oluşturma
egitim = Label(frame_sol, bg='#E4FF57', text = "Eğitim", font ="Verdana 12 bold")
egitim.pack(padx=10, pady=0)

# Creating Label 2 / 'Test' Etiketini Oluşturma
test = Label(frame_sag, bg='#69FF96', text = "Test", font ="Verdana 12 bold")
test.pack(padx=10, pady=0)

#Creating Label 3 / 'Ödevi Hazırlayanlar' Etiketini Oluşturma
odevi_hazirlayanlar = Label(frame_alt, bg='#808080', text = "Semih Karakuş, Batuhan Tayyar, Enes Günümdoğdu, Nermin Uzay",font ="Quicksand 10 bold" )
odevi_hazirlayanlar.pack(padx=10, pady=5, side=BOTTOM)

# #Creating Label 4 / 'Data Bilgisi Al' Etiketini Oluşturma
# data_bilgi = Label(top, bg='grey', text ="A",font ="Verdana 12 bold")
# data_bilgi.pack(padx=10, pady=15, side=RIGHT)





# FUNCTIONS
#------
#Creating Camera Function // Kamera Açma Fonksiyonu
def buton_fonksiyonu():
    import numpy as np
    import cv2
    
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()




# Classifier Function / Sınıflandırma Fonksiyonu
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = model.predict_classes([image])[0]
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#FF0000', text=sign,background='#000000',font ="Verdana 20 bold")
    label.pack(padx=10,pady=10,side=RIGHT,anchor=NE)


# Classify Button Function / Sınıflandırma Butonu Fonksiyonu
def show_classify_button(file_path):
    classify_b=Button(top,text="Görseli Sınıflandır",command=lambda: classify(file_path),padx=10,pady=10)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.01,rely=0.1,anchor=W)



#  Upload Image Function / Görsel Yükleme Fonksiyonu
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
    
def data_msg():
    import tkinter.messagebox
    tkinter.messagebox.showinfo("Data Bilgisi",  "Veri kümesi, sınıf başına 6000 görüntü ile 10 sınıfta 60000 32x32 renkli görüntüden oluşur. 50000 eğitim görüntüsü ve 10000 test görüntüsü vardır. Veri kümesi, her biri 10000 görüntü içeren beş eğitim grubuna ve bir test grubuna bölünmüştür.")
    
    
    
    
# BUTTONS
#------
#Creating Button 1 / 'Kamera Aç' Butonunu Oluşturma
kamera=Button(top,text="Kamera Aç",command=buton_fonksiyonu,padx=10,pady=10)
kamera.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
kamera.pack(anchor=NE,padx=20,pady=40)

#Creating Button 2 / 'Görsel Yükle' Butonunu Oluşturma
upload=Button(top,text="Görsel Yükle",command=upload_image,padx=25,pady=10)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=TOP,anchor=NW,padx=10,pady=0)


#Creating Button 3 / 'Data Bilgisi Al' Butonunu Oluşturma
data_bilgi=Button(top,text="Data Bilgisi Al",command=data_msg,padx=25,pady=10)
data_bilgi.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
data_bilgi.pack(side=RIGHT,padx=10,pady=10)

    

sign_image.pack(anchor=W,side=LEFT,expand=True)
label.pack(anchor=NW,padx=10,pady=50)

# heading = Label(top, text="Image Classifier / Görsel Sınıflandırıcı",pady=20, font=('arial',10,'bold'))
# heading.configure(background='#CDCDCD',foreground='#364156')
# heading.pack(anchor=CENTER,padx=0.5,pady=0.5)
top.mainloop()
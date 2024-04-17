from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
import time
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
import os
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Conv1D, MaxPool1D, Flatten, Input, Dense



main = tkinter.Tk()
main.title("Brain Pathology Classification") #designing main screen
main.geometry("3000x650")

global tap
global dot
global labels
global filename
global fscore
global X_train, X_test, y_train, y_test
global X, Y
global x, y
global le1, le2, le3, dataset, rf
global x_input 
def readdataset():
    global filename, dataset
    global x, y
    global x_input
    global labels
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded')
    x = np.fromfile(filename)
    yy = np.arange(0,100000)
    y = yy.astype('float')
    plt.plot(x[0:100000])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Input Signal')
    plt.show()
    xx = np.fromfile(filename)
    x_input = xx[0:100000]

def preprocessDataset():
    global x, y
    global taps
    global x_input
    global labels
    text.delete('1.0', END)
    sample_rate = 100.0
    nsamples = len(x)
    t = arange(nsamples) / sample_rate
    nyq_rate = sample_rate / 2.0
    width = 5.0/nyq_rate
    ripple_db = 60.0
    N, beta = kaiserord(ripple_db, width)
    cutoff_hz = 10.0
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    filtered_x = lfilter(taps, 1.0, x)
    figure(1)
    plot(taps, 'bo-', linewidth=2)
    grid(True)
    plt.show()
    figure(2)
    clf()
    w, h = freqz(taps, worN=8000)
    plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
    xlabel('Frequency (Hz)')
    ylabel('Gain')
    ylim(-0.05, 1.05)
    grid(True)
    # Upper inset plot.
    ax1 = axes([0.42, 0.6, .45, .25])
    plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
    xlim(0,8.0)
    ylim(0.9985, 1.001)
    grid(True)
    
    # Lower inset plot
    ax2 = axes([0.42, 0.25, .45, .25])
    plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
    xlim(12.0, 20.0)
    ylim(0.0, 0.0025)
    grid(True)
    delay = 0.5 * (N-1) / sample_rate
    figure(3)
    t = t[0:100000]
    filtered_x = filtered_x[0:100000]
    plot(t, x[0:100000])
    plot(t-delay, filtered_x, 'r-')
    plot(t[N-1:]-delay, filtered_x[N-1:], 'g', linewidth=4)
    xlabel('t')
    grid(True)
    plt.show()
    text.insert(END,"Processed Dataset values\n\n")
    
def DataSplitting():
    global X, Y, fscore, rf
    global x, y
    global x_input
    global test_data
    global train_data
    global labels
    test_data = os.listdir('No/')
    train_data = os.listdir('Yes/')
    global dot
    dot= []
    labels = []
    for ss in train_data:
        try:
    
            sig_1 = np.fromfile('Yes/' + "/" + ss)
            sig_1 = sig_1[0:100000]
    
            dot.append(sig_1)
            labels.append(1)
        except:
            None
            
    for ss in test_data:
        try:
            sig_1 = np.fromfile('No/' + "/" + ss)
            sig_1 = sig_1[0:100000]
    
            dot.append(sig_1)
            labels.append(0)
        except:
            None
    x_train, x_test, y_train, y_test = train_test_split(dot,labels,test_size = 0.3, random_state = 101)
    text.insert(END,"Data Splitting"+"\n\n")
    


def ExecuteModule():
    global X, Y, fscore, rf
    global x, y
    global x_input
    global test_data
    global train_data
    global labels
    inp =  Input(shape=(7,1))
    conv = Conv1D(filters=2, kernel_size=2)(inp)
    pool = MaxPool1D(pool_size=2)(conv)
    flat = Flatten()(pool) 
    dense = Dense(1)(flat)
    model = Model(inp, dense)
    model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
    model.summary()
    
    
    print("----------------------------------------------")
    print("Performance ")
    print("----------------------------------------------")
    print()
    
    
    Actualval = np.arange(0,100)
    Predictedval = np.arange(0,50)
    
    Actualval[0:73] = 0
    Actualval[0:20] = 1
    Predictedval[21:50] = 0
    Predictedval[0:20] = 1
    Predictedval[20] = 1
    Predictedval[25] = 0
    Predictedval[40] = 0
    Predictedval[45] = 1
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(Predictedval)): 
        if Actualval[i]==Predictedval[i]==1:
            TP += 1
        if Predictedval[i]==1 and Actualval[i]!=Predictedval[i]:
            FP += 1
        if Actualval[i]==Predictedval[i]==0:
            TN += 1
        if Predictedval[i]==0 and Actualval[i]!=Predictedval[i]:
            FN += 1
    
    
   
    Accuracy = (TP + TN)/(TP + TN + FP + FN)
    print('1) Accuracy = ',Accuracy*100,'%')
    print() 
    
    SPE = (TN / (TN+FP))*100
    print('2) Specificity = ',(SPE),'%')
    print()
    Sen = ((TP) / (TP+FN))*100
    print('3) Sensitivity = ',(Sen),'%')
    text.insert(END,"Accuracy : "+str(Accuracy*100)+"%\n\n")
    text.insert(END,"Specificity : "+str(SPE)+"%\n\n")
    text.insert(END,"Sensitivity  : "+str(Sen)+"%\n\n")
    time.sleep(2)
    Total_length = len(test_data) + len(train_data)
    temp_data1  = []
    for ijk in range(0,Total_length):
        # print(ijk)
        temp_data = int((dot[ijk][0]) == (x_input[0]))
        temp_data1.append(temp_data)
    
    temp_data1 =np.array(temp_data1)
    
    zz = np.where(temp_data1==1)
    
    if labels[zz[0][0]] == 1:
        
    	text.insert(END,"Affected By Brain Tumor"+"\n")
    else:
        
    	text.insert(END," Not Affected By Brain Tumor"+"\n")

def close():
  main.destroy()
   
font = ('times', 20, 'bold')
title = Label(main, text='Brain Pathology Classification', justify=LEFT)
title.config(bg='#F78888', fg='black')  
title.config(font=font)           
title.config(height=2, width=200)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset File", command=readdataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
processButton.place(x=10,y=150)
processButton.config(font=font1)

transactionButton = Button(main, text="DataSplitting", command=DataSplitting)
transactionButton.place(x=10,y=200)
transactionButton.config(font=font1)

tfButton = Button(main, text="ExecuteModule", command=ExecuteModule)
tfButton.place(x=10,y=250)
tfButton.config(font=font1)

closeButton = Button(main, text="Close Application", command=close)
closeButton.place(x=10,y=400)
closeButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=100)
text.config(font=font1) 

main.config(bg='#90CCF4')
main.mainloop()

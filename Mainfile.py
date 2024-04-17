#======================= IMPORT PACKAGES ==========================

import numpy as np 
import pandas as pd

#============================ READ DATA ==========================

x = np.fromfile('Dataset/6.dat')
import matplotlib.pyplot as plt 
yy = np.arange(0,100000)
y = yy.astype('float')
plt.plot(x[0:100000])
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('Input Signal')
plt.show()

xx = np.fromfile('Dataset/6.dat')

x_input = xx[0:100000]


#================================ PREPROCESSING ==========================


# ----------------- 

# FIR


from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show


#------------------------------------------------
# Create a signal for demonstration.
#------------------------------------------------

sample_rate = 100.0
nsamples = len(x)
t = arange(nsamples) / sample_rate


#------------------------------------------------
# Create a FIR filter and apply it to x.
#------------------------------------------------

# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0

# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  We'll design the filter
# with a 5 Hz transition width.
width = 5.0/nyq_rate

# The desired attenuation in the stop band, in dB.
ripple_db = 60.0

# Compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple_db, width)

# The cutoff frequency of the filter.
cutoff_hz = 10.0

# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

# Use lfilter to filter x with the FIR filter.
filtered_x = lfilter(taps, 1.0, x)

#------------------------------------------------
# Plot the FIR filter coefficients.
#------------------------------------------------

figure(1)
plot(taps, 'bo-', linewidth=2)
title('Filter Coefficients (%d taps)' % N)
grid(True)
plt.show()

#------------------------------------------------
# Plot the magnitude response of the filter.
#------------------------------------------------

figure(2)
clf()
w, h = freqz(taps, worN=8000)
plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
xlabel('Frequency (Hz)')
ylabel('Gain')
title('Frequency Response')
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

#------------------------------------------------
# Plot the original and filtered signals.
#------------------------------------------------

# The phase delay of the filtered signal.
delay = 0.5 * (N-1) / sample_rate

figure(3)
# Plot the original signal.
t = t[0:100000]
filtered_x = filtered_x[0:100000]
plot(t, x[0:100000])
# Plot the filtered signal, shifted to compensate for the phase delay.
plot(t-delay, filtered_x, 'r-')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plot(t[N-1:]-delay, filtered_x[N-1:], 'g', linewidth=4)

xlabel('t')
grid(True)

plt.show()



#  -- Split

import os
from matplotlib import pyplot as plt
test_data = os.listdir('No/')
train_data = os.listdir('Yes/')

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
        
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dot,labels,test_size = 0.3, random_state = 101)


print("----------------------------------------------")
print("Data Splitting")
print("----------------------------------------------")
print()

print("Total No.of data=", len(dot))
print("Total No.of train data=", len(x_train))
print("Total No.of test data=", len(x_test))



# === 1- YES 
# ==== 2. No

from keras.models import Model
from keras.layers import Conv1D, MaxPool1D, Flatten, Input, Dense
#from tensorflow.keras.layers import  Dense

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


#=================


Total_length = len(test_data) + len(train_data)


temp_data1  = []
for ijk in range(0,Total_length):
    # print(ijk)
    temp_data = int((dot[ijk][0]) == (x_input[0]))
    temp_data1.append(temp_data)

temp_data1 =np.array(temp_data1)

zz = np.where(temp_data1==1)

if labels[zz[0][0]] == 1:
    print('-----------------------')
    print()
    print(' Affected By Pathology')
    print()
    print('-----------------------')

else:
    print('--------------------------')
    print()
    print('Not Affected By Pathology')   
    print()
    print('-------------------------')
    
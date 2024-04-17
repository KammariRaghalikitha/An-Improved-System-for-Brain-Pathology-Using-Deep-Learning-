# import pandas as pd

# df = pd.read_table("1.dat", encoding="utf8")



# import numpy as np
# sed = np.loadtxt('1.dat', unpack = True)

# import csv

# # read flash.dat to a list of lists
# datContent = [i.strip().split() for i in open("1.dat").readlines()]


# dat_file = r"1.dat"   
# file = open('1.dat', encoding="utf8")

# with open(file, 'r') as file:
#     text = file.read()
#     print(text)
    
# with open(file) as datFile:
#     print([data.split()[0] for data in datFile])


#============= 


import numpy as np
import pandas as pd

x = np.fromfile('1.dat')



#============

data=pd.read_csv("eeg_data.csv")


from matplotlib import pyplot as plt

t = np.fromfile('1.dat')

# t = np.linspace(-0.02, 0.05, 1000)
plt.plot(t, 325 * np.sin(2*np.pi*50*t));
plt.xlabel('t');
plt.ylabel('x(t)');
plt.title(r'Plot of CT signal \$x(t)=325 \sin(2\pi 50 t)\$');
plt.xlim([-0.02, 0.05]);
#@savefig sineplot.png
plt.show()















import numpy as np
import pandas as pd
import os,csv
csv_path='Estate.csv'
data=pd.read_csv(csv_path)
print(type(data))
data1=np.array(data)
print(type(data1))

data1=data1.transpose(1,0)
a=data1.shape[0]
b=data1.shape[1]

# with open("E_1.csv", "w") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerows(data1)

print(a)
print(b)
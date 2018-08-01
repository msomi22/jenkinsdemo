import numpy,pandas 
import matplotlib.pyplot as plt 
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler 

my_array = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[1,2,3]])  
row_names =['a','b','c','d','e']  
col_names = ['one','two','three'] 
my_data_frame = pandas.DataFrame(my_array, index=row_names, columns=col_names)  

'''
print(my_data_frame.shape)  
print(my_data_frame.describe) 
print(my_data_frame.groupby('three').size()) 
''' 
url = '/home/peter/data/dummy/maps2.csv' 
names = ['map_id','status_a','status_c','status_d','status_o','status_t','status_u','status_w'] 
data = pandas.read_csv(url,names=names) 
#print(data.shape)   
#print(data.describe)   
#print(data.groupby('map_id').size()) 
#print(data.head(20))  
#scatter_matrix(data) 
#plt.show()  
#data.hist()
#plt.show() 
array = data.values 
X = array[:,1:8]
Y = array[:,0] 

scaler = StandardScaler().fit(X) 
rescaledX = scaler.transform(X) 
numpy.set_printoptions(precision=3) 
print(rescaledX[0:5,:])


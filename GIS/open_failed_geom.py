import os

num = {'00021': [16],
 '00026': [26],
 '00027': [37],
 '00028': [1],
 '00037': [16, 20],
 '00046': [16],
 '00048': [12],
 '00049': [1],
 '00050': [10],
 '00080': [17],
 '00086': [34, 37],
 '00088': [20, 38, 43],
 '00095': [5],
 '00101': [24],
 '00102': [36],
 '00103': [5, 40],
 '00104': [12],
 '00108': [2],
 '00110': [3],
 '00114': [12, 34],
 '00118': [32],
 '00128': [2],
 '00130': [43],
 '00131': [10],
 '00134': [14],
 '00139': [3]}
 
path = '/home/felipesa/greening/VECTOR/CORTES'

vector_num = []
for filename in os.listdir(path):
    if os.path.splitext(filename)[0].lower() in num.keys():
        vector_num.append(os.path.join(path, filename))

        iface.addVectorLayer(os.path.join(path, filename), filename, "ogr")

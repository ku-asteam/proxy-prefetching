import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as pltss
import time
from sklearn import preprocessing


#data preprocessing
def One_hot_encoder(path):
    link = []
    for i in path:
        dir_list = os.listdir(i)
        for dir in dir_list:
            try:
                data_path = i+dir
                fp = open(data_path)
                features = fp.readlines()
                for vec in features:
                    vec = vec.split(' ')
                    if vec[3][-4:-1] == "gif" or vec[3][-4:-1] == "GIF":
                        pass
                    else:
                        link.append(vec[3][1:-1])
            except:
                print('errors with data')
    cat_links = list(set(link))
    l_encoder =  preprocessing.LabelEncoder().fit(link)
    link_cat_encoded = l_encoder.fit_transform(link)
    link_cat_encoded = link_cat_encoded.tolist()
    label_encoded = []
    for i in link_cat_encoded:
        label_encoded.append([i])
    encoder = preprocessing.OneHotEncoder(sparse=False).fit(label_encoded)
    return l_encoder, encoder, len(cat_links)




if __name__ == "__main__":
    #path = ['BU_dataset/b19/Apr95/', 'BU_dataset/b19/Feb95/', 'BU_dataset/b19/Jan95/', 'BU_dataset/b19/Mar95/', 'BU_dataset/b19/May95/']
    #label_encoder,one_hot_encoder, cat_num = One_hot_encoder(path)
    #print(one_hot_encoder.transform([label_encoder.transform(['gopher://gopher.bu.edu/11/MetroGuide/Dining%20Out'])]))
    pass

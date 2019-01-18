import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as pltss
import time
from sklearn import preprocessing

def get_files(path):
    data_set = []
    for i in path:
        dir_list = os.listdir(i)
        for dir in dir_list:
            try:
                data_path = i+dir
                fp = open(data_path)
                features = fp.readlines()
                sequence = []
                for vec in features:
                    vec = vec.split(' ')
                    if vec[3][-4:-1] == "gif" or vec[3][-4:-1] == "GIF":
                        pass
                    else:
                        sequence.append(vec[3][1:-1])
                if len(sequence) >3:
                    data_set.append(sequence)
            except:
                print('errors with data')

    #print(len(link_one_hot[0]))
    #字典中的key值即为csv中列名
    # dataframe = pd.DataFrame({'a_name':a,'b_name':b})
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    # dataframe.to_csv("test.csv",index=False,sep=',')
    return data_set, len(data_set)


if __name__ == "__main__":
    path = ['BU_dataset/b19/Apr95/', 'BU_dataset/b19/Feb95/', 'BU_dataset/b19/Jan95/', 'BU_dataset/b19/Mar95/', 'BU_dataset/b19/May95/']
    #path = ['BU_dataset/b19/Apr95/']
    data, length = get_files(path)
    print('data_example: ',data[0])
    print('data_set_length: ',length)

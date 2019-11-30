import pandas as pd
import numpy as np
import os
import time
from sklearn import preprocessing
import collections
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from countminsketch import CountMinSketch
from bloom_filter import BloomFilter
from probables import (CountMinSketch)

#IGNORE_PATTERN = ['gif', 'GIF', 'if?', 'xbm']
IGNORE_PATTERN = []
def link_pure_maker(buff_link):
    filter_buff = buff_link[2].split(':')[0]
    try:
        return filter_buff+'/'+buff_link[3].split('"')[0]
        #return filter_buff+'/'
    except:
        return filter_buff+'/'

#Object_saving
def obj_save(obj, path_name):
    file=open(path_name,'w')
    file.write(str(obj))
    file.close()
    return True
    
def obj_load(path_name):
    file=open(path_name,'r')
    buff = file.read()
    file.close()
    return eval(buff)

def get_withtime_files(path, filter_threshold):
    raw_data_set = collections.defaultdict(list)
    count = 0
    prefetching_list ={}
    ID_stack =[]
    for dir in path:
        dir_list = os.listdir(dir)
        time_stack =[]
        for filename in dir_list:
            data_path = dir+filename
            try:
                fp = open(data_path)
                features = fp.readlines()
                time_slide = []
                last_same = link_pure_maker(features[0].split(' ')[3].split('/'))
                for num in range(len(features)):
                    vec = features[num].split(' ')
                    #delete the same pattern
                    if vec[3][-4:-1] in IGNORE_PATTERN:
                        pass
                    else:
                        buff_link_list = vec[3].split('/')
                        link_cat = link_pure_maker(buff_link_list)
                        if num == 0 or link_cat != last_same:
                            last_same = link_cat
                            tarray = time.localtime(int(vec[1]))
                            # time_str_buff= str(tarray.tm_mon)+str(tarray.tm_mday)+str(tarray.tm_hour)+str(tarray.tm_min)+str(tarray.tm_sec)
                            # time_str_buff= str(tarray.tm_hour)+' '+str(tarray.tm_min)
                            time_str_buff=int((tarray.tm_hour*360+tarray.tm_min*60+tarray.tm_sec)/10)
                            # time_str_buff=int((tarray.tm_hour*60+tarray.tm_min)/8)
                            try:
                                if int(vec[4])>0:
                                    raw_data_set[time_str_buff].append([str(filename.split('.')[0][3:]),link_cat,vec[4]])
                            except:
                                pass
                            ID_stack.append(int(str(filename.split('.')[0][3:])))
                            #[link, user_ID, time_hour]
                            try:
                                if int(vec[4]) > filter_threshold:
                                    if link_cat in prefetching_list:
                                        if vec[4]>prefetching_list[link_cat]:
                                            prefetching_list[link_cat]=vec[4]
                                    else:
                                        prefetching_list[link_cat]=vec[4]
                            except:
                                pass
                        else:
                            pass
            except:
                count+=1

#monitoring
    time_list = []
    for key, cont in raw_data_set.items():
        capacity = 0
        cms = CountMinSketch(width=1000, depth=5)
        bloom = BloomFilter(max_elements=10000, error_rate=0.1)
        for i in raw_data_set[key]:
            bloom.add(i[1])
            cms.add(i[1])
        type_count=0
        amount =0
        for i,l in prefetching_list.items():
            if i in bloom:
                capacity+=int(l)*100
                amount+=1
                type_count+= cms.check(i)
        if type_count==0:
            rep_eta = 0
        else:
            rep_eta = amount/type_count
        # print(str(rep_eta)[:5])
        time_list.append([key,capacity,rep_eta])

    time_list.sort()
    a_list =[]
    for i in range(1,len(time_list)):
        a_list.append([time_list[i][0],time_list[i-1][1]-time_list[i][1],time_list[i][1],time_list[i][2]])
    trigger_list=[]
    for i in a_list:
        if abs(i[1])>250000000:
            trigger_list.append([i[0],i[1]/i[2],i[2],i[3]])
    obj_save(trigger_list, 'trigger_list_dir.txt')
    obj_save(time_list, 'time_list_dir.txt')
    print('激活点位', len(trigger_list))
    print('总时间点位', len(time_list))

    plt.title('   ')
    plt.xlabel('timestamp')
    plt.xticks(rotation=45)
    plt.ylabel('Loads')
    plt.plot([i[0] for i in time_list], [i[1] for i in time_list],'-',color='b',label='Prefetching Loads')
    plt.plot([i[0] for i in a_list], [i[1] for i in a_list],'-',color='r',label='fluctuation')
    plt.legend()
    plt.grid()
    plt.show()
    print(len(raw_data_set))
    user_id = np.array(ID_stack)
    user_scale =np.max(user_id)+1
    print('irregular_format:',count)
    print('Num of users: ',user_scale)
    return raw_data_set, len(raw_data_set), prefetching_list, user_scale

if __name__ == "__main__":
    path_1 = ['BU_dataset/b19/Apr95/', 'BU_dataset/b19/Feb95/', 'BU_dataset/b19/Jan95/', 'BU_dataset/b19/Mar95/', 'BU_dataset/b19/May95/']
    path_2 = ['BU_dataset/272/Apr95/', 'BU_dataset/272/Feb95/', 'BU_dataset/272/Jan95/', 'BU_dataset/272/Mar95/', 'BU_dataset/272/May95/']
    path = path_1+path_2
    get_withtime_files(path, 50000)

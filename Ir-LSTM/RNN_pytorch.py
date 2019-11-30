import numpy as np
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Preprocess as pre
import Encoder
import Url2vec_Encoding as u2v
import random
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
from scipy.interpolate import spline

#----------------------------------
#HYPERPARAMETERS
RATIO = 0.95    #Skipping Ratio
EMBEDDING_DIM = 100     #URL2vec Dim
HIDDEN_DIM = 128    #Num of kernels in 1 hidden layer
HIDDEN_DIM_2 = 256  #Num of kernels in 2 hidden layer
HIDDEN_DIM_3 = 256  #Num of kernels in 3 hidden layer
TIME_EMBEDDING = 2  #Time feature dim
USER_EMBEDDING = 10 #User_ID Featrue dim
MAX_EPOCH = 50 #Max EPOCH_num for a training round
LEARNING_RATE = 0.1  #learning rate
SET_DEVIDE = 0.8    #ratio for training and testing set
FILTERING_SIZE = 50000  #filtering threshold /b
R_UB = 4500000000   #Resource upper bound
EPSILON = 0.85  #Protecting parameters
#data_set scale
PATH = ['BU_dataset/b19/Apr95/', 'BU_dataset/b19/Feb95/', 'BU_dataset/b19/Jan95/', 'BU_dataset/b19/Mar95/', 'BU_dataset/b19/May95/']
PATH += ['BU_dataset/272/Apr95/', 'BU_dataset/272/Feb95/', 'BU_dataset/272/Jan95/', 'BU_dataset/272/Mar95/', 'BU_dataset/272/May95/']
# PATH = ['BU_dataset/b19/Apr95/']
#----------------------------------
#-------------Encoding Mode---------------------
#label encoding with random embedding layer
def Label_encoding_preprocessing(path, set_devide):
    ###init____
    random.seed(1)
    raw_data, ds_size, prefetching_list= pre.get_aimed_files(path, FILTERING_SIZE)
    #label_encoder,cat_num = Encoder.Label_encoder(raw_data)
    label_encoded,cat_num = Encoder.Label_list(path, FILTERING_SIZE)
    prefetching_list.append('null')
    y_label_encoder, y_cat_num = Encoder.Label_encoder([prefetching_list])
    raw_set =[]
    data_set = []
    training_set = []
    testing_set = []
    count = 0
    with_pref_count =0
    without_pre_count = 0
    ###encoding___
    print('start Label_encoding!...')
    recorder =0
    for l in range(len(raw_data)):
        buff = []
        x=[]
        y=[]
        count+=1
        for i in range(len(raw_data[l])):
            x.append(label_encoded[recorder])
            recorder += 1
            if i<len(raw_data[l])-1:
                if raw_data[l][i+1] in prefetching_list:
                    with_pref_count += 1
                    y.append(y_label_encoder.transform([raw_data[l][i+1]])[0])
                else:
                    without_pre_count += 1
                    y.append(y_label_encoder.transform(['null'])[0])
        data_set.append((x[:-1], y))
        sum = Encoder.progress_bar(count, len(raw_data))
    print('Encoding Finished!!\n')
    print('NULL number: ',y_label_encoder.transform(['null'])[0])
    #----
    size = len(data_set)
    random.shuffle(data_set)
    training_set = data_set[:int(size*set_devide)]
    #training_set = data_set
    testing_set = data_set[int(size*set_devide):-1]
    print('BURSTY STAMP:', with_pref_count)
    print('NORMAL CONNECT:', without_pre_count)
    print('Length of training_set:',len(training_set))
    print('Length of testing_set:',len(testing_set))
    print('Number of categories:',cat_num)
    print('Number of Y categories:',y_cat_num)
    return training_set,testing_set, cat_num, y_cat_num, y_label_encoder.transform(['null'])[0]
#with time and user_ID features
def Time_User_encoding_preprocessing(path, set_devide):
    ###init____
    raw_data, ds_size, prefetching_list, user_cat= pre.get_withtime_files(path, FILTERING_SIZE)
    #label_encoder,cat_num = Encoder.Label_encoder(raw_data)
    label_encoded,cat_num = Encoder.Label_list(path, FILTERING_SIZE)
    prefetching_list.append('null')
    y_label_encoder, y_cat_num = Encoder.Label_encoder([prefetching_list])
    raw_set =[]
    data_set = []
    training_set = []
    testing_set = []
    count = 0
    with_pref_count =0
    without_pre_count = 0
    ###encoding___
    print('start Label_encoding!...')
    recorder =0
    for l in range(len(raw_data)):
        buff = []
        x=[]
        x_time = []
        x_id = []
        y=[]
        count+=1
        for i in range(len(raw_data[l])):
            x.append(label_encoded[recorder])
            x_time.append([int(raw_data[l][i][2])])
            x_id.append([int(raw_data[l][i][1])])
            recorder += 1
            if i<len(raw_data[l])-1:
                if raw_data[l][i+1][0] in prefetching_list:
                    with_pref_count += 1
                    y.append(y_label_encoder.transform([raw_data[l][i+1][0]])[0])
                else:
                    without_pre_count += 1
                    y.append(y_label_encoder.transform(['null'])[0])
        data_set.append(([x[:-1],x_id[:-1],x_time[:-1]], y))
        sum = Encoder.progress_bar(count, len(raw_data))
    print('Encoding Finished!!\n')
    print('NULL number: ',y_label_encoder.transform(['null'])[0])
    #----
    size = len(data_set)
    random.shuffle(data_set)
    training_set = data_set[:int(size*set_devide)]
    #training_set = data_set
    testing_set = data_set[int(size*set_devide):-1]
    print('BURSTY STAMP:', with_pref_count)
    print('NORMAL CONNECT:', without_pre_count)
    print('Length of training_set:',len(training_set))
    print('Length of testing_set:',len(testing_set))
    print('Number of categories:',cat_num)
    print('Number of Y categories:',y_cat_num)
    return training_set,testing_set, cat_num, y_cat_num, y_label_encoder.transform(['null'])[0],user_cat
#with URL2vec encoding
def Url2vec_encoding(path, set_devide):
    raw_data, ds_size, prefetching_list, user_cat= pre.get_withtime_files(path, FILTERING_SIZE)
    url2vec_encoder = u2v.Url2vec_Model()
    url2vec_transformer = url2vec_encoder.reload('./100_15_MyModel')
    label_encoded,cat_num = Encoder.Label_list(path, FILTERING_SIZE)
    prefetching_list.append('null')
    y_label_encoder, y_cat_num = Encoder.Label_encoder([prefetching_list])
    raw_set =[]
    data_set = []
    training_set = []
    testing_set = []
    count = 0
    with_pref_count =0
    without_pre_count = 0
    ###encoding___
    for l in range(len(raw_data)):
        buff = []
        x=[]
        x_time = []
        x_id = []
        y=[]
        count+=1
        for i in range(len(raw_data[l])):
            x.append(list(url2vec_transformer[raw_data[l][i][0]]))
            x_time.append([int(raw_data[l][i][2])])
            x_id.append([int(raw_data[l][i][1])])
            if i<len(raw_data[l])-1:
                if raw_data[l][i+1][0] in prefetching_list:
                    with_pref_count += 1
                    y.append(y_label_encoder.transform([raw_data[l][i+1][0]])[0])
                else:
                    without_pre_count += 1
                    y.append(y_label_encoder.transform(['null'])[0])
        data_set.append(([x[:-1],x_id[:-1],x_time[:-1]], y))
        sum = Encoder.progress_bar(count, len(raw_data))
    print('Encoding Finished!!\n')
    print('NULL number: ',y_label_encoder.transform(['null'])[0])
    #----
    size = len(data_set)
    random.shuffle(data_set)
    training_set = data_set[:int(size*set_devide)]
    #training_set = data_set
    testing_set = data_set[int(size*set_devide):-1]
    print('BURSTY STAMP:', with_pref_count)
    print('NORMAL CONNECT:', without_pre_count)
    print('Length of training_set:',len(training_set))
    print('Length of testing_set:',len(testing_set))
    print('Number of categories:',cat_num)
    print('Number of Y categories:',y_cat_num)
    return training_set,testing_set, cat_num, y_cat_num, y_label_encoder.transform(['null'])[0],user_cat
#with URL2vec encoding for streaming simulation.
def Url2vec_encoding_stream(path, set_devide):
    raw_data, ds_size, prefetching_list, user_cat= pre.get_withtime_files_stream(path, FILTERING_SIZE)
    url2vec_encoder = u2v.Url2vec_Model()
    url2vec_transformer = url2vec_encoder.reload('./100_15_MyModel')
    label_encoded,cat_num = Encoder.Label_list(path, FILTERING_SIZE)
    prefetching_list.append('null')
    y_label_encoder, y_cat_num = Encoder.Label_encoder([prefetching_list])
    raw_set =[]
    data_set = []
    training_set = []
    testing_set = []
    count = 0
    with_pref_count =0
    without_pre_count = 0
    ###encoding___
    for l in range(len(raw_data)):
        buff = []
        x=[]
        x_time = []
        x_id = []
        time_unit_stamp = []
        y=[]
        count+=1
        for i in range(len(raw_data[l])):
            x.append(list(url2vec_transformer[raw_data[l][i][0]]))
            x_time.append([int(raw_data[l][i][2])])
            x_id.append([int(raw_data[l][i][1])])
            time_unit_stamp.append([int(raw_data[l][i][3])])
            if i<len(raw_data[l])-1:
                if raw_data[l][i+1][0] in prefetching_list:
                    with_pref_count += 1
                    y.append(y_label_encoder.transform([raw_data[l][i+1][0]])[0])
                else:
                    without_pre_count += 1
                    y.append(y_label_encoder.transform(['null'])[0])
        data_set.append((time_unit_stamp[0][0],[x[:-1],x_id[:-1],x_time[:-1]], y))
        sum = Encoder.progress_bar(count, len(raw_data))
    print('Encoding Finished!!\n')
    print('NULL number: ',y_label_encoder.transform(['null'])[0])
    #----
    size = len(data_set)
    random.shuffle(data_set)
    training_set = data_set[:int(size*set_devide)]
    #training_set = data_set
    testing_set = data_set[int(size*set_devide):-1]
    print('BURSTY STAMP:', with_pref_count)
    print('NORMAL CONNECT:', without_pre_count)
    print('Length of training_set:',len(training_set))
    print('Length of testing_set:',len(testing_set))
    print('Number of categories:',cat_num)
    print('Number of Y categories:',y_cat_num)
    return training_set,testing_set, cat_num, y_cat_num, y_label_encoder.transform(['null'])[0],user_cat
#----------------------------------
torch.manual_seed(1) #Fixed random seed
random.seed(1)
#----------------------------------
#-------------MODELS---------------------
#LSTM Model
class regular_LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(regular_LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first= True, dropout=0.8)
        #（num_layers * num_directions, batch_size,  hidden_size）
        self.h0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.c0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.hiddentag = nn.Linear(hidden_dim, tagset_size)
    def hidden_init(self):
        self.h0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.c0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        #print('embeds:', embeds.size())
        lstm_out, (self.h_out, self.c_cout) = self.lstm(embeds.view(1,len(sentence), -1),  (self.h0, self.c0))
        #print('layer1:', lstm_out.size())
        tag_space = self.hiddentag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores
#LSTM with uRL2vec
class regular_with_url2vec_LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(regular_with_url2vec_LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first= True, dropout=0.8)
        #（num_layers * num_directions, batch_size,  hidden_size）
        self.h0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.c0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.hiddentag = nn.Linear(hidden_dim, tagset_size)
    def hidden_init(self):
        self.h0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.c0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
    def forward(self, sentence):
        embeds = sentence
        #print('embeds:', embeds.size())
        lstm_out, (self.h_out, self.c_cout) = self.lstm(embeds.view(1,len(sentence), -1),  (self.h0, self.c0))
        #print('layer1:', lstm_out.size())
        tag_space = self.hiddentag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores
#multi_LSTM Model 2_layer_lstm
class multi_regular_LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim,hidden_dim_2, vocab_size, tagset_size):
        super(multi_regular_LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim_2 = hidden_dim_2
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_l0 = nn.LSTM(embedding_dim, hidden_dim, batch_first= True, dropout=0.5)
        #（num_layers * num_directions, batch_size,  hidden_size）
        self.h0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.c0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.lstm_l1 = nn.LSTM(hidden_dim, hidden_dim_2, batch_first= True, dropout=0.5)
        #（num_layers * num_directions, batch_size,  hidden_size）
        self.h1 = autograd.Variable(torch.randn(1, 1, self.hidden_dim_2))
        self.c1 = autograd.Variable(torch.randn(1, 1, self.hidden_dim_2))
        self.hiddentag = nn.Linear(hidden_dim_2, tagset_size)
    def hidden_init(self):
        self.h0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.c0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.h1 = autograd.Variable(torch.randn(1, 1, self.hidden_dim_2))
        self.c1 = autograd.Variable(torch.randn(1, 1, self.hidden_dim_2))
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        #print('embeds:', embeds.size())
        lstm_l1_out, (self.h_out, self.c_cout) = self.lstm_l0(embeds.view(1,len(sentence), -1),  (self.h0, self.c0))
        lstm_l2_out, (self.h_out_2, self.c_cout_2) = self.lstm_l1(lstm_l1_out.view(1,len(sentence), -1),  (self.h1, self.c1))
        #print('layer1:', lstm_out.size())
        tag_space = self.hiddentag(lstm_l2_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores
#LSTM Model WITH TIME&&USERID features
class with_added_feature_LSTMTagger(nn.Module):
    def __init__(self, time_embedding_dim, user_embedding_dim, embedding_dim, hidden_dim, hidden_dim_2, hidden_dim_3,vocab_size, tagset_size, user_cat, time_cat):
        super(with_added_feature_LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.time_embeddings = nn.Embedding(time_cat, time_embedding_dim)
        self.id_embeddings = nn.Embedding(user_cat, user_embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first= True, dropout=0.8)
        #（num_layers * num_directions, batch_size,  hidden_size）
        self.h0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.c0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.hiddentag = nn.Linear(hidden_dim+time_embedding_dim +user_embedding_dim, hidden_dim_2)
        self.hiddentag_2 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.hiddentag_3 = nn.Linear(hidden_dim_3, tagset_size)
    def hidden_init(self):
        self.h0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.c0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
    def forward(self, sentence):
        url_embeds = self.word_embeddings(sentence[0])
        time_embeds = self.time_embeddings(sentence[2]).view(len(sentence[0]), -1)
        id_embeds = self.id_embeddings(sentence[1]).view(len(sentence[0]), -1)
        #print('url_embeds:', url_embeds.size())
        #print('time_embeds:', time_embeds.size())
        #print('id_embeds:', id_embeds.size())
        lstm_out, (self.h_out, self.c_cout) = self.lstm(url_embeds.view(1,len(sentence[0]), -1),(self.h0, self.c0))
        #print('layer1:', lstm_out.size())
        #tag_space = self.hiddentag(lstm_out.view(len(sentence[0]), -1))
        #Merging layer
        cat_buff = torch.cat((lstm_out.view(len(sentence[0]),-1),time_embeds),1)
        #print(cat_buff.size())
        cat_out = torch.cat((cat_buff,id_embeds),1)
        layer_1 = self.hiddentag(cat_out.view(len(sentence[0]), -1))
        layer_1_relu = F.relu(layer_1)
        layer_2 = self.hiddentag_2(layer_1_relu.view(len(sentence[0]), -1))
        layer_2_relu = F.relu(layer_2)
        final_out= self.hiddentag_3(layer_2_relu.view(len(sentence[0]), -1))
        tag_scores = F.log_softmax(final_out)
        return tag_scores
#LSTM Model WITH TIME&&Url2vec_Model
class with_added_feature_url2vec_LSTMTagger(nn.Module):
    def __init__(self, time_embedding_dim, user_embedding_dim, embedding_dim, hidden_dim, hidden_dim_2, hidden_dim_3,vocab_size, tagset_size, user_cat, time_cat):
        super(with_added_feature_url2vec_LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_embeddings = nn.Embedding(time_cat, time_embedding_dim)
        self.id_embeddings = nn.Embedding(user_cat, user_embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first= True, dropout=0.8)
        #（num_layers * num_directions, batch_size,  hidden_size）
        self.h0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.c0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.hiddentag = nn.Linear(hidden_dim+time_embedding_dim +user_embedding_dim, hidden_dim_2)
        self.hiddentag_2 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.hiddentag_3 = nn.Linear(hidden_dim_3, tagset_size)
    def hidden_init(self):
        self.h0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
        self.c0 = autograd.Variable(torch.randn(1, 1, self.hidden_dim))
    def forward(self, sentence):
        url_embeds = sentence[0]
        time_embeds = self.time_embeddings(sentence[2]).view(len(sentence[0]), -1)
        id_embeds = self.id_embeddings(sentence[1]).view(len(sentence[0]), -1)
        #print('url_embeds:', url_embeds.size())
        #print('time_embeds:', time_embeds.size())
        #print('id_embeds:', id_embeds.size())
        lstm_out, (self.h_out, self.c_cout) = self.lstm(url_embeds.view(1,len(sentence[0]), -1),(self.h0, self.c0))
        #print('layer1:', lstm_out.size())
        #tag_space = self.hiddentag(lstm_out.view(len(sentence[0]), -1))
        #Merging layer
        cat_buff = torch.cat((lstm_out.view(len(sentence[0]),-1),time_embeds),1)
        #print(cat_buff.size())
        cat_out = torch.cat((cat_buff,id_embeds),1)
        layer_1 = self.hiddentag(cat_out.view(len(sentence[0]), -1))
        layer_1_relu = F.tanh(layer_1)
        layer_2 = self.hiddentag_2(layer_1_relu.view(len(sentence[0]), -1))
        layer_2_relu = F.tanh(layer_2)
        final_out= self.hiddentag_3(layer_2_relu.view(len(sentence[0]), -1))
        tag_scores = F.log_softmax(final_out)
        return tag_scores
#----------------------------------
#-------------TOOLs---------------------
#type_transform
def turn_FloatTensor(x):
    tensor = torch.Tensor(x)
    return autograd.Variable(tensor)
def turn_LongTensor(x):
    tensor = torch.LongTensor(x)
    return autograd.Variable(tensor)
#evaluation function
#Output result in string type
# labels: label list, pred_score: prediction results, ratio: skipping ratio, null_num: the skip pattern, block_num: num of the output target
def get_score(labels, pred_score, ratio, null_num, block_num):
    full_amount = 0
    label_noneed_amount = 0
    label_prefetch_amount =0
    #-------------------------
    pred_right_noneed_amount = 0
    pred_wrong_noneed_amount = 0
    pred_right_predict_amount = 0
    pred_wong_predict_amount = 0
    hit_amount =0
    #this is only consisted in the prefeted parts
    mis_hit = 0
    #whole
    for i in range(len(labels)):
            #sentence
            for l in range(len(labels[i])):
                prefetching_judge = math.exp(pred_score[i][l][null_num])
                full_amount += 1
                #No need condition
                if labels[i][l] == null_num:
                    label_noneed_amount += 1
                    if prefetching_judge >= ratio:
                        pred_right_noneed_amount += 1
                    else:
                        pred_wrong_noneed_amount += 1
                else:
                    label_prefetch_amount += 1
                    if prefetching_judge < ratio:
                        pred_right_predict_amount += 1
                        if labels[i][l] in [x for x in pred_score[i][l].data.numpy().argsort()[(-1*block_num):]]:
                            hit_amount += 1
                        else:
                            mis_hit += 1
                    else:
                        pred_wong_predict_amount += 1
    # for i in range(len(labels)):
    #         #sentence
    #         for l in range(len(labels[i])):
    #             prefetching_judge = pred_score[i][l].data.numpy().argsort()[(-1*block_num):]
    #             full_amount += 1
    #             #No need condition
    #             if labels[i][l] == null_num:
    #                 label_noneed_amount += 1
    #                 if prefetching_judge == null_num:
    #                     pred_right_noneed_amount += 1
    #                 else:
    #                     pred_wrong_noneed_amount += 1
    #             else:
    #                 label_prefetch_amount += 1
    #                 if prefetching_judge != null_num:
    #                     pred_right_predict_amount += 1
    #                     if labels[i][l] in [x for x in pred_score[i][l].data.numpy().argsort()[(-1*block_num):]]:
    #                         hit_amount += 1
    #                     else:
    #                         mis_hit += 1
    #                 else:
    #                     pred_wong_predict_amount += 1
    # no_need_ratio = str((label_noneed_amount/full_amount)*100)[:4]+'%'
    # right_noneed = str((pred_right_noneed_amount/label_noneed_amount)*100)[:4]+'%'
    # wasting_num = str(pred_wrong_noneed_amount+mis_hit)
    # wasting_ratio = str(((pred_wrong_noneed_amount+mis_hit)/(pred_wrong_noneed_amount+pred_right_predict_amount))*100)[:4]+'%'
    try:
        hit_ratio = str((hit_amount/label_prefetch_amount)*100)[:4]+'%'
        accurracy = str(((pred_right_noneed_amount+hit_amount)/full_amount)*100)[:4]+'%'
    except:
        hit_ratio ='null'
        accurracy ='null'
    # buff_result = no_need_ratio +' '+ right_noneed +' '+ wasting_num +' '+ wasting_ratio +' '+ hit_ratio +' '+ accurracy + '\n'
    buff_result = 'null' +' '+ 'null' +' '+'null' +' '+ 'null' +' '+ hit_ratio +' '+ accurracy
    # print('正确拦截比:', str((pred_right_noneed_amount/label_noneed_amount)*100)[:4]+'%')
    # print('未做Prefetching比:', str((pred_wong_predict_amount/label_prefetch_amount)*100)[:4]+'%')
    # print('浪费比例:', str(((pred_wrong_noneed_amount+mis_hit)/(pred_wrong_noneed_amount+pred_right_predict_amount))*100)[:4]+'%')
    print('HIT_RATIO:', hit_ratio)
    print('ACCURRACY:', accurracy)
    return buff_result
#Output result in list type
def get_value_score(labels, pred_score, ratio, null_num, block_num):
    full_amount = 0
    label_noneed_amount = 0
    label_prefetch_amount =0
    #-------------------------
    pred_right_noneed_amount = 0
    pred_wrong_noneed_amount = 0
    pred_right_predict_amount = 0
    pred_wong_predict_amount = 0
    hit_amount =0
    #this is only consisted in the prefeted parts
    mis_hit = 0
    #whole
    for i in range(len(labels)):
            #sentence
            for l in range(len(labels[i])):
                prefetching_judge = math.exp(pred_score[i][l][null_num])
                full_amount += 1
                #No need condition
                if labels[i][l] == null_num:
                    label_noneed_amount += 1
                    if prefetching_judge >= ratio:
                        pred_right_noneed_amount += 1
                    else:
                        pred_wrong_noneed_amount += 1
                else:
                    label_prefetch_amount += 1
                    if prefetching_judge < ratio:
                        pred_right_predict_amount += 1
                        if labels[i][l] in [x for x in pred_score[i][l].data.numpy().argsort()[(-1*block_num):]]:
                            hit_amount += 1
                        else:
                            mis_hit += 1
                    else:
                        pred_wong_predict_amount += 1
    no_need_ratio = label_noneed_amount/full_amount
    Fs = (pred_right_predict_amount+ pred_wrong_noneed_amount)/(full_amount-label_noneed_amount)
    right_noneed = pred_right_noneed_amount/label_noneed_amount
    wasting_num = pred_wrong_noneed_amount+mis_hit
    wasting_ratio = (pred_wrong_noneed_amount+mis_hit)/(pred_wrong_noneed_amount+pred_right_predict_amount)
    hit_ratio = hit_amount/label_prefetch_amount
    accurracy = (pred_right_noneed_amount+hit_amount)/full_amount
    buff_result = [no_need_ratio,right_noneed,wasting_num,wasting_ratio,hit_ratio,accurracy,Fs]
    return buff_result
#Frist Hit Output result in string type
def T_score(labels, pred_score, null_num):
    full_amount = 0
    label_noneed_amount = 0
    label_prefetch_amount =0
    #-------------------------
    pred_right_noneed_amount = 0
    pred_wrong_noneed_amount = 0
    pred_right_predict_amount = 0
    pred_wong_predict_amount = 0
    hit_amount =0
    #this is only consisted in the prefeted parts
    mis_hit = 0
    #whole
    for i in range(len(labels)):
            #sentence
            for l in range(len(labels[i])):
                prefetching_judge = pred_score[i][l].data.numpy().argsort()[-1]
                full_amount += 1
                #No need condition
                if labels[i][l] == null_num:
                    label_noneed_amount += 1
                    if prefetching_judge == null_num:
                        pred_right_noneed_amount += 1
                    else:
                        pred_wrong_noneed_amount += 1
                else:
                    label_prefetch_amount += 1
                    if prefetching_judge != null_num:
                        pred_right_predict_amount += 1
                        if labels[i][l] == pred_score[i][l].data.numpy().argsort()[-1]:
                            hit_amount += 1
                        else:
                            mis_hit += 1
                    else:
                        pred_wong_predict_amount += 1
    no_need_ratio = str((label_noneed_amount/full_amount)*100)[:4]+'%'
    right_noneed = str((pred_right_noneed_amount/label_noneed_amount)*100)[:4]+'%'
    wasting_num = str(pred_wrong_noneed_amount+mis_hit)
    wasting_ratio = str(((pred_wrong_noneed_amount+mis_hit)/(pred_wrong_noneed_amount+pred_right_predict_amount))*100)[:4]+'%'
    hit_ratio = str((hit_amount/label_prefetch_amount)*100)[:4]+'%'
    accurracy = str(((pred_right_noneed_amount+hit_amount)/full_amount)*100)[:4]+'%'
    hit_per = hit_amount/label_prefetch_amount
    buff_result = no_need_ratio +' '+ right_noneed +' '+ wasting_num +' '+ wasting_ratio +' '+ hit_ratio +' '+ accurracy
    print('Correctly Skipping Rate:', right_noneed)
    print('Wrong Prefetching rate:', str((pred_wong_predict_amount/label_prefetch_amount)*100)[:4]+'%')
    print('Wasting Rate:', str(((pred_wrong_noneed_amount+mis_hit)/(pred_wrong_noneed_amount+pred_right_predict_amount))*100)[:4]+'%')
    print('Hit Ratio:', hit_ratio)
    print('Accurracy:', accurracy)
    return buff_result, hit_per
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
#3D Visualization
def visual():
    dict = obj_load('./3D_dict_withhumanfeature_19.txt')
    thre_dict = obj_load('./3D_dict_withhumanfeature_19_withFS.txt')

    Y_label = [str(0.05*x)[:4] for x in range(1,20)]
    X_label = range(1,16)
    X = []
    Y = []
    included_vector  = collections.defaultdict(list)
    Z = []
    buff =[]
    r_ub = []
    for n in [0.05*x for x in range(1,20)]:
        buff =[]
        for n_block in range(1,16):
            X.append(n)
            Y.append(n_block)
            fs = thre_dict[str(n)[:4]+'_'+str(1)][6]
            loading = 3*0.1*fs*n_block
            # value  = dict[str(n)[:4]+'_'+str(n_block)[:4]][5]
            r_ub.append(5.5)
            buff.append(loading)
            # included_vector[value].append([dict[str(n)[:4]+'_'+str(n_block)[:4]],n,n_block])
        Z.append(buff)
    # statist = np.array(Z)
    # print(np.max(statist))
    # print(included_vector[np.max(statist)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(Y_label)))
    ax.set_yticklabels(Y_label)
    ax.set_xticks(range(len(X_label)))
    ax.set_xticklabels(X_label)
    # Z = [i for i in Z]
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(X, Y, Z, cmap='rainbow')
    # plt.xlabel('Threshod')
    # plt.ylabel('num_block')
    im = ax.imshow(Z, cmap=plt.cm.hot_r)
    #增加右侧的颜色刻度条
    # title_buff = 'Accuracy'
    # title_buff = 'Hitting Ratio'
    title_buff = 'Loading Capacity'
    plt.title(title_buff)
    plt.colorbar(im)
    plt.show()
#----------------------------------
#PROXY Server Loading STREAMING SIMULATION
def simulation():
    dict = obj_load('./3D_dict_withhumanfeature_19.txt')
    thre_dict = obj_load('./3D_dict_withhumanfeature_19_withFS.txt')
    bursty_list = obj_load('./trigger_list_dir.txt')
    time_list = obj_load('./time_list_dir.txt')
    with_dynamic_change=[]
    without_dynamic_change = []
    bursty_withvar_list = {}
    for i in bursty_list:
        #i[0]:time, i[1]:capacity,i[2]:rep_rate
        max = 0
        best_fitting =[]
        dy_loads=[]
        for n in [0.05*x for x in range(1,20)]:
            buff =[]
            for n_block in range(1,16):
                fs = thre_dict[str(int(n*100))+'_'+str(1)][6]
                loads = i[2]*i[3]*fs*n_block
                if loads<R_UB:
                    hit_ratio  = dict[str(n)[:4]+'_'+str(n_block)[:4]][4]
                    if hit_ratio>max:
                        max = hit_ratio
                        best_fitting = [n, n_block]
        bursty_withvar_list[i[0]]=best_fitting
    obj_save(bursty_withvar_list, 'bursty_withvar_list.txt')
    # print(bursty_withvar_list)
    loads=0
    n_block=1
    theta=50
    fs =0.5
    FIXED_n_block =8
    FIXED_THETA = 70
    utilize_rate = []
    dy_utilize_rate = []
    overload_count =0
    dy_overload_count =0
    for i in time_list:
        loads_fixed = i[1]*i[2]*thre_dict[str(FIXED_THETA)[:4]+'_'+str(1)][6]*FIXED_n_block
        if loads_fixed/R_UB>1:
            utilize_rate.append(1)
            overload_count+=1
        else:
            utilize_rate.append(loads_fixed/R_UB)
        without_dynamic_change.append(loads_fixed)
        if i[0] in bursty_withvar_list:
            n_block = bursty_withvar_list[i[0]][1]
            theta = int(bursty_withvar_list[i[0]][0]*85/5)*5
            fs = thre_dict[str(theta)[:4]+'_'+str(1)][6]
            loads = i[1]*i[2]*fs*n_block
            with_dynamic_change.append(loads)
            if loads/R_UB>1:
                dy_utilize_rate.append(1)
                dy_overload_count+=1
            else:
                dy_utilize_rate.append(loads/R_UB)
        else:
            loads = i[1]*i[2]*fs*n_block
            with_dynamic_change.append(loads)
            if loads/R_UB>1:
                dy_utilize_rate.append(1)
                dy_overload_count+=1
            else:
                dy_utilize_rate.append(loads/R_UB)
    # print(with_dynamic_change)
    # print(without_dynamic_change)
    plt.title('   ')
    plt.xlabel('timestamp')
    plt.xticks(rotation=45)
    plt.ylabel('Sever Loads')
    # plt.plot([i[0] for i in time_list], with_dynamic_change,'-',color='b',label='Dynamic Allocation')
    # plt.plot([i[0] for i in time_list], without_dynamic_change,'-',color='r',label='fixed Allocation')
    X = np.linspace(np.array([i[0] for i in time_list]).min(),np.array([i[0] for i in time_list]).max(),300)
    DY_smooth = spline([i[0] for i in time_list],with_dynamic_change,X)
    ST_smooth = spline([i[0] for i in time_list],without_dynamic_change,X)
    print('Fixed Mode Resource_Utilization',np.array(utilize_rate).mean())
    print('Dynamic Mode Resource_Utilization',np.array(dy_utilize_rate).mean())
    print('Fixed Mode Overload_Amplitude',(ST_smooth.max()-R_UB)/R_UB)
    print('Dynamic Mode Overload_Amplitude',(DY_smooth.max()-R_UB)/R_UB)
    print('Fixed Mode Overload_Frequency',overload_count/len(time_list))
    print('Dynamic Mode Overload_Frequency',dy_overload_count/len(time_list))
    plt.plot(X,DY_smooth,'-',color='b',label='Dynamic Allocation')
    plt.plot(X,ST_smooth,'-',color='r',label='fixed Allocation')
    plt.legend()
    plt.grid()
    plt.show()
#----------------------------------
#Training
def training_model(model_name, last_epoch_num):
    #----------------------
    # A set
    #Without time_user features
    # training_set,testing_set, cat_num, y_cat_num, null_num= Label_encoding_preprocessing(PATH, SET_DEVIDE)
    # #model = multi_regular_LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,HIDDEN_DIM_2,cat_num, y_cat_num)
    # model = regular_LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,cat_num, y_cat_num)
    #----------------------
    # B set
    #With time_user features
    # training_set,testing_set, cat_num, y_cat_num, null_num,user_cat= Time_User_encoding_preprocessing(PATH, SET_DEVIDE)
    # model = with_added_feature_LSTMTagger(TIME_EMBEDDING, USER_EMBEDDING, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM_2,HIDDEN_DIM_3,cat_num, y_cat_num, 820, 24)
    #----------------------
    # C set
    #Based on trained Skip-gram
    training_set,testing_set, cat_num, y_cat_num, null_num,user_cat= Url2vec_encoding(PATH, SET_DEVIDE)
    # model =  with_added_feature_url2vec_LSTMTagger(TIME_EMBEDDING, USER_EMBEDDING, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM_2,HIDDEN_DIM_3,cat_num, y_cat_num, 820, 24)
    model = regular_with_url2vec_LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,cat_num, y_cat_num)
    #----------------------
    #MODEL_LOADING
    try:
        model.load_state_dict(torch.load(model_name+'.pkl'))
    except:
        pass
    #----------------------
    #GD optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum = 0.9, weight_decay=1e-5)
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # epoch, skip/predict ratio, correct skipping rate, wasting operation num, wasting rate, hit ratio, accuracy, training time per epoch, loss value
    train_recording_buff ='epoch'+' '+'skip/predict ratio'+' '+'correct skipping rate'+' '+'wasting operation num'+' '+'wasting rate'+' '+'hit ratio'+' '+'accuracy'+' '+'training time per epoch'+' '+'mean_loss'+'\n'
    test_recording_buff ='epoch'+' '+'skip/predict ratio'+' '+'correct skipping rate'+' '+'wasting operation num'+' '+'wasting rate'+' '+'hit ratio'+' '+'accuracy'+' '+'training time per epoch'+' '+'mean_loss'+'\n'
    present_epoch = last_epoch_num
    #Loss funciton
    loss_fn = torch.nn.CrossEntropyLoss()
    #loss_fn = torch.nn.MultiLabelSoftMarginLoss()
    best_epoch = 0
    best_accuracy = 0
    for epoch in range(MAX_EPOCH):
        timestamp = time.time()
        portion = 0
        total_loss = 0
        amount = accurrate_count = ture_positive_amount = ture_positive_accurrate_count= 0
        model.hidden_init()
        #-------------------------------------
        #Training set monitor
        preditions_list=[]
        tag_list = []
        for sentence, tags in training_set:
            model.zero_grad()
            #X
            # sentence_in = turn_LongTensor(sentence)
            # sentence_in = [turn_LongTensor(sentence[0]),turn_LongTensor(sentence[1]),turn_LongTensor(sentence[2])]
            # sentence_in = [turn_FloatTensor(sentence[0]),turn_LongTensor(sentence[1]),turn_LongTensor(sentence[2])]
            sentence_in = turn_FloatTensor(sentence[0])
            target_buff = []
            for i in tags:
                amount+=1
                target_buff.append(i)
            targets = turn_LongTensor(target_buff)
            tag_scores = model(sentence_in)
            for i in range(len(tags)):
                a = tag_scores[i].data.numpy()
            preditions_list.append(tag_scores)
            tag_list.append(target_buff)
            #Backpropagation
            loss = loss_fn(tag_scores, targets)
            total_loss+= loss.item()
            loss.backward()
            #Clipping for gradient explode
            nn.utils.clip_grad_norm(filter(lambda p:p.requires_grad,model.parameters()),max_norm=0.5)
            optimizer.step()
            # 调用方式
            total = len(training_set)
            portion+=1
            sum = Encoder.progress_bar(portion, total)
        print('----------------------------------------------------------')
        print('Training集:')
        period = time.time() - timestamp
        timestamp = time.time()
        mean_loss = total_loss/len(training_set)
        present_epoch += 1
        print('Epoch:', present_epoch)
        print('CrossEntropyLoss:',mean_loss)
        print('Time consuming:',period)
        records, nouse = T_score(tag_list, preditions_list,null_num)
        train_recording_buff += str(present_epoch)+' '+records+' '+str(period)+' '+str(mean_loss)+'\n'
        #-------------------------------------
        #Testing set monitor
        preditions_list=[]
        tag_list = []
        test_loss = 0
        for sentence,tags in testing_set:
            # sentence_in = turn_LongTensor(sentence)
            # sentence_in = [turn_LongTensor(sentence[0]),turn_LongTensor(sentence[1]),turn_LongTensor(sentence[2])]
            # sentence_in = [turn_FloatTensor(sentence[0]),turn_LongTensor(sentence[1]),turn_LongTensor(sentence[2])]
            sentence_in = turn_FloatTensor(sentence[0])
            target_buff = []
            for i in tags:
                target_buff.append(i)
            targets = turn_LongTensor(target_buff)
            tag_scores = model(sentence_in)
            loss = loss_fn(tag_scores, targets)
            test_loss += loss.item()
            for i in range(len(tags)):
                a = tag_scores[i].data.numpy()
            preditions_list.append(tag_scores)
            tag_list.append(target_buff)
        print('----------------------------------------------------------')
        print('Testing:')
        records, accuracy = T_score(tag_list, preditions_list,null_num)
        mean_test_loss = test_loss/len(testing_set)
        print('CrossEntropyLoss:', mean_test_loss)
        test_recording_buff += str(present_epoch)+' '+records+' '+str(mean_test_loss)+'\n'
        print('----------------------------------------------------------')
        #模型存储
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), model_name+'_best.pkl')
            best_accuracy = accuracy
        if epoch!=0 and (epoch+1)%10 ==0:
            torch.save(model.state_dict(), model_name+'__'+str(epoch)+'.pkl')
        #----------------------
    #----------------------
    #保存结果文件&& 模型
    print('training finished! \n loss: ', mean_loss)
    #时间记录保存
    fp = open(model_name+'_train_recodding.txt','a')
    fp.write(train_recording_buff )
    fp.close()
    fp = open(model_name+'_test_recodding.txt','a')
    fp.write(test_recording_buff)
    fp.close()
#----------------------------------
#Testing
def testing_model(model_name):
    # #----------------------
    # training_set,testing_set, cat_num, y_cat_num, null_num= Label_encoding_preprocessing(PATH, SET_DEVIDE)
    # #model = multi_regular_LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,HIDDEN_DIM2,cat_num, cat_num)
    # model = regular_LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,cat_num, y_cat_num)
    # #----------------------
    # B set
    #With time_user features
    # training_set,testing_set, cat_num, y_cat_num, null_num,user_cat= Time_User_encoding_preprocessing(PATH, SET_DEVIDE)
    # model = with_added_feature_LSTMTagger(TIME_EMBEDDING, USER_EMBEDDING, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM_2,HIDDEN_DIM_3,cat_num, y_cat_num, 820, 24)
    #----------------------
    # C set
    #Based on trained word embedding_dim
    training_set,testing_set, cat_num, y_cat_num, null_num,user_cat= Url2vec_encoding(PATH, SET_DEVIDE)
    model =  with_added_feature_url2vec_LSTMTagger(TIME_EMBEDDING, USER_EMBEDDING, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM_2,HIDDEN_DIM_3,cat_num, y_cat_num, 820, 24)
    #----------------------
    model.load_state_dict(torch.load(model_name+'.pkl'))
    result = 'Threshold/num of targets'+' '+'skip/predict ratio'+' '+'correct skipping rate'+' '+'wasting operation num'+' '+'wasting rate'+' '+'hit ratio'+' '+'accuracy'+' '+'training time per epoch'+' '+'mean_loss'+'\n'
    # run all result in different set [skip threshold, nun of targets]
    for n in [0.05*x for x in range(1,20)]:
        preditions_list=[]
        tag_list = []
        for sentence,tags in testing_set:
            # sentence_in = turn_LongTensor(sentence)
            sentence_in = [turn_FloatTensor(sentence[0]),turn_LongTensor(sentence[1]),turn_LongTensor(sentence[2])]
            target_buff = []
            for i in tags:
                target_buff.append(i)
            targets = turn_LongTensor(target_buff)
            tag_scores = model(sentence_in)
            for i in range(len(sentence[0])):
                a = tag_scores[i].data.numpy()
            preditions_list.append(tag_scores)
            tag_list.append(target_buff)
        result+= str(n) + ' ' + get_score(tag_list, preditions_list, n, null_num, 1)
        # result+= str(n) + ' ' + T_score(tag_list, preditions_list, null_num)[0]
    fp = open('RNN_threshold_result'+'.txt','w')
    fp.write(result)
    fp.close()
#Testing and save the results in a 3D matrix
def testing_3D_model(model_name):
    # #----------------------
    # training_set,testing_set, cat_num, y_cat_num, null_num= Label_encoding_preprocessing(PATH, SET_DEVIDE)
    # #model = multi_regular_LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,HIDDEN_DIM2,cat_num, cat_num)
    # model = regular_LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,cat_num, y_cat_num)
    # #----------------------
    # B set
    #With time_user features
    # training_set,testing_set, cat_num, y_cat_num, null_num,user_cat= Time_User_encoding_preprocessing(PATH, SET_DEVIDE)
    # model = with_added_feature_LSTMTagger(TIME_EMBEDDING, USER_EMBEDDING, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM_2,HIDDEN_DIM_3,cat_num, y_cat_num, 820, 24)
    #----------------------
    # C set
    #Based on trained word embedding_dim
    training_set,testing_set, cat_num, y_cat_num, null_num,user_cat= Url2vec_encoding(PATH, SET_DEVIDE)
    model =  with_added_feature_url2vec_LSTMTagger(TIME_EMBEDDING, USER_EMBEDDING, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM_2,HIDDEN_DIM_3,cat_num, y_cat_num, 820, 24)
    #----------------------
    model.load_state_dict(torch.load(model_name+'.pkl'))
    Z = {}
    total = 19*20
    count =0
    for n in [0.05*x for x in range(1,20)]:
        for n_block in range(1,2):
            count+=1
            preditions_list=[]
            tag_list = []
            for sentence,tags in testing_set:
                # sentence_in = [turn_LongTensor(sentence[0]),turn_LongTensor(sentence[1]),turn_LongTensor(sentence[2])]
                sentence_in = [turn_FloatTensor(sentence[0]),turn_LongTensor(sentence[1]),turn_LongTensor(sentence[2])]
                target_buff = []
                for i in tags:
                    target_buff.append(i)
                targets = turn_LongTensor(target_buff)
                tag_scores = model(sentence_in)
                for i in range(len(tags)):
                    a = tag_scores[i].data.numpy()
                preditions_list.append(tag_scores)
                tag_list.append(target_buff)
            Z[str(n)[:4]+'_'+str(n_block)[:4]] = get_value_score(tag_list, preditions_list, n, null_num, n_block)
            sum = Encoder.progress_bar(count, total)
    obj_save(Z, '3D_dict_withhumanfeature_19_withFS.txt')
    # fp = open('RNN_threshold_result'+'.txt','w')
    # fp.write(result)
    # fp.close()
#----------------------------------
#Testing in a stream environment
def streaming_testing(model_name):
    # #----------------------
    # training_set,testing_set, cat_num, y_cat_num, null_num= Label_encoding_preprocessing(PATH, SET_DEVIDE)
    # #model = multi_regular_LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,HIDDEN_DIM2,cat_num, cat_num)
    # model = regular_LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,cat_num, y_cat_num)
    # #----------------------
    # B set
    #With time_user features
    # training_set,testing_set, cat_num, y_cat_num, null_num,user_cat= Time_User_encoding_preprocessing(PATH, SET_DEVIDE)
    # model = with_added_feature_LSTMTagger(TIME_EMBEDDING, USER_EMBEDDING, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM_2,HIDDEN_DIM_3,cat_num, y_cat_num, 820, 24)
    #----------------------
    # C set
    #Based on trained word embedding_dim
    training_set,testing_set, cat_num, y_cat_num, null_num,user_cat= Url2vec_encoding_stream(PATH, SET_DEVIDE)
    model =  with_added_feature_url2vec_LSTMTagger(TIME_EMBEDDING, USER_EMBEDDING, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM_2,HIDDEN_DIM_3,cat_num, y_cat_num, 820, 24)
    #----------------------
    model.load_state_dict(torch.load(model_name+'.pkl'))
    dy_result = 'bursty_time_stamp'+' '+'skip/predict ratio'+' '+'correct skipping rate'+' '+'wasting operation num'+' '+'wasting rate'+' '+'hit ratio'+' '+'accuracy'+' '+'training time per epoch'+' '+'mean_loss'+'\n'
    fixed_result = 'bursty_time_stamp'+' '+'skip/predict ratio'+' '+'correct skipping rate'+' '+'wasting operation num'+' '+'wasting rate'+' '+'hit ratio'+' '+'accuracy'+' '+'training time per epoch'+' '+'mean_loss'+'\n'
    no_result = 'bursty_time_stamp'+' '+'skip/predict ratio'+' '+'correct skipping rate'+' '+'wasting operation num'+' '+'wasting rate'+' '+'hit ratio'+' '+'accuracy'+' '+'training time per epoch'+' '+'mean_loss'+'\n'
    preditions_list=[]
    tag_list = []
    #time sort
    stream_test = sorted(testing_set)
    burst_dict = obj_load('./bursty_withvar_list.txt')
    X = []
    burst_cap_count =0
    for ts, sentence,tags in stream_test:
        # sentence_in = turn_LongTensor(sentence)
        if ts in burst_dict:
            if len(X)>0:
                dy_result+= str(ts) + ' ' + get_score(tag_list, preditions_list, theta_s*0.85, null_num, n_block)+' '+str(burst_cap_count)+'\n'
                fixed_result+= str(ts) + ' ' + get_score(tag_list, preditions_list, 0.7, null_num, 8)+' '+str(burst_cap_count)+'\n'
                no_result+= str(ts) + ' ' + get_score(tag_list, preditions_list, 0.7, null_num, 1)+' '+str(burst_cap_count)+'\n'
                burst_cap_count=0
                preditions_list=[]
                tag_list = []
            X.append(ts)
            theta_s = burst_dict[ts][0]
            n_block = burst_dict[ts][1]
        sentence_in = [turn_FloatTensor(sentence[0]),turn_LongTensor(sentence[1]),turn_LongTensor(sentence[2])]
        target_buff = []
        for i in tags:
            target_buff.append(i)
        targets = turn_LongTensor(target_buff)
        tag_scores = model(sentence_in)
        for i in range(len(sentence[0])):
            a = tag_scores[i].data.numpy()
        preditions_list.append(tag_scores)
        tag_list.append(target_buff)
        burst_cap_count +=1
    fp = open('Dynamic_mode'+'.txt','w')
    fp.write(dy_result)
    fp.close()
    fp = open('Fixed_mode'+'.txt','w')
    fp.write(fixed_result)
    fp.close()
    fp = open('Only_Highest_term'+'.txt','w')
    fp.write(no_result)
    fp.close()
#-----------------------------------------
if __name__ == "__main__":
    # visual()  # infulence by both skipping rate and number of targets
    # simulation()  # overloading simulation
    testing_3D_model('./all_1_with_feature_all_15epoch_best') # testing and save the results in a 3D matrix
    # testing_model('./all_1_with_feature_all_15epoch_best') # testing
    # training_model('./testing_progress_bar', 0) # training
    # streaming_testing('./all_1_with_feature_all_15epoch_best') # Hit ratio testing in streaming environment

from gensim.models.word2vec import Word2Vec
import os

def get_files(path):
    dir_list = os.listdir(path)
    data_set = []
    tube =()
    link = []
    for dir in dir_list:
        try:
            data_path = path+dir
            fp = open(data_path)
            features = fp.readlines()
            sequence = []
            for vec in features:
                vec = vec.split(' ')
                # if vec[3][-4:-1] == "gif" or vec[3][-4:-1] == "GIF":
                #     pass
                # else:
                sequence.append(vec[3])
                    #link.append(vec[3])
            if len(sequence) >3:
                data_set.append(sequence)
        except:
            print('errors with data')
    return data_set

def read_data():
    return get_files('BU_dataset/b19/Apr95/')

sentences = read_data()
model= Word2Vec()
model.build_vocab(sentences)
model.train(sentences, total_examples = model.corpus_count, epochs = 200)
model.save('/tmp/MyModel')


#model.save_word2vec_format('/tmp/mymodel.txt',binary = False)
#model.save_word2vec_format('/tmp/mymodel.bin.gz',binary = True)

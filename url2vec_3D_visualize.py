#!usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import math
import random
import jieba
import numpy as np
from six.moves import xrange
import tensorflow as tf
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


#Download the data.
# Read the data into a list of strings.
def read_data():
    return get_files('BU_dataset/b19/Apr95/')


#step 1:读取文件中的内容组成一个列表
sequences = read_data()
words =[]
for i in sequences:
    for p in i:
        words.append(p)

print('Data size', len(words))

#Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000
def build_dataset(words):

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    print("count",len(count))

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))


    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  #删除words节省内存
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


data_index = 0
#Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],'->', labels[i, 0], reverse_dictionary[labels[i, 0]])

#Build and train a skip-gram model.
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
valid_size = 3     #切记这个数字要和len(valid_word)对应，要不然会报错哦
valid_window = 100
num_sampled = 64    # Number of negative examples to sample.



#Validation
valid_word = ['"http://www.cqs.washington.edu/icons/text.xbm"','"http://www.cqs.washington.edu/icons/unknown.xbm"','"http://cs-www.bu.edu/students/grads/tahir/CS111/"','"http://cs-www.bu.edu/lib/logo/Todd_Ginsberg0.gif"']
valid_examples =[dictionary[li] for li in valid_word]

graph = tf.Graph()
with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]),dtype=tf.float32)
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,biases=nce_biases, inputs=embed, labels=train_labels,
                 num_sampled=num_sampled, num_classes=vocabulary_size))
    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    # Add variable initializer.
    init = tf.global_variables_initializer()

#Begin training.
num_steps = 20000
with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 3  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[:top_k]
                try:
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
                except:
                    pass
    final_embeddings = normalized_embeddings.eval()


#Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='tsne.png',fonts=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    #ax.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y, z= low_dim_embs[i, :]
        ax.scatter(x, y, z)
        # ax.annotate(label,
        #             fontproperties=fonts,
        #             xy=(x, y, z),
        #             xytext=(5, 2),
        #             textcoords='offset points',
        #             ha='right',
        #             va='bottom')
        ax.text(x,y,z,  '%s' % (label), size=6, zorder=1,
 color='k')
    fig.savefig(filename, dpi = 600)
    plt.show()


try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.mplot3d import Axes3D
    #font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    tsne = TSNE(perplexity=30, n_components=3, init='pca', n_iter=5000)
    plot_only = 60
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

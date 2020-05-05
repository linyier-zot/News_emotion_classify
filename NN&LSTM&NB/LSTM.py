import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from utils import load_curpus
from gensim.models import FastText

start = time.clock()

# 加载停用词
stopwords = []
with open("stopwords.txt", "r", encoding="utf8") as f:
    for w in f:
        stopwords.append(w.strip())



# 加载训练好的FastText模型
model = FastText.load("model/model_100.txt")

sentiment_vocab = ['agreeable', 'believable', 'good', 'hated', 'sad', 'worried', 'objective']
for the_emotion in sentiment_vocab:     #循环训练7个情感
    train_data = load_curpus("data/" + the_emotion + "/train.txt")
    test_data = load_curpus("data/" + the_emotion + "/test.txt")
    train_df = pd.DataFrame(train_data, columns=["content", "sentiment"])
    #    print(train_df)
    test_df = pd.DataFrame(test_data, columns=["content", "sentiment"])

    data_str = [" ".join(content) for content, sentiment in train_data] + \
               [" ".join(content) for content, sentiment in test_data]
    tfidf = TfidfVectorizer(token_pattern='\[?\w+\]?', stop_words=stopwords)
    tfidf_fit = tfidf.fit_transform(data_str)

    max_length = 128

    X_train, train_length, y_train = [], [], []
    for content, sentiment in train_data:
        X, y = [], sentiment
        for w in content[:max_length]:
            if w in model:
                X.append(np.expand_dims(model[w], 0))
        if X:
            length = len(X)
            X = X + [np.zeros_like(X[0])] * (max_length - length)
            X = np.concatenate(X)
            X = np.expand_dims(X, 0)
            X_train.append(X)
            train_length.append(length)
            y_train.append(y)

    X_test, test_length, y_test = [], [], []
    for content, sentiment in test_data:
        X, y = [], sentiment
        for w in content[:max_length]:
            if w in model:
                X.append(np.expand_dims(model[w], 0))
        if X:
            length = len(X)
            X = X + [np.zeros_like(X[0])] * (max_length - length)
            X = np.concatenate(X)
            X = np.expand_dims(X, 0)
            X_test.append(X)
            test_length.append(length)
            y_test.append(y)



    batch_size = 512
    lr = 1e-3
    hidden_size = 100

    X = tf.placeholder(shape=(batch_size, max_length, 100), dtype=tf.float32, name="X")
    L = tf.placeholder(shape=(batch_size), dtype=np.int32, name="L")
    y = tf.placeholder(shape=(batch_size, 1), dtype=np.float32, name="y")
    dropout = tf.placeholder(shape=(), dtype=np.float32, name="dropout")
    with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
        def lstm_cell(hidden_size, cell_id=0):
            # LSTM细胞生成器
            cell = rnn.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name='cell%d' % cell_id)
            cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout)
            return cell


        cell = rnn.MultiRNNCell([lstm_cell(hidden_size, 0),
                                 lstm_cell(hidden_size, 1)], state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        cell_output, cell_state = tf.nn.dynamic_rnn(cell, X,
                                                    sequence_length=L,
                                                    initial_state=initial_state,
                                                    dtype=tf.float32)
        W1 = tf.get_variable("W1", shape=(hidden_size, 50))
        b1 = tf.get_variable("b1", shape=(50,))
        W2 = tf.get_variable("W2", shape=(50, 1))
        b2 = tf.get_variable("b2", shape=(1,))
        fcn1 = tf.nn.xw_plus_b(cell_state[1][1], W1, b1)
        logists = tf.nn.xw_plus_b(fcn1, W2, b2)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logists, labels=y))
        op = tf.train.AdamOptimizer(lr).minimize(loss)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)

    total_step = 1001
    step = 0
    cursor = 0
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    while step < total_step:
        _X, _L, _y = X_train[cursor: cursor + batch_size], train_length[cursor: cursor + batch_size], y_train[
                                                                                                      cursor: cursor + batch_size]
        cursor += batch_size
        if len(_X) < batch_size:
            cursor = batch_size - len(_X)
            _X += X_train[: cursor]
            _L += train_length[: cursor]
            _y += y_train[: cursor]
        _X = np.concatenate(_X)
        _L = np.reshape(np.array(_L, dtype=np.int32), (-1))
        _y = np.reshape(np.array(_y, dtype=np.float32), (batch_size, 1))
        _, l = sess.run([op, loss], feed_dict={X: _X, L: _L, y: _y, dropout: .75})
        if step % 100 == 0:
            print("step:", step, " loss:", l)
            saver.save(sess, 'model/lstm/model', global_step=step)
        step += 1

    _X = np.concatenate(X_test + [np.zeros_like(X_test[0])] * (batch_size - len(X_test)))
    _L = np.array(test_length + [0] * (batch_size - len(test_length)))

    result = sess.run(tf.nn.sigmoid(logists), feed_dict={X: _X, L: _L, dropout: 1.})
    prediction = []
    for i in result[:len(X_test)]:
        if i > 0.5:
            prediction.append(1)
        else:
            prediction.append(0)



    print(metrics.classification_report(y_test, prediction))
    print('训练', the_emotion)
    print("准确率:", metrics.accuracy_score(y_test, prediction))

elapsed = (time.clock() - start)
print("Time used:", elapsed)

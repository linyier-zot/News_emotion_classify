from utils import tokenize, load_curpus
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import FastText
from sklearn import metrics
import warnings
import time
import tensorflow as tf

start = time.clock()

warnings.filterwarnings("ignore")

#加载停用词
stopwords = []
with open("stopwords.txt", "r", encoding="utf8") as f:
    for w in f:
        stopwords.append(w.strip())

# 加载训练好的FastText模型
model = FastText.load("model/model_100.txt")

sentiment_vocab = ['agreeable', 'believable', 'good', 'hated', 'sad', 'worried', 'objective']
for the_emotion in sentiment_vocab:     #循环训练7个情感
    train_data = load_curpus("data/"+the_emotion+"/train.txt")
    test_data = load_curpus("data/"+the_emotion+"/test.txt")
    train_df = pd.DataFrame(train_data, columns=["content", "sentiment"])
    test_df = pd.DataFrame(test_data, columns=["content", "sentiment"])

    data_str = [" ".join(content) for content, sentiment in train_data] + \
               [" ".join(content) for content, sentiment in test_data]
    tfidf = TfidfVectorizer(token_pattern='\[?\w+\]?', stop_words=stopwords)
    tfidf_fit = tfidf.fit_transform(data_str)

    key_words = 20

    X_train, y_train = [], []
    for content, sentiment in train_data:
        X, y = [], sentiment
        X_tfidf = tfidf.transform([" ".join(content)]).toarray()
        keywords_index = np.argsort(-X_tfidf)[0, :key_words]
        for w in content:
            if w in model and w in tfidf.vocabulary_ and tfidf.vocabulary_[w] in keywords_index:
                X.append(np.expand_dims(model[w], 0) * X_tfidf[0, tfidf.vocabulary_[w]])
        if X:
            X = np.concatenate(X)
            X = np.expand_dims(np.mean(X, axis=0), 0)
            X_train.append(X)
            y_train.append(y)

    X_test, y_test = [], []
    for content, sentiment in test_data:
        X, y = [], sentiment
        X_tfidf = tfidf.transform([" ".join(content)]).toarray()
        keywords_index = np.argsort(-X_tfidf)[0, :key_words]
        for w in content:
            if w in model and w in tfidf.vocabulary_ and tfidf.vocabulary_[w] in keywords_index:
                X.append(np.expand_dims(model[w], 0) * X_tfidf[0, tfidf.vocabulary_[w]])
        if X:
            X = np.concatenate(X)
            X = np.expand_dims(np.mean(X, axis=0), 0)
            X_test.append(X)
            y_test.append(y)



    batch_size = 1000
    lr = 1e-3
    X = tf.placeholder(shape=(batch_size, 100), dtype=tf.float32, name="X")
    y = tf.placeholder(shape=(batch_size, 1), dtype=np.float32, name="y")
    with tf.variable_scope("fcn", reuse=tf.AUTO_REUSE):
        W1 = tf.get_variable("W1", shape=(100, 50))
        b1 = tf.get_variable("b1", shape=(50,))
        W2 = tf.get_variable("W2", shape=(50, 1))
        b2 = tf.get_variable("b2", shape=(1,))
        fcn1 = tf.nn.xw_plus_b(X, W1, b1)
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
        _X, _y = X_train[cursor: cursor + batch_size], y_train[cursor: cursor + batch_size]
        cursor += batch_size
        if len(_X) < batch_size:
            cursor = batch_size - len(_X)
            _X += X_train[: cursor]
            _y += y_train[: cursor]
        _X = np.concatenate(_X)
        _y = np.reshape(np.array(_y, dtype=np.float32), (batch_size, 1))
        _, l = sess.run([op, loss], feed_dict={X: _X, y: _y})
        if step % 100 == 0:
            print("step:", step, " loss:", l)
            saver.save(sess, 'model/nn/model', global_step=step)
        step += 1

    _X = np.concatenate(X_test + [np.zeros_like(X_test[0])] * (batch_size - len(X_test)))

    result = sess.run(tf.nn.sigmoid(logists), feed_dict={X: _X})
    prediction = []
    for i in result[:len(X_test)]:
        if i > 0.5:
            prediction.append(1)
        else:
            prediction.append(0)

    print(metrics.classification_report(y_test, prediction))
    print("训练", the_emotion)
    print("准确率:", metrics.accuracy_score(y_test, prediction))

elapsed = (time.clock() - start)
print("Time used:",elapsed)
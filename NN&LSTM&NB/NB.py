from utils import tokenize, load_curpus
import numpy as np
import pandas as pd
import warnings
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

warnings.filterwarnings("ignore")


start = time.clock()

sentiment_vocab = ['agreeable', 'believable', 'good', 'hated', 'sad', 'worried', 'objective']
for the_emotion in sentiment_vocab:     #循环训练7个情感
    train_data = load_curpus("data/" + the_emotion + "/train.txt")
    test_data = load_curpus("data/" + the_emotion + "/test.txt")
    # print(train_data)
    train_df = pd.DataFrame(train_data, columns=["content", "sentiment"])
#    print(train_df)
    test_df = pd.DataFrame(test_data, columns=["content", "sentiment"])

    stopwords = []
    with open("stopwords.txt", "r", encoding="utf8") as f:
        for w in f:
            stopwords.append(w.strip())

    data_str = [" ".join(content) for content, sentiment in train_data] + \
               [" ".join(content) for content, sentiment in test_data]
    vectorizer = CountVectorizer(token_pattern='\[?\w+\]?', stop_words=stopwords)
    vectorizer.fit_transform(data_str)

    X_data, y_data = [], []
    for content, sentiment in train_data:
        X, y = [], sentiment
        X_data.append(" ".join(content))
        y_data.append(sentiment)
    X_train = vectorizer.transform(X_data)
    y_train = y_data

    X_data, y_data = [], []
    for content, sentiment in test_data:
        X, y = [], sentiment
        X_data.append(" ".join(content))
        y_data.append(sentiment)
    X_test = vectorizer.transform(X_data)
    y_test = y_data



    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    result = clf.predict(X_test)



    print(metrics.classification_report(y_test, result))
    print('训练', the_emotion)
    print("准确率:", metrics.accuracy_score(y_test, result))

elapsed = (time.clock() - start)
print("Time used:",elapsed)




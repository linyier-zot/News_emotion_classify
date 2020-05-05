import pandas as pd
import os

sentiment_vocab = ['agreeable', 'believable', 'good', 'hated', 'sad', 'worried', 'objective']

for name in sentiment_vocab:
    os.mkdir(name)
    df = pd.read_json('train.json')
    df = df.reset_index(drop=True)
    df.rename(columns={'news_content': 'content'}, inplace=True)
    df.rename(columns={name: 'sentiment'}, inplace=True)
    # 要转换出的所需情感数据    填入7个情感依次运行
    f = open(".\\" + name + "\\train.txt", 'w', encoding='utf-8')
    for i in range(len(df)):
        text = '888888,' + str(df['sentiment'].loc[i]) + ',' + (df['content'].loc[i])
        f.write(repr(text))
        f.write('\n')

    df = pd.read_json('test.json')
    df = df.reset_index(drop=True)
    df.rename(columns={'news_content': 'content'}, inplace=True)
    df.rename(columns={name: 'sentiment'}, inplace=True)
    f = open(".\\" + name + "\\test.txt", 'w', encoding='utf-8')
    for i in range(len(df)):
        text = '888888,' + str(df['sentiment'].loc[i]) + ',' + (df['content'].loc[i])
        f.write(repr(text))
        f.write('\n')

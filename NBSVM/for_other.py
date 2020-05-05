import pandas as pd
import os

from sklearn.model_selection import train_test_split

"""小组数据集合并、去重"""

temp = pd.DataFrame({})

for i in os.listdir('.\\data'):
    temp = pd.concat([temp, pd.read_json('.\\data\\' + i)])
temp = temp.drop_duplicates(['id'])
temp = temp.reset_index(drop=True)
output = open('Groupdataset.json', 'w', encoding="utf-8")
output.write(temp.to_json(force_ascii=False))  # 保存合并、去重后的小组数据集
output.close()

"""数据集清洗，转换为模型所需格式"""

droplist = ['news_content_translate_cn', 'news_title_translate_cn', 'news_emotion_basis', 'news_subject', 'news_type',
            'news_about_china', 'news_position', 'news_title_translate_en']
order = ['sentiment_1', 'sentiment_2', 'news_content', 'id']

Dataset = pd.read_json('.\\Groupdataset.json')  # 读取小组合数据
Dataset.drop(droplist, axis=1, inplace=True)  # 去除无用数据列
Dataset.rename(columns={'news_content_translate_en': 'news_content'}, inplace=True)  # 重命名列

Dataset['sentiment_1'] = 'none'  # 初始化两个情感列
Dataset['sentiment_2'] = 'none'

for i in range(len(Dataset)):  # 转换成模型所需数据格式
    Dataset['sentiment_1'].loc[i] = Dataset['news_emotion'].loc[i][0]
    if Dataset['news_emotion'].loc[i][-1] != Dataset['news_emotion'].loc[i][0]:
        Dataset['sentiment_2'].loc[i] = Dataset['sentiment_1'].loc[i]

for i in range(len(Dataset)):  # 清洗数据，移除无新闻内容的错误数据
    if Dataset['news_content'].loc[i] is None:
        Dataset.drop([i], inplace=True)

Dataset = Dataset.reset_index(drop=True)
Dataset.drop(['news_emotion'], axis=1, inplace=True)
Dataset = Dataset[order]

"""将示例数据集做额外数据集"""
Extra_dataset = pd.read_excel('.\\extra_dataset.xlsx')
Dataset = pd.concat([Dataset, Extra_dataset])
Dataset = Dataset.drop_duplicates('id')
Dataset = Dataset.reset_index(drop=True)

Dataset.to_excel('.\\Final_Groupdataset.xlsx', index=False)  # 保存最后输入模型的数据集


"""转化为其他模型需要"""
sentiment_vocab = ['agreeable', 'believable', 'good', 'hated', 'sad', 'worried', 'objective']
df = pd.concat([Dataset, pd.DataFrame(columns=sentiment_vocab)], axis=1)
df = df.fillna(0)  # 若有缺失值，自动填充为0
# 构造数据集格式，类别标签一共七列，0代表不属于该类，1代表属于  数据集中无sad情感，删除一列
# 详情可见print(df)输出
for i in range(len(df)):
    sentiment_1 = df['sentiment_1'].loc[i]
    sentiment_2 = df['sentiment_2'].loc[i]
    if sentiment_1 in sentiment_vocab:
        df[sentiment_1].loc[i] = 1
    if sentiment_2 in sentiment_vocab:
        df[sentiment_2].loc[i] = 1

df.to_json('.\\for_other.json')
train, test = train_test_split(df, random_state=42, test_size=0.15, shuffle=True)
train.to_json('.\\train.json')
test.to_json('.\\test.json')

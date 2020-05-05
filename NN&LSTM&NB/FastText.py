from utils import load_curpus
import pandas as pd

data = load_curpus("pre_process/train.txt") + load_curpus("pre_process/test.txt")
df = pd.DataFrame(data, columns=["content", "sentiment"])
print(df)

from gensim.models import FastText
model = FastText(df["content"], 
                 size=100,
                 window=5, 
                 min_count=3, # 只保留出现次数大于3的词语
                 iter=1000,  # 10000次训练
                 min_n=3,     # 默认为3,因为文本是中文这里改为2
                 max_n=6,     # 默认为6,因为文本是中文这里改为5
                 word_ngrams=1)

model.save("model/model_100.txt")
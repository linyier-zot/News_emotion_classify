import pandas as pd
import codecs
import os


sum_count = 0

for i in os.listdir('.\\data'):
    sum_count = sum_count + len(pd.read_json('.\\data\\'+i))

print("原小组数据合计：", sum_count, '条')
print("合并去重后：", len(pd.read_json('.\\Groupdataset.json')), '条')
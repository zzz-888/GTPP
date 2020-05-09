import json
import re
import os
import nltk
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from collections import defaultdict, Counter
stemmer = nltk.stem.SnowballStemmer('english')
stop_words = stopwords.words("english")

from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()


def preprocessing(raw_cont):
    no_http = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    # no_number = re.compile(r'\b\d+\b')
    # no_punctuation = re.compile(r"[\!\?\,\;\.\(\)\#\$\%\^\&\*\-\_\-\+\=\<\>\/\\\|\:\{\}\[\]\@\~\`\"\'\‘\“\·\~\！\@\#\￥\%\……\&\*\（\）\—\-\+\=\{\【\】\}\、\|\/\？\，\《\。\》\：\；]")
    new_text = no_http.sub("", raw_cont).replace("  ", " ")   # 去网址
    # new_text = no_number.sub("", new_text).replace("  ", " ")   # 去数字
    # new_text = no_punctuation.sub(" ", new_text).replace("  ", " ")  # 去标点符号
    new_text = re.sub(r"[^a-zA-Z]", " ", new_text)
    new_text = [lancaster_stemmer.stem(word) for word in nltk.word_tokenize(new_text) if lancaster_stemmer.stem(word) not in stop_words]
    return new_text

def convert_label(label_list):
    dict = {"P1": "1", "P2": "2", "P3": "3", "P4": "4", "P5": "5"}
    result = []
    for label in label_list:
        label = dict[label]
        result.append(label)
    return result

def delect_over5(text_list):
    new_text_list = []
    for c in text_list:
        c = c.lower()
        c = preprocessing(c) 
        new_text_list.append(c)
    print(new_text_list)
    frequency = defaultdict(int)
    for c in new_text_list:
        for token in c:
            frequency[token] += 1
    split_corpus = [[word for word in x if frequency[word] > 4] for x in new_text_list]
    return split_corpus


datafolder = "./sorted_data/"
df = pd.read_csv(os.path.join(datafolder,"mozilla.csv"), encoding="utf-8")
priority = list(df["priority"])
print(len(priority))
priority = convert_label(priority)
row_description = list(df["description"])
pre_description = delect_over5(row_description)
summary = list(df["summary"])
severity = list(df["severity"])
component = list(df["component"])
product = list(df["product"])
row_all = [i + " " + j + " " + k + " " + x + " " + g for i, j, k, x, g in zip(row_description, summary, severity, component, product)]
pre_all = delect_over5(row_all)
i = list(df["id"])
pri_list = []
des_list = []
id_list = []
all_list = []
pre_all_list = []
cont_dict = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
for des, pri, i, a, x, y in zip(pre_description, priority, i, pre_all, row_all, row_description):
    if cont_dict[pri] > 199:
        continue
    if (len(des) < 30 and (pri == "1" or pri == "2" or pri == "3")) and i != 479078: 
        continue
    pri_list.append(pri)
    des_list.append(y)
    id_list.append(i)
    pre_all_list.append(a)
    all_list.append(x)
    cont_dict[pri] += 1

data = {"priority": pri_list, "description": des_list, "id": id_list, "my_mothod": all_list, "result": pre_all_list}
df_data = pd.DataFrame(data) 
df_data.to_csv("./data/mozilla.csv", index=0)
print(cont_dict)
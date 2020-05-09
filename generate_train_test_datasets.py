import os
import re
import pickle     # pickle模块是以二进制的形式序列化后保存到文件中
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import nltk.stem
import numpy as np
import csv
from collections import defaultdict, Counter
from pandas.core.frame import DataFrame
import networkx as nx
from collections import OrderedDict  
from itertools import combinations 
import math
from tqdm import tqdm      
import logging
from nltk.corpus import stopwords
stemmer = nltk.stem.SnowballStemmer('english')
stop_words = stopwords.words("english")

from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
        
def nCr(n, r):
    f = math.factorial
    return int(f(n)/(f(r)*f(n-r)))


def preprocessing(raw_cont):
    no_http = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    # no_number = re.compile(r'\b\d+\b')
    # no_punctuation = re.compile(r"[\!\?\,\;\.\(\)\#\$\%\^\&\*\-\_\-\+\=\<\>\/\\\|\:\{\}\[\]\@\~\`\"\'\‘\“\·\~\！\@\#\￥\%\……\&\*\（\）\—\-\+\=\{\【\】\}\、\|\/\？\，\《\。\》\：\；]")
    new_text = no_http.sub("", raw_cont).replace("  ", " ")   # 去网址
    # new_text = no_number.sub("", new_text).replace("  ", " ")   # 去数字
    # new_text = no_punctuation.sub(" ", new_text).replace("  ", " ")  # 去标点符号
    new_text = re.sub(r"[^a-zA-Z]", " ", new_text)
    new_text = [lancaster_stemmer.stem(word) for word in nltk.word_tokenize(new_text) if stemmer.stem(word) not in stop_words]
    return new_text


def dummy_fun(doc):
    return doc

def word_word_edges(p_ij):
    word_word = []
    cols = list(p_ij.columns); cols = [str(w) for w in cols]
    for w1, w2 in tqdm(combinations(cols, 2), total=nCr(len(cols), 2)):
        if (p_ij.loc[w1,w2] > 0):
            word_word.append((w1,w2,{"weight":p_ij.loc[w1,w2]}))
    return word_word

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

def generate_text_graph(window=15):
    """ generates graph based on text corpus; window = sliding window size to calculate point-wise mutual information between words """
    logger.info("Preparing data...")
    datafolder = "./data/"
    df = pd.read_csv(os.path.join(datafolder,"gcc.csv"), encoding="utf-8")
    p = list(df["priority"])
    # p = convert_label(p)
    d = list(df["description"])
    d = delect_over5(d)
    data = {"c": d, "b": p}
    df_data = pd.DataFrame(data) 
    del df
    print(df_data)

    # df_data["c"] = df_data["c"].apply(lambda x: x.lower()).apply(lambda x: preprocessing(x))
    print(df_data)
    save_as_pickle("gcc_df_data_description.pkl", df_data)
    
    ### Tfidf
    logger.info("Calculating Tf-idf...")
    vectorizer = TfidfVectorizer(input="content", max_features=None, tokenizer=dummy_fun, preprocessor=dummy_fun)
    vectorizer.fit(df_data["c"])
    df_tfidf = vectorizer.transform(df_data["c"])
    df_tfidf = df_tfidf.toarray()
    vocab = vectorizer.get_feature_names()
    print(vocab)
    vocab = np.array(vocab)
    print(vocab)
    df_tfidf = pd.DataFrame(df_tfidf, columns=vocab)
    
    ### PMI between words
    names = vocab
    n_i  = OrderedDict((name, 0) for name in names)
    word2index = OrderedDict( (name,index) for index,name in enumerate(names) )

    occurrences = np.zeros( (len(names), len(names)), dtype=np.int32)
    # Find the co-occurrences:
    no_windows = 0; logger.info("Calculating co-occurences...")
    for l in tqdm(df_data["c"], total=len(df_data["c"])):
        for i in range(len(l) - window):
            no_windows += 1
            d = set(l[i: (i+window)])  # set() 函数创建一个无序不重复元素集
            for w in d:
                n_i[w] += 1       # w(i)值加一
            for w1, w2 in combinations(d, 2):
                i1 = word2index[w1]
                i2 = word2index[w2]

                occurrences[i1][i2] += 1  # w(i, j)值加一
                occurrences[i2][i1] += 1
    del df_data
    logger.info("Calculating PMI*...")
    ### convert to PMI
    p_ij = pd.DataFrame(occurrences, index = names, columns=names) / no_windows
    p_i = pd.Series(n_i, index=n_i.keys()) / no_windows

    del occurrences
    del n_i
    for col in p_ij.columns:
        p_ij[col] = p_ij[col] / p_i[col]
    for row in p_ij.index:
        p_ij.loc[row,:] = p_ij.loc[row,:] / p_i[row]
    print(p_ij)
    p_ij = p_ij + 1E-9
    print(p_ij)
    for col in p_ij.columns:
        p_ij[col] = p_ij[col].apply(lambda x: math.log(x))
    print(p_ij)
    del p_i
    ### Build graph
    logger.info("Building graph (No. of document, word nodes: %d, %d)..." %(len(df_tfidf.index), len(vocab)))
    G = nx.Graph()
    logger.info("Adding document nodes to graph...")
    G.add_nodes_from(df_tfidf.index) ## document nodes
    logger.info("Adding word nodes to graph...")
    G.add_nodes_from(vocab) ## word nodes
    ### build edges between document-word pairs
    logger.info("Building document-word edges...")
    document_word = [(doc,w,{"weight":df_tfidf.loc[doc,w]}) for doc in tqdm(df_tfidf.index, total=len(df_tfidf.index))\
                     for w in df_tfidf.columns]
    
    logger.info("Building word-word edges...")
    word_word = word_word_edges(p_ij)
    save_as_pickle("gcc_word_word_edges_description.pkl", word_word)
    del p_ij
    logger.info("Adding document-word edges...")
    G.add_edges_from(document_word)
    logger.info("Adding word-word edges...")
    G.add_edges_from(word_word)
    save_as_pickle("gcc_text_graph_description.pkl", G)
    logger.info("Done and saved!")
    
if __name__=="__main__":
    generate_text_graph()    
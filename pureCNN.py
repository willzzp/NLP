# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 22:25:37 2018
@author: Administrator
"""
################################初始化##########################################
import numpy as np
np.random.seed(42)
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Dropout
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Merge, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from sklearn.model_selection import KFold
from keras import backend as K      #attention
from keras.engine.topology import Layer   #attention
from keras.layers import *
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')



################################数据读入##########################################

EMBEDDING_FILE = 'C:/Users/Mzzp/Desktop/workshop/toxic/glove.twitter.27B.200d.txt'
#glove词向量 :EMBEDDING_FILE
train = pd.read_csv('C:/Users/Mzzp/Desktop/workshop/toxic/train.csv')
#训练集train
submission = pd.read_csv('C:/Users/Mzzp/Desktop/workshop/toxic/sample_submission.csv')
test = pd.read_csv('C:/Users/Mzzp/Desktop/workshop/toxicm/test.csv')
################################数据清理##########################################
rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)
def rpt_repl(match):
	return match.group(1)+match.group(1)

def clean_str(string):
    #deal with twitter
    eyes = "[8:=;]"
    nose = "['`\-]?"
    string = re.sub("https?:* ", "<URL>", string)
    string = re.sub("www.* ", "<URL>", string)
    string = re.sub("\[\[User(.*)\|", '<USER>', string)
    string = re.sub("<3", '<HEART>', string)
    string = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", string)
    string = re.sub(eyes + nose + "[Dd)]", '<SMILE>', string)
    string = re.sub("[(d]" + nose + eyes, '<SMILE>', string)
    string = re.sub(eyes + nose + "p", '<LOLFACE>', string)
    string = re.sub(eyes + nose + "\(", '<SADFACE>', string)
    string = re.sub("\)" + nose + eyes, '<SADFACE>', string)
    string = re.sub(eyes + nose + "[/|l*]", '<NEUTRALFACE>', string)
    string = re.sub("/", " / ", string)
    string = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", string)
    string = re.sub("([!]){2,}", "! <REPEAT>", string)
    string = re.sub("([?]){2,}", "? <REPEAT>", string)
    string = re.sub("([.]){2,}", ". <REPEAT>", string)
    pattern = re.compile(r"(.)\1{2,}")
    string = pattern.sub(r"\1" + " <ELONG>", string)
    #EMOJIS
    string = re.sub(r":\)"," emojihappy1",string)
    string = re.sub(r":P"," emojihappy2",string)
    string = re.sub(r":p"," emojihappy3",string)
    string = re.sub(r":>"," emojihappy4",string)
    string = re.sub(r":3"," emojihappy5",string)
    string = re.sub(r":D"," emojihappy6",string)
    string = re.sub(r" XD "," emojihappy7",string)
    string = re.sub(r" <3 "," emojihappy8",string)
    string = re.sub(r"&lt;3"," emojihappy9",string)
    string = re.sub(r":d"," emojihappy10",string)
    string = re.sub(r":dd"," emojihappy11",string)
    string = re.sub(r"8\)"," emojihappy12",string)
    string = re.sub(r":-\)"," emojihappy13",string)
    string = re.sub(r";\)"," emojihappy15",string)
    string = re.sub(r"\(-:"," emojihappy16",string)
    string = re.sub(r"\(:"," emojihappy14",string)
    string = re.sub(r":-D"," emojihappy17",string)
    string = re.sub(r"X-D"," emojihappy18",string)
    string = re.sub(r"xD"," emojihappy19",string)
    string = re.sub(r":\\*"," emojihappy20",string)
    string = re.sub(r";-D"," emojihappy21",string)
    string = re.sub(r";D"," emojihappy22",string)
    string = re.sub(r"\(;"," emojihappy23",string)
    string = re.sub(r"\(-;"," emojihappy24",string)
    # Repeating words like hurrrryyyyyy
    string=re.sub( rpt_regex, rpt_repl, string )
    #以防万一
    string = re.sub(r"yay!"," happygood",string)
    string = re.sub(r"yay"," happygood",string)
    string = re.sub(r"yaay!"," happygood",string)
    string = re.sub(r"yaaay!"," happygood",string)
    string = re.sub(r"yaaaay!"," happygood",string)
    string = re.sub(r"yaaaaay!"," happygood",string)
    string = re.sub(r"ha"," happyha",string)
    string = re.sub(r"haha"," happyha",string)
    string = re.sub(r"hahaha"," happyha",string)
    
    string = re.sub(r":/"," emojisad1",string)
    string = re.sub(r":&gt"," emojisad2",string)
    string = re.sub(r":'\)"," emojisad3",string)
    string = re.sub(r":-\("," emojisad4",string)
    string = re.sub(r":\("," emojisad5",string)
    string = re.sub(r":s"," emojisad6",string)
    string = re.sub(r":-s"," emojisad7",string)
    string = re.sub(r":\\\("," emojisad9",string)
    string = re.sub(r":<"," emojisad10",string)
    string = re.sub(r":<"," emojisad11",string)
    string = re.sub(r">:\\\("," emojisad12",string)
    string = re.sub(r":,\("," emojisad13",string)
    string = re.sub(r":\\'\("," emojisad14",string)
    string = re.sub(r":\(\("," emojisad15",string)
    string = re.sub(r":\"\("," emojisad1",string)
    
    #MENTIONS "(@)\w+"
    string = re.sub(r"(@)\w+"," mentiontoken",string)
    
    #WEBSITES
    string = re.sub(r"http(s)*:(\S)*"," linktoken",string)

    #STRANGE UNICODE \x...
    string = re.sub(r"\\x(\S)*","",string)

    #General Cleanup and Symbols
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

train['comment_text']=train['comment_text'].apply(clean_str)
test['comment_text']=test['comment_text'].apply(clean_str)
print('done')
################################进一步预处理##########################################

X_train = train["comment_text"].fillna("fillna").values
#X_train是train的评论列，用fillna补上空值，输出训练集的数据到x_train
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
#y_train是train的评分列，6个维度的数组表
X_test = test["comment_text"].fillna("fillna").values

max_features = 68000  #分词函数将选取出现频率最多的前68000个词
maxlen = 150          #填充的最大长度
embed_size = 200      #这个好像是维度？

tokenizer = text.Tokenizer(num_words=max_features)     #设置分词tokenizer类
#这个类用来对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示。传入词典的最大值num_words
tokenizer.fit_on_texts(list(X_train) + list(X_test))    #将训练集和预测集生成文档词典
#使用一系列文档X_train来生成token文档词典，texts为list类，每个元素为一个文档
X_train = tokenizer.texts_to_sequences(X_train)        #得到train中每个list元素的索引（用于查表）
X_test = tokenizer.texts_to_sequences(X_test)          #得到test中每个list元素的索引（用于查表）
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)    #把train中每个list索引的最大长度填充至 maxlen（保持一致）
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)     #把test中每个list索引的最大长度填充至 maxlen（保持一致）
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

################################词嵌入##########################################

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
#分别输出词和词向量
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE,encoding='utf-8'))
#读入glove，把词和词向量存入字典embeddings_index。
"""embeddings_index 是通过glove预训练词向量构造的一个字典，每个单词都有一个对应的300维度的词向量,
词向量来源于glove的预训练。接着，我们构造了一个embedding_matrix，只取了排名靠前的6.8W单词，
并且把词向量填充进embedding_matrix。参考 https://spaces.ac.cn/archives/4122/ 
Embedding层就是以one hot为输入、中间层节点为字向量维数的全连接层！而这个全连接层的参数，就是一个“字向量表”"""

word_index = tokenizer.word_index
#保存索引的词典，在词典中，每个单词都有一个对应的下标序号
print('Found %s unique tokens' % len(word_index))
nb_words = min(max_features, len(word_index))
#这个变量是出现频率前6.8万和词索引长度之间的较小值。
embedding_matrix = np.zeros((nb_words, embed_size))
#设定初始零矩阵
for word, i in word_index.items():                  #返回词典中可遍历的(键, 值) 元组数组
    if i >= max_features: continue                  #当i大于6.8万时,跳过（频率较低的词）
    embedding_vector = embeddings_index.get(word)   #赋值词的索引
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector  
# embedding index中找不到的词会是全零矩阵。

class RocAucEvaluation(Callback):                   #定义AUC函数 继承自keras.callbacks的子类
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()            #为子类调用了callbacks的初始化方法

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            #计算auc值（y_val是测试集，y_pred是预测集）
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

####################################模型################################################


def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    left = Conv1D(128, 4, activation='tanh')(x)
    left = MaxPooling1D(4)(left)
    #left = Conv1D(128, 4, activation='tanh')(left)
    #left = MaxPooling1D(4)(left)   
    left = Flatten()(left)
    
    mid = Conv1D(128, 5, activation='tanh')(x)
    mid = MaxPooling1D(5)(mid)
    #mid = Conv1D(128, 5, activation='tanh')(mid)
    #mid = MaxPooling1D(5)(mid)   
    mid = Flatten()(mid)

    right = Conv1D(128, 6, activation='tanh')(x)
    right = MaxPooling1D(5)(right)
    #right = Conv1D(128, 5, activation='tanh')(right)
    #right = MaxPooling1D(5)(right)   
    right = Flatten()(right)   
            
    conc = concatenate([left, mid, right])

    x = Dense(64, activation='relu')(conc)
    x = Dropout(0.3)(x)
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = get_model()

################################开始训练##########################################
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train,          #划分训练集和测试集
                                              train_size=0.95,          #训练集占比95%
                                              random_state=5)        #随机种子
print('Train...')
model.fit(X_tra, y_tra,                         
          batch_size=32, 
          epochs=2, 
          validation_data=(X_val, y_val)) 
          #callbacks=[RocAuc, early_stop],)
print('predict...')
y_pred = model.predict(x_test, batch_size=1024)

################################输出结果##########################################
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('C:/Users/Mzzp/Desktop/workshop/toxicm/submission.csv', index=False)

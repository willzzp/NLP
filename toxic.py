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
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from sklearn.model_selection import KFold
from keras import backend as K      #attention
from keras.engine.topology import Layer   #attention
from keras.layers import *

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'
#四线程
################################数据读入##########################################

EMBEDDING_FILE = 'C:/Users/Mzzp/Desktop/workshop/toxic/glove.twitter.27B.200d.txt'
#glove词向量 :EMBEDDING_FILE
train = pd.read_csv('C:/Users/Mzzp/Desktop/workshop/toxic/train.csv')
#训练集train
submission = pd.read_csv('C:/Users/Mzzp/Desktop/workshop/toxic/sample_submission.csv')

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
print('done')
################################进一步预处理##########################################

X_train = train["comment_text"].fillna("fillna").values
#X_train是train的评论列，用fillna补上空值，输出训练集的数据到x_train
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
#y_train是train的评分列，6个维度的数组表

max_features = 68000  #分词函数将选取出现频率最多的前68000个词
maxlen = 150          #填充的最大长度
embed_size = 200      #这个好像是维度？

tokenizer = text.Tokenizer(num_words=max_features)        #设置分词tokenizer类 
#这个类用来对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示。传入词典的最大值num_words
tokenizer.fit_on_texts(list(X_train))                       #生成文档词典
#使用一系列文档X_train来生成token文档词典，texts为list类，每个元素为一个文档
X_train = tokenizer.texts_to_sequences(X_train)           #得到train中每个list元素的索引
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)   #把train中每个list索引的最大长度填充至 maxlen（保持一致）


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
################################注意力模型##########################################

class Position_Embedding(Layer):
    
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size #必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)
        
    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size,seq_len = K.shape(x)[0],K.shape(x)[1]
        position_j = 1. / K.pow(10000., \
                                 2 * K.arange(self.size / 2, dtype='float32' \
                               ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1)-1 #K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)
        
    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)

'''
Multi-Head Attention的实现
'''
class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
        
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
                
    def call(self, x):
        #如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        #如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        #对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3])
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))    
        A = K.softmax(A)
        #输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
        
    def compute_output_shape(self, input_shape):
return (input_shape[0][0], input_shape[0][1], self.output_dim)


################################学习模型##########################################
def get_model():                            #定义学习模型
    inp = Input(shape=(maxlen, ))           #定义输入层，输入为maxlen长度的列向量。
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)    #嵌入层，把下表转化为向量
    x = Attention(8,16)([x,x,x])
    x = GlobalAveragePooling1D()(x)    #时域信号施加全局平均值池化
    x = Dropout(0.5)(x)
    outp = Dense(6, activation="sigmoid")(conc)  #Dense是普通全连接层，输出维度6,激活函数是sigmoid.
    
    model = Model(inputs=inp, outputs=outp)    #打包模型 输入输出
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
# 编译学习过程，此处选择损失函数为对数损失函数，优化器为adam, ( Adam算法可以看做是修正后的Momentum+RMSProp算法)
    return model

model = get_model()
################################训练和预测##########################################

batch_size = 32
epochs = 5
result=0
kf = KFold(n_splits=5,random_state=2018,shuffle=True)  #K折交叉验证
"""将训练/测试数据集划分n_splits个互斥子集，每次用其中一个子集当作验证集，
剩下的n_splits-1个作为训练集，进行n_splits次训练和测试，得到n_splits个结果
n_splits：表示划分几等份
shuffle：在每次划分时，是否进行洗牌
random_state：随机种子数"""
for train_index,test_index in kf.split(X_train):
    train,y_tra=x_train[train_index],y_train[train_index]
    test,y_tes=x_train[test_index],y_train[test_index]
    hist = model.fit(train, y_tra, batch_size=batch_size, epochs=epochs)
    y_pred = model.predict(test, batch_size=1024)
    result+=roc_auc_score(y_tes,y_pred)
print('score: ',result)

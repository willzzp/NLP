
import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Dropout
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Merge, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing import text, sequence
from keras.callbacks import Callback

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'


EMBEDDING_FILE = '../input/glove-global-vectors-for-word-representation/glove.twitter.27B.200d.txt'

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

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
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

train['comment_text']=train['comment_text'].apply(clean_str)
test['comment_text']=test['comment_text'].apply(clean_str)
print('done')

X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values


max_features = 68000
maxlen = 150
embed_size = 200

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE,encoding='utf-8'))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

####################################模型################################################

# train a 1D convnet with global maxpoolinnb_wordsg

#left model 第一块神经网络，卷积窗口是5*50（50是词向量维度）
model_left = Sequential()
#model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
model_left.add(embedding_layer)
model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(5))
model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(5))
model_left.add(Conv1D(128, 5, activation='tanh'))
model_left.add(MaxPooling1D(35))
model_left.add(Flatten())

#right model 第二块神经网络，卷积窗口是4*50

model_right = Sequential()
model_right.add(embedding_layer)
model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(4))
model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(4))
model_right.add(Conv1D(128, 4, activation='tanh'))
model_right.add(MaxPooling1D(28))
model_right.add(Flatten())

#third model 第三块神经网络，卷积窗口是6*50
model_3 = Sequential()
model_3.add(embedding_layer)
model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(3))
model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(3))
model_3.add(Conv1D(128, 6, activation='tanh'))
model_3.add(MaxPooling1D(30))
model_3.add(Flatten())


merged = Merge([model_left, model_right,model_3], mode='concat') # 将三种不同卷积窗口的卷积层组合 连接在一起，当然也可以只是用三个model中的一个，一样可以得到不错的效果，只是本文采用论文中的结构设计
model = Sequential()
model.add(merged) # add merge
model.add(Dense(128, activation='tanh')) # 全连接层
model.add(Dense(len(labels_index), activation='softmax')) # softmax，输出文本属于20种类别中每个类别的概率

# 优化器我这里用了adadelta，也可以使用其他方法
model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])


####################################模型################################################
batch_size = 32
epochs = 5

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc])


y_pred = model.predict(x_test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)

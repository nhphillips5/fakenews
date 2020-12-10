import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
recall_score, roc_auc_score, roc_curve, accuracy_score)


full_train = pd.read_csv("/Users/nickphillips/Downloads/fake-news/train.csv")
after_test = pd.read_csv("/Users/nickphillips/Downloads/fake-news/test.csv")

full_train['set'] = 'train'

after_test['set'] = 'test'

concat_df = pd.concat([full_train, after_test])

concat_df.text = concat_df.text.astype(str)

concat_df = concat_df.fillna(' ')


sw = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
        'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
        'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
        'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
        'about', 'against', 'between', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
        'each', 'few', 'more', 'mot', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'ver', 's', 't',
        'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
        'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
        'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
        "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
        'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
        "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
        'won', "won't", 'wouldn', "wouldn't"]

tfidf = TfidfVectorizer(min_df = 50, stop_words = sw)
tfidf.fit(concat_df['text'])

train_df = concat_df[concat_df['set'] == 'train']
test_df = concat_df[concat_df['set'] == 'test']

train, test = train_test_split(train_df, test_size=.4, stratify=train_df.label)

X_train = tfidf.transform(train['text'])
X_test = tfidf.transform(test['text'])

y_train = (train['label'] == 1)
y_test = (test['label'] == 1)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth = 100, n_estimators = 500, n_jobs = -1)
rf.fit(X_train, y_train)
yhatrf = (rf.predict(X_test))
y_probrf = rf.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_probrf))
print(accuracy_score(y_test, yhatrf))
print(f1_score(y_test, yhatrf))


new_test = tfidf.transform(test_df['text'])

submit = rf.predict(new_test)
submit = submit.astype(int)
submit = pd.Series(submit, name = 'label')

submission = pd.concat([after_test.id, submit], axis=1)
submission = submission.set_index('id')
submission.to_csv('test.csv')

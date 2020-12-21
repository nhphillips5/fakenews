# fakenews
Fake News Kaggle competition

## Fake News Detection

This will be a basic text classification, using Scikit Learn's Random Forest Classifier. First let's import what we'll need.

Just by default, I import numpy and pandas. Numpy is a package for python that allows for scientific computing. It allows for large matrices of data and has many functions which make manipulating those easy. While I don't think I expressly use it here, it's good to have imported at the beginning. Pandas is another package that is built off of Numpy and it is great for creating and manipulating DataFrames. You'll see in the code I use it alot. The np and pd are aliases that I'm giving these two packages so that I don't have to type out their full names to access their functions.

Sci-Kit Learn (sklearn) is Python's main machine learning package. I pull out several specific functions that I need for this. And last but not least Nltk is the Natural Language Toolkit. It has lots of awesome functions for natural language processing. Here I will just be using their stopwords. Stopwords are words that are very common but don't give lots of meaning. We get rid of these because they can make analysis foggier by adding noise.

Typically here I would start by importing numpy and pandas and then grab the others as I need them.

```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
recall_score, roc_auc_score, roc_curve, accuracy_score)
from nltk.corpus import stopwords
```

Now, let's read in the data and combine it before we clean the data.

Pandas read_csv function reads in the data from a csv file. Then I'm adding a column in each dataset named 'set', with all the values being either 'train' or 'test' depending on which dataset it came from. I'm doing this to make it easy to separate the two once I'm done cleaning the data.

Pandas concat function allows us to combine the two datasets essentially stacking one on top of the other.

After I combine the two datasets I start to do some basic cleaning. I first make sure that all the text is a string type in python. this will allow us to tokenize (split up the text into individual words) the text. Finally, I need to make sure there are no null values in our dataset. To fix this I change all the null values to and empty space (' ') using the fillna() function.

```
full_train = pd.read_csv("/Users/nickphillips/Downloads/fake-news/train.csv")
after_test = pd.read_csv("/Users/nickphillips/Downloads/fake-news/test.csv")

full_train['set'] = 'train'

after_test['set'] = 'test'

concat_df = pd.concat([full_train, after_test])

concat_df.text = concat_df.text.astype(str)

concat_df = concat_df.fillna(' ')
```

Now we will establish the stopwords we will use which are from the nltk corpus.

```
sw = stopwords.words('english')
```

Here we will apply our TFIDF vectorizor. Min_df means it will dismiss all words that aren't mentioned more that 50 times. This will also ignore all of our stopwords which are words that don't really add much meaning, like the, and, but, ect. Then we will separate our two datasets using the 'set' column we made earlier.

```
tfidf = TfidfVectorizer(min_df = 50, stop_words = sw)
tfidf.fit(concat_df['text'])

train_df = concat_df[concat_df['set'] == 'train']
test_df = concat_df[concat_df['set'] == 'test']
```

Now we will split our training data set using train_test_split(). The first argument is the dataframe that I am splitting up. the test_size = .4 means that the test will be made with 40% of the full dataset. And the stratify argument allows us to stratify based on the column of our choosing. This way we make sure our test and train have equivalent proportions of posative and negative labels for fake news. Afterward we apply our tfidf vectorizor to our train and test text.

```
train, test = train_test_split(train_df, test_size=.4, stratify=train_df.label)

X_train = tfidf.transform(train['text'])
X_test = tfidf.transform(test['text'])

y_train = (train['label'] == 1)
y_test = (test['label'] == 1)
```

Now, we just fit the model and use it to predict our test sub-dataset to see how it does then we can use it to predict our actual dataset.

First, we import the random forest classifier. Then, we instanciate the model. Max_depth is the maximum depth of the trees in the forest. N_estimators is the number of trees created. Finally, n_jobs means how many jobs we want to run in parrellel, -1 means to use all processors.

We fit then the model using the fit() method. After we fit the model we will use it to predict X_test using the predict() method. We'll then use the predict_proba() method. This method returns the probabilities that the news isn't or is fake, and since we only want the probability it is fake we have `[:,1]` to slice just those values.

Then we get a few metrics to see how our model did. Here I check ROC AUC score, accuracy, and F1.

```
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth = 100, n_estimators = 500, n_jobs = -1)
rf.fit(X_train, y_train)
yhatrf = (rf.predict(X_test))
y_probrf = rf.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_probrf))
print(accuracy_score(y_test, yhatrf))
print(f1_score(y_test, yhatrf))
```

Looks like we are about 95% accurate! Awesome!

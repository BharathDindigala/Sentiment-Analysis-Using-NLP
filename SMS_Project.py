#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


file_path = "./sms/spam.csv"


# In[4]:


dataset = pd.read_csv(file_path, encoding="latin1")


# In[5]:


dataset.head()


# In[6]:


dataset.columns


# In[7]:


dataset.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)


# In[8]:


dataset.head()


# In[9]:


dataset['v1'] = dataset['v1'].map({'ham': 0, 'spam': 1})


# In[10]:


dataset.head()


# In[11]:


dataset.isna().sum()


# In[12]:


sns.countplot(data=dataset, x="v1", palette=["blue", "red"])

plt.show()


# In[13]:


import re
import nltk
nltk.download('stopwords')


# In[14]:


from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# In[15]:


corpus = []


# In[16]:


dataset["v2"][1]


# In[17]:


ss = SnowballStemmer(language="english")
for i in range(0, len(dataset)):
    # Replace all non Alphabets and non numerics, substitute with space in particular place 
    message = re.sub(r'\W', ' ', dataset["v2"][i])
    message = message.lower()
    message = message.split()
    message = [ss.stem(word) for word in message if not word in set(stopwords.words("english"))]
    message = " ".join(message)
    corpus.append(message)


# In[18]:


corpus[0:10]


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values


# In[20]:


x.shape


# In[28]:


x[:10]


# In[22]:


y


# In[29]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[30]:


x_train.shape


# In[31]:


y_train.shape


# In[32]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train, y_train)


# In[33]:


y_pred = clf.predict(x_test)


# In[34]:


y_pred[:10]


# In[35]:


y_test[:10]


# In[36]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)


# In[37]:


from sklearn.metrics import precision_score, recall_score
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))


# In[ ]:





# In[ ]:


# Custom Prediction


# In[43]:


txt = "There is 50% offer on this latest item"


# In[44]:


txt = np.array([txt])


# In[45]:


text = cv.transform(txt)


# In[46]:


text


# In[50]:


test = clf.predict(text) # 0-Ham, 1-Spam


# In[49]:


test[0]


# In[51]:


if test[0] == 0:
    print("Ham")
else:
    print("Spam")


# In[52]:


import pickle


# In[54]:


with open("sms_model.pl", "wb") as f:
    pickle.dump(clf, f)


# In[ ]:





import numpy as np
import pandas as pd
import pickle
headers=['url','state']
df=pd.read_csv('datathon_train.csv',names=headers)
#df.isnull().sum()
df['url']=df['url'].apply(lambda x:x.replace("://"," "))
df['url']=df['url'].apply(lambda x:x.replace("/"," "))
df['url']=df['url'].apply(lambda x:x.replace("."," "))
df['url']=df['url'].apply(lambda x:x.replace("-"," "))
df['url']=df['url'].apply(lambda x:x.replace("%20"," "))
#df.head()
from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
vec=v.fit_transform(df['url'])
#vec.shape
y=df['state']
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
x_train,x_test,y_train,y_test=train_test_split(vec,y,test_size=0.2,random_state=42)
n_samples, n_features = 506, 13
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
clf = Ridge(alpha=1.0)
clf.fit(x_test,y_test)
Ridge()
y_pred1=clf.predict(x_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred1))

# coding: utf-8

# In[2]:

# Import necessary packages
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import csv as csv


# In[3]:

# Read the train and test data into pandas data framesfrom StringIO import StringIO
from StringIO import StringIO
import requests
import json

# This function includes credentials to your Object Storage.
# You might want to remove those credentials before you share your notebook.
def get_object_storage_file_with_credentials_f2ced6cdb4894c42945223d461f16d59(container, filename):
    """This functions returns a StringIO object containing
    the file content from Bluemix Object Storage V3."""

    url1 = ''.join(['https://identity.open.softlayer.com', '/v3/auth/tokens'])
    data = {'auth': {'identity': {'methods': ['password'],
            'password': {'user': {'name': 'admin_d62aa33975e0d99158757aee850892d2cdfcf245','domain': {'id': 'b00f5856f86c411e99a1ed33654a0ec7'},
            'password': 'gtdtV_)A5^rYuev9'}}}}}
    headers1 = {'Content-Type': 'application/json'}
    resp1 = requests.post(url=url1, data=json.dumps(data), headers=headers1)
    resp1_body = resp1.json()
    for e1 in resp1_body['token']['catalog']:
        if(e1['type']=='object-store'):
            for e2 in e1['endpoints']:
                        if(e2['interface']=='public'and e2['region']=='dallas'):
                            url2 = ''.join([e2['url'],'/', container, '/', filename])
    s_subject_token = resp1.headers['x-subject-token']
    headers2 = {'X-Auth-Token': s_subject_token, 'accept': 'application/json'}
    resp2 = requests.get(url=url2, headers=headers2)
    return StringIO(resp2.content)

train = pd.read_csv(get_object_storage_file_with_credentials_f2ced6cdb4894c42945223d461f16d59('notebooks', 'train.csv'), dtype={"Age": np.float64, "Fare": np.float64},)
test = pd.read_csv(get_object_storage_file_with_credentials_f2ced6cdb4894c42945223d461f16d59('notebooks', 'test.csv'), dtype={"Age": np.float64, "Fare": np.float64},)


# In[4]:

# Preview of the training data
train.head()


# In[6]:

#Data audit
## Age has some missing values that can be imputed for downstream analysis; 
## Cabin has too many missings to be immediately useful
train.describe(include='all')


# In[7]:

#Replace missing values in Age
##Compute the average age for passengers by sex and class
train_aggSexPclass = train.groupby(['Sex','Pclass'], as_index=False).aggregate(np.mean).drop(['PassengerId','Survived','SibSp','Parch','Fare'], axis=1)
train_aggSexPclass = train_aggSexPclass.rename(columns={'Age': 'Age_Mean'})
print(train_aggSexPclass)


# In[8]:

# Merge the mean Age values by Sex and Pclass with the primary dataframe
train_merge = pd.merge(train,train_aggSexPclass,on=['Sex','Pclass'])
# Replace the missing values of Age with Age_Mean, then drop Age_Mean
train_merge.loc[train_merge.Age.isnull(), 'Age'] = train_merge.loc[train_merge.Age.isnull(), 'Age_Mean'] 
train_merge = train_merge.drop('Age_Mean', axis=1)
train_merge.describe(include='all')


# In[9]:

# Create an array of numeric fields that can be used to train the estimator
train_data = train_merge.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'], axis=1).values


# In[10]:

# Training...
forest_estimator = RandomForestClassifier(n_estimators=10)
forest_model = forest_estimator.fit( train_data[0::,1::], train_data[0::,0] )


# In[11]:

# Merge the mean Age values by Sex and Pclass with the primary test dataframe
test_merge = pd.merge(test,train_aggSexPclass,on=['Sex','Pclass'])
# Replace the missing values of Age with Age_Mean, then drop Age_Mean
test_merge.loc[test_merge.Age.isnull(), 'Age'] = test_merge.loc[test_merge.Age.isnull(), 'Age_Mean'] 
# Replace the missing values of Fare with the overall (test) mean of Fare
test_merge['Fare_Mean'] = np.mean(test_merge.Fare)
test_merge.loc[test_merge.Fare.isnull(), 'Fare'] = test_merge.loc[test_merge.Fare.isnull(), 'Fare_Mean'] 
test_merge = test_merge.drop(['Age_Mean','Fare_Mean'], axis=1)
test_merge.describe(include='all')


# In[12]:

# Create an array of numeric fields that can be used for scoring on the built model
test_data = test_merge.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'], axis=1).values


# In[13]:

# Score test data ...
forest_scores = forest_model.predict(test_data).astype(int)


# In[63]:

# ... and write out to file
predictions_file = open("titanic_python_01.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(test_merge['PassengerId'], forest_scores))
predictions_file.close()


# In[ ]:




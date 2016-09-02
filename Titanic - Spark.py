
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pyspark.sql.functions
from pyspark.sql.functions import array


# In[2]:

#Read the train and test data into pandas data framesfrom pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

# This function includes credentials to your Object Storage.
# You might want to remove those credentials before you share your notebook.
def set_hadoop_config_with_credentials_f2ced6cdb4894c42945223d461f16d59(name):
    """This function sets the Hadoop configuration so it is possible to
    access data from Bluemix Object Storage V3 using Spark"""

    prefix = 'fs.swift.service.' + name
    hconf = sc._jsc.hadoopConfiguration()
    hconf.set(prefix + '.auth.url', 'https://identity.open.softlayer.com'+'/v3/auth/tokens')
    hconf.set(prefix + '.auth.endpoint.prefix', 'endpoints')
    hconf.set(prefix + '.tenant', '5d9314bd83c8420a8be5abb3608958bc')
    hconf.set(prefix + '.username', '5be67aca97714d2781f516dfdd48a567')
    hconf.set(prefix + '.password', 'gtdtV_)A5^rYuev9')
    hconf.setInt(prefix + '.http.port', 8080)
    hconf.set(prefix + '.region', 'dallas')
    hconf.setBoolean(prefix + '.public', True)

# you can choose any name
name = 'keystone'
set_hadoop_config_with_credentials_f2ced6cdb4894c42945223d461f16d59(name)

train = sqlContext.read.format('com.databricks.spark.csv')    .options(header='true', inferschema='true')    .load("swift://notebooks." + name + "/train.csv")

test = sqlContext.read.format('com.databricks.spark.csv')    .options(header='true', inferschema='true')    .load("swift://notebooks." + name + "/test.csv")


# In[3]:

# Difference between Spark dataframes and Pandas; you always need to show() in Spark
train.head()


# In[4]:

train.show(10)


# In[5]:

# The describe() function in SparkSQL is not as useful as the one in Pandas.  I can't get summary stats for non-numeric fields 
train.describe().show()


# In[6]:

train.dtypes


# In[7]:

#Replace missing values in Age
##Compute the average age for passengers by sex and class
train_aggSexPclass = train.groupby(['Sex','Pclass']).agg({"Age":"mean"})
train_aggSexPclass = train_aggSexPclass.withColumn('Age', train_aggSexPclass['avg(Age)']).drop('avg(Age)')
train_aggSexPclass.show(10)


# In[8]:

# Merge the mean Age values by Sex and Pclass with the primary dataframe
# First, merge the mean values into a dataframe with the original records that had null values
train_merge = train.where(train.Age.isNull()).drop('Age').join(train_aggSexPclass, ['Sex','Pclass'], 'inner')
# Union with the original records 
train_ready = train.where(train.Age.isNotNull()).unionAll(train_merge)
train_ready.show(10)


# In[9]:

train_ready.dtypes


# In[10]:

# Train a RandomForest model.
from pyspark.ml.classification import RandomForestClassifier
estimator_rf = RandomForestClassifier(labelCol="indexed", numTrees=3, maxDepth=2, seed=42)


# In[12]:

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
vecAssembler = VectorAssembler(inputCols=['Age','Pclass','SibSp'], outputCol="features")
stringIndexer = StringIndexer(inputCol="Survived", outputCol="indexed")
pipeline = Pipeline(stages=[vecAssembler, stringIndexer, estimator_rf])
model = pipeline.fit(train_ready)


# In[ ]:

# Some testing of VectorAssembler to see if I was using it correctly
#trainingData = vecAssembler.transform(train_ready)
#trainingData.dtypes


# In[ ]:

###  SUCCESS!  Now I'd like to turn the whole process from start to finish into a pipeline in order to apply the transformations to the test set
###  however, it looks like join can't be pipelined



# In[13]:

# Merge the mean Age values by Sex and Pclass with the test dataframe
#test_merge = test.join(train_aggSexPclass, ['Sex','Pclass'], 'inner')
# Replace the missing values of Age with Age_Mean, then drop Age_Mean
#test_ready = test_merge.where(test_merge.Age.isNotNull()).unionAll(test_merge.where(test_merge.Age.isNull()).withColumn('Age',test_merge.Age_Mean)).drop('Age_Mean')

# Merge the mean Age values by Sex and Pclass with the primary dataframe
# First, merge the mean values into a dataframe with the original records that had null values
test_merge = test.where(test.Age.isNull()).drop('Age').join(train_aggSexPclass, ['Sex','Pclass'], 'inner')
# Union with the original records 
test_ready = test.where(test.Age.isNotNull()).unionAll(test_merge)
test_ready.show(10)


# In[14]:

prediction = model.transform(test_ready)
prediction.select('PassengerId','prediction').show()


# In[ ]:




# In[52]:

# Hey, pipelines can also be used just for transformations, though it's a little clunky
pipe2 = Pipeline(stages=[vecAssembler, stringIndexer])
model2 = pipe2.fit(train_ready)
pred2 = model2.transform(train_ready)
pred2.show(10)


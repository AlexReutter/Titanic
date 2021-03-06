﻿* Encoding: UTF-8.

PRESERVE.
SET DECIMAL DOT.

GET DATA  /TYPE=TXT
  /FILE="C:\Users\IBM_ADMIN\Downloads\Titanic\train.csv"
  /DELIMITERS=","
  /QUALIFIER='"'
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /DATATYPEMIN PERCENTAGE=95.0
  /VARIABLES=
  PassengerId AUTO
  Survived AUTO
  Pclass AUTO
  Name AUTO
  Sex AUTO
  Age AUTO
  SibSp AUTO
  Parch AUTO
  Ticket AUTO
  Fare AUTO
  Cabin AUTO
  Embarked AUTO
  /MAP.
RESTORE.
CACHE.
EXECUTE.
DATASET NAME DataSet1 WINDOW=FRONT.


DATASET ACTIVATE DataSet1.
CODEBOOK  PassengerId [s] Survived [n] Pclass [n] Name [n] Sex [n] Age [s] SibSp [n] Parch [n] 
    Ticket [n] Fare [s] Cabin [n] Embarked [n]
  /VARINFO LABEL TYPE FORMAT MEASURE MISSING
  /OPTIONS VARORDER=VARLIST SORT=ASCENDING MAXCATS=200
  /STATISTICS COUNT PERCENT MEAN STDDEV QUARTILES.




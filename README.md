# Titanic Disaster
In this project we work on a dataset from the titanic disaster that occured in 1912.We work on this data set to perform some exploratory data analysis, visualisation and a bit of logistic regression. Our Dataframe contains the following columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.

## Project Data Source : 
Kaggle

## Project Outcomes:
By the end of this project we would have sucessfully answered a few questions :

### *The Imports*
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Reading the Data
```
titanic_df = pd.read_csv("train.csv")
titanic_df.head()
```

### *Exploratory Data Analysis*
### Null Values in our dataset
```
titanic_df.isnull()
```

### Data types of all our columns
```
titanic_df.dtypes
```

### Basic Information about our dataset
```
titanic_df.info()
```

### Representation of null values for better understanding
```
sns.heatmap(titanic_df.isnull(),yticklabels= False,cbar=False)
```

### Survivors in our dataset
```
sns.countplot(x='Survived',data=titanic_df)
```

### How many males and female survivors and deceased are in our dataset
```
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=titanic_df,hue='Sex',palette='Set1',lw=2,ec='white',hatch='/')
```

### What is the relationship between deceased and survivors with Pclass in our dataset
```
sns.countplot(x='Survived',data=titanic_df,hue='Pclass',palette='Set2',hatch='/',lw=3,ec='black')
```

### Different Ages of people in our dataset
```
sns.displot(titanic_df['Age'].dropna(),aspect = 1.5,color='Yellow')
```

### People with Siblings or Spouse on board
```
sns.countplot(data=titanic_df,x='SibSp')
```

### Plot a distribution chart for fare
```
sns.displot(x='Fare',data=titanic_df,bins=40,aspect=2)
```

### *Data Cleaning*
### Inserting Values for Null Columns
```
def age_correction(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        if Pclass == 2:
            return 29
        if Pclass == 3:
            return 24
    else :
     return Age

titanic_df['Age'] = titanic_df[['Age','Pclass']].apply(age_correction,axis=1)
```

### Heatmap after data cleaning
```
sns.heatmap(titanic_df.isnull(),yticklabels= False,cbar=False)
```

### Dropping the Cabin column
```
titanic_df.drop('Cabin',inplace=True,axis=1)
titanic_df.head()
```

### Since we only have a few more NaN values we can drop them
```
titanic_df.dropna(inplace=True)
```

### Dataset After Cleaning
```
sns.heatmap(titanic_df.isnull(),yticklabels=False,cbar=False)
```

### Make a Dummy Column for the columns "Sex" and "Embarked"
```
titanic_df['sex'] = pd.get_dummies(titanic_df['Sex'],drop_first=True)
embark = pd.get_dummies(titanic_df['Embarked'])
titanic_df = pd.concat([titanic_df, embark],axis=1) 
titanic_df.head()
```

### Dropping the columns we aren't going to use
```
titanic_df.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)
titanic_df.head()
```

### *Machine Learning*

### Splitting the data to X and y
```
titanic_df.columns
X = titanic_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'sex', 'C', 'Q', 'S']]
y = titanic_df['Survived']
```

### splitting the data into train and test
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)
```
### Predictions and Evaluation
```
Import Logistic Regression
from sklearn.linear_model import LogisticRegression
```
### Create an instance of your logistic regression model
```
logmodel = LogisticRegression(max_iter=1000)
```
### Training the model
```
logmodel.fit(X_train, y_train)
```
### Call some predictions of out X_test dataset
```
X_test
predictions = logmodel.predict(X_test)
```
### Use Classification Report on the dataset
```
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
```

## Project Setup:
To clone this repository you need to have Python compiler installed on your system alongside pandas and seaborn libraries. I would rather suggest that you download jupyter notebook if you've not already.

To access all of the files I recommend you fork this repo and then clone it locally. Instructions on how to do this can be found here: https://help.github.com/en/github/getting-started-with-github/fork-a-repo

The other option is to click the green "clone or download" button and then click "Download ZIP". You then should extract all of the files to the location you want to edit your code.

Installing Jupyter Notebook: https://jupyter.readthedocs.io/en/latest/install.html<br>
Installing Pandas library: https://pandas.pydata.org/pandas-docs/stable/install.html










































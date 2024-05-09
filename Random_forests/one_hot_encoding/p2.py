import pandas as pd
from sklearn.model_selection import train_test_split

#Read the data
X = pd.read_csv('/Users/shlok_vaishnavi/Documents/ML/Random_forests/one_hot_encoding/train.csv', index_col= 'Id')
X_test = pd.read_csv('/Users/shlok_vaishnavi/Documents/ML/Random_forests/one_hot_encoding/train.csv', index_col='Id')
#Remove the rows with missing target
X.dropna(axis=0,subset=['SalePrice'],inplace=True)
# Sales price saved in variable Y
y=X.SalePrice
# now, drop saleprice from X 
X.drop(['SalePrice'],axis=1,inplace=True)
# To keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing,axis=1,inplace=True)
# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,
                                                      test_size=0.2,
                                                      random_state=0)
# Print the columns
print(X_train.head())

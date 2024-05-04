import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
import sys

def get_mae(max_leaf_nodes,train_X,value_x,train_y,val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X,train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y,preds_val)
    return mae
# print("current directory", os.getcwd())
# print("list of directories", os.listdir(os.getcwd()))
# iowa_file_path = '../ML/Validate_Model/train.csv'
iowa_file_path = '/Users/shlok_vaishnavi/Documents/ML/Decision_Tree_Regressor/Validate_Model/train.csv'
try:
    home_data = pd.read_csv(iowa_file_path)
except FileNotFoundError:
    print("File not found !")
    sys.exit()
    

# "fill mean value in all blank values
home_data['SalePrice'] = home_data['SalePrice'].fillna(home_data['SalePrice'].mean())
# Create a variable for 'home prices'
y = home_data['SalePrice']

# Create the list of features from data on which basis you want to do prediction
feature_names = [ 'LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

# Select data corresponding to features in feature_names in variable X
X = home_data[feature_names]

# So,now we have 2 variables X,y:
# y - having only Sale Price
# x - having only feature names on which prediction needs to be done
X.describe()
X.head()

# set the random state=1 
iowa_model = DecisionTreeRegressor(random_state=1)
# fit the model
iowa_model.fit(X,y)
# let's predict
predictions = iowa_model.predict(X)
#print prediction
# print(predictions)
# fill in and comment
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=1)
# fit iowa_model with training data
# set the random state=1  - it's already done above , 
# so will not do it again
# will jump to fit
iowa_model.fit(train_X,train_y)

val_predictions = iowa_model.predict(val_X)
print(val_predictions[0:5])
print(val_y.head().values)
# calculating mean absolute error
# val_mae = mean_absolute_error(val_y,val_predictions)
# print("mae:",val_mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {}
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
    scores[max_leaf_nodes] = my_mae
# store the best tree size value
best_tree_size = min(scores, key=scores.get)
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=0)
# concatenate all data
all_X = pd.concat([train_X,val_X])
all_y = pd.concat([train_y,val_y])

final_model.fit(all_X,all_y)
final_prediction = final_model.predict(val_X)
final_mae = mean_absolute_error(val_y,final_prediction)
print(final_mae)



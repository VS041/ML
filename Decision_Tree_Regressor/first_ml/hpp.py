import pandas as pd
from sklearn.tree import DecisionTreeRegressor

iowa_file_path = '../ML/train.csv'
home_data = pd.read_csv(iowa_file_path)
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
print(predictions)









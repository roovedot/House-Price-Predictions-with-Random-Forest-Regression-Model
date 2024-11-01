print("Importing libraries...")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

#load data
housingTrainingData = 'HousePricePredictionKaggle/train.csv'
print("loading data...", end=' ')
homeData = pd.read_csv(housingTrainingData)
print("done.")

print(homeData.head()) #preview of data
print(homeData.columns) #names of the columns

#features used for prediction of SalePrice
feature_names = [
    'OverallQual',  # Overall quality of the material and finish
    'GrLivArea',    # Above ground living area square feet
    'YearBuilt',    # Year the house was built
    'TotalBsmtSF',  # Total square feet of basement area
    '1stFlrSF',     # First-floor square feet
    'GarageCars',   # Garage size in car capacity
    'GarageArea',   # Total garage area
    'FullBath',     # Number of full bathrooms above ground
    'KitchenQual',  # Kitchen quality
    'Fireplaces',   # Number of fireplaces
    'LotArea',      # Lot size
    'Neighborhood', # Physical location within the city
    'ExterQual',    # Quality of exterior materials
    'BsmtQual',     # Basement quality
    'YearRemodAdd'  # Year of remodeling
]

#set SalePrice as the Target (y)
y = homeData.SalePrice

#set the features (X)
X = homeData[feature_names]

#One-Hot Encoding to turn categorical values into dense 'numpy' boolean arrays
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[['KitchenQual', 'Neighborhood', 'ExterQual', 'BsmtQual']])
# Convert the encoded features back into a DataFrame and concatenate with numerical features
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['KitchenQual', 'Neighborhood', 'ExterQual', 'BsmtQual']))
X = pd.concat([X.drop(columns=['KitchenQual', 'Neighborhood', 'ExterQual', 'BsmtQual']), X_encoded_df], axis=1)

#LABEL ENCODER OPTION (DISCARDED)
'''label_encoder = LabelEncoder() #turns categorical values into ints (labels), doesnt increase columns
# Apply label encoding to each categorical feature
for column in ['KitchenQual', 'Neighborhood', 'ExterQual', 'BsmtQual']:
    X.loc[:, column] = label_encoder.fit_transform(X[column])'''

#print(X.head()) #preview of features

#INITIAL MODEL TEST TO ADJUST FEATURES:
'''#Build Random Forest Model
model = RandomForestRegressor()
model.fit(train_X, train_y) #fits the model to the train Split
predictions = model.predict(test_X) #makes predictions on the test Split 
mae = mean_absolute_error(predictions, test_y)
print("MAE: ",mae)'''

#TESTS TO OPTIMISE PARAMETERS n_estimators & max_depth

# Split into validation and training data
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)

def get_mae(n, max_depth, train_X, val_X, train_y, val_y):
    maes = []
    for i in range(10):  # Run the model 10 times
        model = RandomForestRegressor(n_estimators=n, max_depth=max_depth, random_state=(i*3))
        model.fit(train_X, train_y)  # Train model 
        preds_val = model.predict(val_X)  # Make predictions
        mae = mean_absolute_error(val_y, preds_val)  # Calculate MAE
        maes.append(mae)
    
    return np.mean(maes)  # Return the mean MAE over 10 runs

for nEstimators in [525, 550, 575]:
    for maxDepth in [18, 19, 20]:  # Try different depths, including no limit
        my_mae = get_mae(nEstimators, maxDepth, train_X, test_X, train_y, test_y)
        print(f"n_estimators: {nEstimators}, max_depth: {maxDepth} \t Mean Absolute Error: {my_mae}")



#####################################################################################

#BUILD FINAL MODEL (OPTIMISED)
'''
#Create and Fit Model using All the data from the test.csv
fullModel = RandomForestRegressor(n_estimators=550, max_depth= 18)
fullModel.fit(X, y)

#Load test Data
housingTestData = 'HousePricePredictionKaggle/test.csv'
testData = pd.read_csv(housingTestData)

contest_X = testData[feature_names]

# Apply one-hot encoding to categorical features in the test data
contest_X_encoded = encoder.transform(contest_X[['KitchenQual', 'Neighborhood', 'ExterQual', 'BsmtQual']])
# Convert the encoded features back into a DataFrame and concatenate with numerical features
contest_X_encoded_df = pd.DataFrame(contest_X_encoded, columns=encoder.get_feature_names_out(['KitchenQual', 'Neighborhood', 'ExterQual', 'BsmtQual']))
contest_X = pd.concat([contest_X.drop(columns=['KitchenQual', 'Neighborhood', 'ExterQual', 'BsmtQual']), contest_X_encoded_df], axis=1)

contest_predictions = fullModel.predict(contest_X)

#Save predictions in Submission Format
output = pd.DataFrame({'Id': testData.Id,
                       'SalePrice': contest_predictions})
output.to_csv('optimisedSubmission.csv', index=False)
'''
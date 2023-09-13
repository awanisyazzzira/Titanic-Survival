import pandas as pd
import os #to assign a file path for the created csv file
from sklearn.ensemble import RandomForestClassifier 
from sklearn.impute import SimpleImputer #filling in missing values is better than dropping data points 

#Collecting train dataset
train_data = pd.read_csv(r'C:\Users\User\Desktop\talentlabs\titanic\train.csv') 
print("\nTrain dataset:\n")
print(train_data.head(10))

#Collecting test dataset
test_data = pd.read_csv(r'C:\Users\User\Desktop\talentlabs\titanic\test.csv')
print("\nTest dataset:\n")
print(test_data.head(10))

#Pre-processing for train_data set
train_data.drop(['PassengerId', 'Name', 'Cabin','Ticket'], axis=1, inplace=True) #insignificant factors
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True) #assigning binaries to categorical factors
imputer = SimpleImputer(strategy='median')
columns_to_impute = ['Age', 'Fare'] 
train_data[columns_to_impute] = imputer.fit_transform(train_data[columns_to_impute]) #filling in missing data points
print("\nPre-processed Train dataset:\n")
print(train_data.head(10))

#Pre-processing for test_data set
test_passenger_ids = test_data['PassengerId'] #extract PassengerIds before dropping them
test_data.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis=1, inplace=True) #insignificant factors
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True) #assigning binaries to categorical factor
imputer = SimpleImputer(strategy='median')
columns_to_impute = ['Age', 'Fare'] 
test_data[columns_to_impute] = imputer.fit_transform(test_data[columns_to_impute]) #filling in missing data points
print("Pre-processed Test dataset:\n")
print(test_data.head(10))

#Model building
X_train = train_data.drop('Survived', axis=1) #the features would be all data points except 'Survived'
y_train = train_data['Survived'] #'Survived' is the only target variable
model = RandomForestClassifier()
model.fit(X_train, y_train)

#Model testing 
x_test = test_data
y_pred = model.predict(x_test)
result_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': y_pred
})
print("\nPrediction results:\n")
print(result_df.head(10))

#Converting dataframe to csv file
folder_path = r'C:\Users\User\Desktop\talentlabs\titanic'
csv_file_path = os.path.join(folder_path, 'titanic_predictions_py.csv')
result_df.to_csv(csv_file_path, index=False)
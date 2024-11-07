import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor



#-----------------------Step1: Preprocessing--------------------------#
df = pd.read_csv('Breast_Cancer_dataset.csv')

#print the number of missing values
print(df.isnull().sum())

#handle missing values
i = 0
for val in df.isnull().sum():
  if(val > 0):
    if(df.iloc[:, i].dtype == int or df.iloc[:, i].dtype == float):
      mean = df.iloc[:, i].mean().round(1)
      df.iloc[:, i].fillna(mean, inplace = True)
    else:
      mode = df.iloc[:, i].mode()[0]
      df.iloc[:, i].fillna(mode, inplace = True)
  i = i + 1

print("Missing values after imputation:")
print(df.isnull().sum())

#drop duplicated rows
df.drop_duplicates(inplace=True)

#types of each column
#print(df.dtypes)

#rename T Stage
df.rename(columns={'T Stage ': 'T Stage'}, inplace=True)

numeric_columns = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
categorical_columns = ['Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status', 'Status']

#find unique values of each column
for col in df.columns:
    unique_values = df[col].unique()
    #print(f"'{col}': {unique_values}")

#convert values to 0/1 if there are only 2 kinds of values in a column
df['A Stage'] = df['A Stage'].replace({'Regional': 1, 'Distant': 0})
df['Estrogen Status'] = df['Estrogen Status'].replace({'Positive': 1, 'Negative': 0})
df['Progesterone Status'] = df['Progesterone Status'].replace({'Positive': 1, 'Negative': 0})
df['Status'] = df['Status'].replace({'Alive': 1, 'Dead': 0})

# One-hot encode the data using pandas get_dummies
df = pd.get_dummies(df)
df = df.astype(int)


#feature selection
# Labels are the values we want to predict
labels = np.array(df['Status'])

# Remove the labels from the features
df_fs = df.drop('Status', axis = 1)
feature_list = list(df_fs.columns)
df_fs = np.array(df_fs)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df_fs, labels, test_size = 0.25, random_state = 42)

# Train random forest and get feature importances
model = RandomForestClassifier()
model.fit(train_features, train_labels)
importances = model.feature_importances_

# Display feature importances
feature_importances = pd.Series(importances, index=np.array(feature_list))
print(feature_importances.sort_values(ascending=False))

threshold = 0.01
selected_features = feature_importances[feature_importances > threshold].index
selected_features = selected_features.append(pd.Index(['Status']))

df_original = df
df = df[selected_features]
#df.to_csv('preprocessed_dataset.csv', index=False) 
#------------- 

#standardization
scaler = StandardScaler()
df.loc[:, df.columns != 'Status'] = scaler.fit_transform(df.loc[:, df.columns != 'Status'])

print("\nFeature statistics after standardization:")
print(df.describe())

#detect outliers, LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = lof.fit_predict(df) 

#the number of outliers and inliners
value_counts = pd.Series(y_pred).value_counts()
print(value_counts)
df.to_csv('preprocessed_dataset.csv', index=False) 

#PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(df.loc[:, df.columns != 'Status']) 

n_components = X_pca.shape[1]
columns = [f'PC{i+1}' for i in range(n_components)]

pca_df = pd.DataFrame(data=X_pca, columns=columns)
pca_df['Status'] = df['Status']
pca_df.to_csv('pca_transformed_data.csv', index=False)

feature_names = df.drop('Status', axis=1).columns.tolist()

components_df = pd.DataFrame(
    data=pca.components_,
    columns=feature_names,
    index=[f'PC{i+1}' for i in range(n_components)]
)

explained_variance_ratio = pd.Series(
    pca.explained_variance_ratio_,
    index=[f'PC{i+1}' for i in range(n_components)]
)
components_df['Explained_Variance_Ratio'] = explained_variance_ratio
components_df['Cumulative_Variance_Ratio'] = explained_variance_ratio.cumsum()

components_df.to_csv('pca_components_info.csv')

print(f"PCA - Explained variance ratio by component:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f}")

#export preprocessed dataset
df.to_csv('preprocessed_dataset.csv', index=False) 

#-----------------------Step2: Modeling-------------------------------#

#Random Forest
# Labels are the values we want to predict
labels = np.array(df['Status'])

# Remove the labels from the features
df = df.drop('Status', axis = 1)
feature_list = list(df.columns)
df = np.array(df)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df, labels, test_size = 0.25, random_state = 42)

# The baseline predictions are the ages
baseline_preds = test_features[:, feature_list.index('Age')]

# Baseline errors, and display age baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Age baseline error: ', round(np.mean(baseline_errors), 2), 'degrees.')
#age baseline error: 53.31

# Instantiate model 
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)


# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(test_labels, predictions)
# Calculate and display accuracy
accuracy = 100 - (mae / np.mean(test_labels)) * 100
print('Accuracy of Random Forest:', round(accuracy, 2), '%.')

#models with different hyperparameters to try and boost performance
#Random Forest Hyperparameter Tuning
print("Random Forest Hyperparameter Tuning:")
rf_param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_param_grid,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_absolute_error'
)

print("\nStart Random Forest Grid Search")
rf_grid.fit(train_features, train_labels)

# Convert results to DataFrame
results = pd.DataFrame(rf_grid.cv_results_)

print("\nBest Random Forest Parameters:")
print(rf_grid.best_params_)

# Evaluate best Random Forest model
best_rf = rf_grid.best_estimator_
rf_predictions = best_rf.predict(test_features)
rf_mae = mean_absolute_error(test_labels, rf_predictions)
rf_accuracy = 100 - (rf_mae / np.mean(test_labels)) * 100
print(f"Random Forest Optimized Accuracy: {rf_accuracy:.2f}%")
#-------

#KNN
def most_common(lst):
    return max(set(lst), key=lst.count)
def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))
class KNeighborsClassifier:
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common, neighbors))
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy

# Test knn model across varying ks
accuracies = []
ks = range(1, 20)
highest_accuracy = 0
highest_k = 0
for k in ks:
    knn = KNeighborsClassifier(k=k)
    knn.fit(train_features, train_labels)
    accuracy = knn.evaluate(test_features, test_labels)
    accuracies.append(accuracy)
    if(highest_accuracy < accuracy):
       highest_accuracy = accuracy
       highest_k = k
highest_accuracy *= 100
print(f"Accuracy of KNN: {highest_accuracy:.2f}%, k: {highest_k}")

# Visualize accuracy vs. k
fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Performance of knn")
plt.show()

#Naive Bayes
# Step 1: Initialize the Gaussian Naive Bayes model
nb = GaussianNB(var_smoothing=1e-9)

# Step 2: Train the model on training data
nb.fit(train_features, train_labels)

# Step 3: Make predictions on the test set
predictions_nb = nb.predict(test_features)

# Step 4: Calculate the accuracy
acc = accuracy_score(test_labels, predictions_nb) * 100
nb_acc = acc
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))

#Decision Tree
dtc = DecisionTreeClassifier(
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None
)
dtc.fit(train_features, train_labels)

acc = dtc.score(test_features, test_labels)*100
dt_acc = acc
print("Accuracy of Decision Tree: {:.2f}%".format(acc))

#Gradient Boosting
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

gbm.fit(train_features, train_labels)

y_pred_gbm = gbm.predict(test_features)

mae_gbm = mean_absolute_error(test_labels, y_pred_gbm)

accuracy_gbm = 100 - (mae_gbm / np.mean(test_labels)) * 100
print(f"Accuracy of Gradient Boosting Regressor: {accuracy_gbm:.2f}%")

#Neural Network
print("\nNeural Network Model Results:")

nn_model = MLPRegressor(
    hidden_layer_sizes=(50, 50),
    activation='tanh', 
    alpha=0.01,
    learning_rate_init=0.01,
    max_iter=500,
    solver='adam',
    random_state=42
)

# Train the model with the training data
nn_model.fit(train_features, train_labels)

# Predicting using the test set
y_pred_nn = nn_model.predict(test_features)

# Calculate mean absolute error (MAE)
mae_nn = mean_absolute_error(test_labels, y_pred_nn)
accuracy_nn = 100 - (mae_nn / np.mean(test_labels)) * 100
print('Accuracy of Neural Network:', round(accuracy_nn, 2), '%.')

#Neural Network Hyperparameter Tuning
nn_param_grid = {
    'hidden_layer_sizes': [(50,), (100,)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [500],
    'alpha': [0.0001, 0.001]
}

nn = MLPRegressor(random_state=42)
nn_grid = GridSearchCV(estimator=nn, param_grid=nn_param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')

print("\nStart Neural Network Grid Search")
nn_grid.fit(train_features, train_labels)

# Convert results to DataFrame
results = pd.DataFrame(nn_grid.cv_results_)

print("\nBest Neural Network parameters:")
print(nn_grid.best_params_)

best_nn = nn_grid.best_estimator_
nn_predictions = best_nn.predict(test_features)
nn_mae = mean_absolute_error(test_labels, nn_predictions)
nn_accuracy = 100 - (nn_mae / np.mean(test_labels)) * 100
print(f"Neural Network Optimized Accuracy: {nn_accuracy:.2f}%")

# Create visualization of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(test_labels, y_pred_nn, alpha=0.5, color='red', label='Predicted')
plt.scatter(test_labels, test_labels, alpha=0.5, color='blue', label='Actual')

# Plot the perfect prediction line
plt.plot(test_labels, test_labels, color='green', linewidth=2)
plt.title('Neural Network Predicted Values vs. Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

# Create performance matrix for Random Forest
print("\nRandom Forest Performance Matrix:")
print("=" * 80)
rf_results = pd.DataFrame(rf_grid.cv_results_)
rf_results_list = []

for i in range(len(rf_results)):
    params = rf_results.iloc[i]['params']
    mae = -rf_results.iloc[i]['mean_test_score']
    accuracy = 100 - (mae / np.mean(train_labels)) * 100
    
    rf_results_list.append({
        'n_estimators': params['n_estimators'],
        'max_depth': str(params['max_depth']), #None
        'min_samples_split': params['min_samples_split'],
        'min_samples_leaf': params['min_samples_leaf'],
        'Accuracy': f"{accuracy:.2f}%"
    })

rf_matrix = pd.DataFrame(rf_results_list)
print(rf_matrix.sort_values('Accuracy', ascending=False))

print("\nNeural Network Performance Matrix:")
print("=" * 80)
nn_results = pd.DataFrame(nn_grid.cv_results_)
nn_results_list = []

for i in range(len(nn_results)):
    params = nn_results.iloc[i]['params']
    mae = -nn_results.iloc[i]['mean_test_score']
    accuracy = 100 - (mae / np.mean(train_labels)) * 100
    
    nn_results_list.append({
        'hidden_layers': str(params['hidden_layer_sizes']),
        'activation': params['activation'],
        'learning_rate': params['learning_rate_init'],
        'alpha': params['alpha'],
        'max_iter': params['max_iter'],
        'Accuracy': f"{accuracy:.2f}%"
        
    })

nn_matrix = pd.DataFrame(nn_results_list)
print(nn_matrix.sort_values('Accuracy', ascending=False))

# Print best parameters and accuracies
print("\nBest Models Summary:")
print("=" * 80)
print("\nRandom Forest Best Parameters:")
print(rf_grid.best_params_)
rf_predictions = rf_grid.best_estimator_.predict(test_features)
rf_mae = mean_absolute_error(test_labels, rf_predictions)
rf_best_accuracy = 100 - (rf_mae / np.mean(test_labels)) * 100
print(f"Best Accuracy: {rf_best_accuracy:.2f}%")

print("\nNeural Network Best Parameters:")
print(nn_grid.best_params_)
nn_predictions = nn_grid.best_estimator_.predict(test_features)
nn_mae = mean_absolute_error(test_labels, nn_predictions)
nn_best_accuracy = 100 - (nn_mae / np.mean(test_labels)) * 100
print(f"Best Accuracy: {nn_best_accuracy:.2f}%")

# Create results table
results_data = []
rf_result = {
    'Model': 'Random Forest',
    'Best Parameters': str(rf_grid.best_params_),
    'Accuracy (%)': f"{rf_accuracy:.2f}"
}
results_data.append(rf_result)

nn_result = {
    'Model': 'Neural Network',
    'Best Parameters': str(nn_grid.best_params_),
    'Accuracy (%)': f"{nn_accuracy:.2f}"
}
results_data.append(nn_result)

# Create DataFrame and save to CSV
results_df = pd.DataFrame(results_data)
results_df.to_csv('model_comparison_results.csv', index=False)

# Display the table
print("\nModel Comparison Results:")
print("=" * 100)
print(results_df.to_string(index=False))

model_results = {
    'Random Forest': rf_result['Accuracy (%)'], #rf_accuracy,
    'KNN': highest_accuracy,
    'Naive Bayes': nb_acc,
    'Decision Tree': dt_acc,
    'Gradient Boosting': accuracy_gbm,
    'Neural Network': nn_result['Accuracy (%)'] #accuracy_nn
}

results_df = pd.DataFrame(list(model_results.items()), columns=['Model', 'Accuracy (%)'])

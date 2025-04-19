from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def random_forest_model(df, features, target, test_float, rand_int, n_trees):
    """ using the random forest model on a data frame with features & target, return MSE and prediction data"""
    # features used to generate target
    X = df[features]
    y = df[target]

    # split data into training and testing sets
    # X is the features of the data set
    # y is the target, what we want to predict
    # test size a float between 0.0 and 1.0, that selects the % of data to be tested on 
        # -- i think this implies train size = 1.0 - test size
    # random_state assigns an int to the instance with which this particular data set was trained
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_float, random_state=rand_int)

    # instantiate the random forest regress
    # n_estimators is the number of trees in the random forest
    rf_model = RandomForestRegressor(n_estimators=n_trees, random_state= rand_int)

    # train the model
    rf_model.fit(X_train, y_train)

    # make predictions on the test set
    predictions = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return y, predictions, mse

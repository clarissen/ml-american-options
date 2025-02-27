import model
import data

df, features, target = data.get_data("calls")

mse, predictions = model.random_forest_model(df, features, target, 0.3, 1, 100)

print("synthetic data set", df)
print("prediction data_set", predictions)
print("mean squared error", mse)


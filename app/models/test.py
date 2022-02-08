import pandas as pd
from sklearn.metrics import mean_squared_error as mse, r2_score as r2
import dill
dill._dill._reverse_typemap['ClassType'] = type
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")
with open('pipeline.dill', 'rb') as in_strm:
    pipeline = dill.load(in_strm)
predictions = pipeline.predict(X_test)
pd.DataFrame({'preds': predictions}).to_csv("test_predictions.csv", index=None)
print(r2(y_test, predictions))
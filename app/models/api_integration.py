import pandas as pd
from sklearn.metrics import roc_auc_score
from urllib import request, parse
from sklearn.metrics import mean_squared_error as mse, r2_score as r2
import urllib.request
import json
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")
def get_prediction(square, rooms, social_3, social_1):
    body = {'Square': square,
                            'Rooms': rooms,
                            'Social_3': social_3,
                            'Social_1': social_1
           }

    myurl = "http://0.0.0.0:8180/predict"
    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')
    req.add_header('Content-Length', len(jsondataasbytes))
    response = urllib.request.urlopen(req, jsondataasbytes)
    return json.loads(response.read())['predictions']
print(get_prediction(101.2, 3.0, 1, 43))

# predictions = X_test[['Square', 'Rooms', 'Social_3', 'Social_1']].iloc[:500].apply(lambda x: get_prediction(x[0],
#                                                                                                x[1],
#                                                                                                x[2],
#                                                                                                x[3]), 1)
# r2(y_test.iloc[:500], predictions)


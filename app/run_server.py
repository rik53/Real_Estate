# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py

# import the necessary packages
import dill
import pandas as pd
import os
dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None
modelpath = "/home/rik/PycharmProjects/project_Real_Estate/Real_Estate/app/models/pipeline.dill"
# modelpath = "/app/app/models/pipeline.dill"

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
	# load the pre-trained model
	# global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)
	return model
model = load_model(modelpath)
print(model)




@app.route("/", methods=["GET"])
def general():
	return """Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST"""

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
	dt = strftime("[%Y-%b-%d %H:%M:%S]")
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":

		square, rooms, social_3, social_1 = 0.0, 0.0, 0, 0
		request_json = flask.request.get_json()
		if request_json["Square"]:
			square = request_json['Square']
		if request_json["Rooms"]:
			rooms = request_json['Rooms']

		if request_json["Social_3"]:
			social_3 = request_json['Social_3']
		if request_json["Social_1"]:
			social_1 = request_json['Social_1']
		logger.info(f'{dt} Data: square={square}, rooms={rooms}, social_3={social_3}, social_1={social_1}')
		if isinstance(request_json["Square"], (int, float)) and isinstance(request_json["Rooms"], (int, float))\
		and isinstance(request_json["Social_3"], (int, float)) and isinstance(request_json["Social_3"], (int, float)):
		    try:
			    preds = model.predict(pd.DataFrame({"Square": [square],
												  "Rooms": [rooms],
												  "Social_3": [social_3],
												  "Social_1": [social_1]
													  }))
		    except AttributeError as e:
			    logger.warning(f'{dt} Exception: {str(e)}')
			    data['predictions'] = str(e)
			    data['success'] = False
			    return flask.jsonify(data)
		else:
			preds = ['Ошибка типа данных']

		data["predictions"] = preds[0]
		# indicate that the request was a success
		data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	port = int(os.environ.get('PORT', 8180))
	app.run(host='0.0.0.0', debug=True, port=port)
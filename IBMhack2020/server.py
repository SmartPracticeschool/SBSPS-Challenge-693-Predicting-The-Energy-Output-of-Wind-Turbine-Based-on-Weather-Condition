from flask import Flask, redirect, url_for, jsonify, request, render_template
import os

EPOCHS = 36
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUMBER_OF_STEPS_IN = 144
NUMBER_OF_STEPS_OUT = 72
CLIP_NORM = 0.5
FEATURES = 6

app = Flask(__name__)

def load_model():
    from tensorflow.keras.models import model_from_json
    
    with open('model_architecture.json', 'r') as f:
		model = model_from_json(f.read())
  
    model.load_weights('weights.h5')
    return model

# def fetch_and_normalize_data(number):

# 	import pandas as pd
# 	from sklearn.preprocessing import MinMaxScaler
# 	from sklearn.metrics import mean_squared_error
# 	# n_steps_in = 144
# 	# n_steps_out = 72
# 	# n_features = 6

# 	df = pd.read_csv('model_data.csv')

# 	sc = MinMaxScaler()
# 	scaled_data = sc.fit_transform(df.values[(number - NUMBER_OF_STEPS_IN) : (number + NUMBER_OF_STEPS_OUT), :])

# 	scaled_x = scaled_data[:NUMBER_OF_STEPS_IN,  :-1]
# 	scaled_y = scaled_data[-NUMBER_OF_STEPS_OUT: , -1] 

# 	return scaled_x, scaled_y

# def fetch_last_n_stepsin(record):
    
# 	import pandas as pd
# 	from sklearn.preprocessing import MinMaxScaler
# 	import numpy as np

# 	# n_steps_in = 144
# 	# n_steps_out = 72
# 	# n_features = 6

# 	df = pd.read_csv('model_data.csv')

# 	sc_x = MinMaxScaler()
# 	data = df.values[-(NUMBER_OF_STEPS_IN - 1):, :-1]
# 	record = np.array(record)
# 	X = np.concatenate( (data, record.reshape(1, record.shape[0]) ), axis = 0)

# 	return sc_x.fit_transform(X)

def predict_label(x, y = None):
    
    model = load_model()
    
    yhat = model.predict(x.reshape(1, NUMBER_OF_STEPS_IN, FEATURES))
    y_hat_reshaped = yhat.reshape(yhat.shape[1])
    
    return  y_hat_reshaped, y

def create_plot(prediction, true = None):
    
	import plotly
	import plotly.graph_objects as go
	import json

	fig = go.Figure()

	fig.add_trace(go.Scatter(x=list(range(72)), y=prediction, mode='lines', name='prediction', opacity=0.8, marker_color='orange'))

	if true is not None:
		fig.add_trace(go.Scatter(x=list(range(72)), y=true, mode='lines', name='True data', opacity=0.8, marker_color='blue'))

	fig.show()
	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	return graphJSON

@app.route('/',methods = ['GET']) 
def index():
	data = {
			'epochs':EPOCHS, 'batch_size':BATCH_SIZE, 'lr':LEARNING_RATE,
			'n_steps_in':NUMBER_OF_STEPS_IN,'n_steps_out':NUMBER_OF_STEPS_OUT,
			'clip_norm':CLIP_NORM
			}
	return render_template('about.html', data = data)

# @app.route('/evaluate',methods = ['GET', 'POST']) 
# def evaluate(): 

# 	if request.method == 'GET':
# 		return render_template('evaluate.html')
# 	else:
# 		number = int(request.form['number'])
# 		print("number", number)
# 		scaled_x, scaled_y = fetch_and_normalize_data(number)
# 		prediction, true = predict_label(scaled_x, scaled_y)

# 		plot = create_plot(prediction, true)
# 		return render_template('graph.html', plot = plot)


@app.route('/predict', methods = ['GET', 'POST'])
def predict():
	'''
		Args: no args:
		If request if GET:
		   returns the template for individual prediction
		else:
			takes all inputs from user and then passes it as argument to 
			fetch_last_n_stepsin() as parameter and then the values are predicted 
			using predict_label() function and plotted using create_plot().
	'''
	if request.method == 'GET':
		return render_template('prediction.html')
	else:
		wind_speed = float(request.form['wind_speed'])
		tpc = float(request.form['tpc'])
		wind_d = float(request.form['wind_d'])
		wind_gust = float(request.form['wind_gust'])
		dew_point = float(request.form['dew_point'])
		wind_chill = float(request.form['wind_chill'])

		record = [wind_speed, tpc, wind_d, wind_gust, dew_point, wind_chill]
		X = fetch_last_n_stepsin(record)
		prediction, y = predict_label(X)

		plot = create_plot(prediction)
		return render_template('graph.html', plot = plot)

if __name__ == '__main__':
	app.run(debug = False)
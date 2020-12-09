import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = []
    for x in request.form.values():
        if(x=='M'):
            x=1
        elif(x=='F'):
            x=0
        elif(x=='S'):
            x=1
        elif(x=='C'):
            x=0
        elif(x=='A'):
            x=0
        elif(x=='CM'):
            x=1
        elif(x=='SC'):
            x=2
        elif(x=='T'):
            x=2
        elif(x=='M'):
            x=0
        elif(x=='O'):
            x=1
        elif(x=='N'):
            x=0
        elif(x=='Y'):
            x=1
        elif(x=='H'):
            x=1
        elif(x=='F'):
            x=0
        
        int_features.append(int(x))
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if(output==1):
        return render_template('index.html', prediction_text='Congratulations! you will be placed ')
    if(output==0):
        return render_template('index.html', prediction_text='keep up the hardwork! Placement chances are low ')
                               

        

    

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
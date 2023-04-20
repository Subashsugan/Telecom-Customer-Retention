import flask
from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn

app = Flask(__name__)

model = pickle.load(open('rfcv.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/getdata', methods=['POST'])
def pred():
    gender = request.form['Gender']
    print(gender)
    seniorcitizen = request.form['Senior_Citizen']
    print(seniorcitizen)
    partner = request.form['Partner']
    print(partner)
    dependents = request.form['Dependents']
    print(dependents)
    tenure = request.form['Tenure']
    print(tenure)
    phoneservice = request.form['Phone_Service']
    print(phoneservice)
    multiplelines = request.form['Multiple_Lines']
    print(multiplelines)
    internetservice = request.form['Internet_Service']
    print(internetservice)
    onlinesecurity = request.form['Online_Security']
    print(onlinesecurity)
    onlinebackup = request.form['Online_Backup']
    print(onlinebackup)
    deviceprotection = request.form['Device_Protection']
    print(deviceprotection)
    techsupport = request.form['Tech_Support']
    print(techsupport)
    streamingtv = request.form['Streaming_TV']
    print(streamingtv)
    streamingmovies = request.form['Streaming_Movies']
    print(streamingmovies)
    contract = request.form['Contract']
    print(contract)
    paperlessbilling = request.form['Paper_less_Billing']
    print(paperlessbilling)
    paymentmethod = request.form['Payment_Method']
    print(paymentmethod)
    month = request.form['Monthly_Charges']
    print(month)
    year = request.form['Yearly_Charge']
    print(year)
    admin = request.form['Admin_Tickets']
    print(admin)
    tech = request.form['Tech_Tickets']
    print(tech)


    inp_features = [[int(gender),int(seniorcitizen), int(partner), int(dependents), int(tenure), int(phoneservice),
                     int(multiplelines),int(internetservice),
                     int(onlinesecurity),
                     int(onlinebackup), int(deviceprotection),int(techsupport), int(streamingtv), int(streamingmovies), int(contract),
                     int(paperlessbilling), int(paymentmethod),np.log(int(month)),int(year),
                     int(admin),int(tech)]]
    print(inp_features)
    prediction = model.predict(inp_features)
    print(type(prediction))
    t = prediction[0]
    print(t)
    if t > 0.5:
        prediction_text = 'Customer will retain'
    else:
        prediction_text = 'Customer will not retain'
    print(prediction_text)
    return render_template('prediction.html', prediction_results=prediction_text)


if __name__ == "__main__":
    app.run()

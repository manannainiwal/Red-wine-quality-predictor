from flask import Flask, request, url_for, redirect, render_template
import pickle
import pandas as pd

app = Flask(__name__)

model=pickle.load(open('wine_quality_model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    if (request.method == 'POST'):
        Volatile_Acidity=request.form['Volatile Acidity']
        Residual_Sugar=request.form['sugar']
        Chlorides=request.form['clorides']
        Total_Sulfur_Dioxide=request.form['Total sulfur dioxide']
        Density=request.form['Density']
        Sulphates=request.form['Sulphates']
        Alcohol=request.form['alcohol']
    try:
        data=[[float(Volatile_Acidity),float(Residual_Sugar),float(Chlorides),float(Total_Sulfur_Dioxide),float(Density),float(Sulphates),float(Alcohol)]]
        testdfarr=[['volatile acidity', 0.12, 1.01, data[0][0]], ['residual sugar', 1.2, 3.65, data[0][1]], ['chlorides', 0.038, 0.121, data[0][2]], ['total sulfur dioxide', 6, 114, data[0][3]], ['density', 0.9924, 1.0004, data[0][4]], ['sulphates', 0.33, 0.94, data[0][5]], ['alcohol', 8.7, 13.4, data[0][6]]]
        testdf=pd.DataFrame(testdfarr, columns=['feature name','min','max','input'])
        testdf['normalised_input']=(testdf['input']-testdf['min'])/(testdf['max']-testdf['min'])
        nordata=testdf['normalised_input'].to_list()
        inputdf = pd.DataFrame([nordata], columns=['volatile acidity','residual sugar','chlorides','total sulfur dioxide','density','sulphates','alcohol'])
        prediction=model.predict(inputdf)
        output=prediction[0]
        
        if output==0:
            return render_template('index.html',pred='Result : Your wine quality is Poor. \n',bhai="mat piyo")
        else:
            return render_template('index.html',pred='Result : Your wine quality is Great. \n',bhai="pilo bhai")
    except:
        return render_template('index.html',pred='Make sure you fill all the values correctly. \n',bhai="")


if __name__ == '__main__':
    app.run(debug=True)

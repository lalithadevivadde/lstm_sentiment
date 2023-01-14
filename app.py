from flask import Flask, render_template, request
import requests
import numpy as np
import pandas as pd
from flask_cors import cross_origin, CORS
import cloudpickle
from preprocessing.text_preprocessing import TextPreprocessor
from tensorflow.keras import models
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["SM_FRAMEWORK"] = "tf.keras"



app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)


with open(os.path.join(".", "models", "preprocessors.bin"), "rb") as f:
    tokenizer, maxlen, padding, truncating = cloudpickle.load(f)

model = models.load_model(os.path.join(".", "models", "review_sentiment_model.h5"), compile=False)





@app.route("/")
@cross_origin()
def index():
    """
    displays the index.html page
    """
    return render_template("index.html")






@app.route("/", methods=["POST"])
@cross_origin()
def form_prediction():
    """
    Returns the API response based on the inputs filled in the form.
    """

    def prediction(txt):
        """
        Processes the API request and returns a prediction
        """
        try:
            txt_ser = pd.Series(txt)  # converting api data dict to df
            tp = TextPreprocessor()
            preprocessed_str = tp.preprocess(txt_ser, dataset="test")
            txt_seq = tokenizer.texts_to_sequences(preprocessed_str)
            txt_seq = pad_sequences(txt_seq, maxlen=maxlen, padding=padding, truncating=truncating)
            pred = model.predict(txt_seq)[0]
            confidence_level = [np.round(p * 100, 2) for p in pred]
            response = {"result": ["negative", "positive"], "confidence": confidence_level}

        except Exception as e:
            # executes in case of any exception
            response = {"result": str(e), "confidence": "-"}
        return response

    try:
        # assigning the inputs from the form to respective variables.
        text = request.form["text_input"]
        # input json to the API in the required format
        error = ""
        response = prediction(text)
        response_json = response
        predicted_label = response_json.get("result")
        confidence_level = np.array(response_json.get("confidence"))
        if confidence_level[0] != "-":
            idx_max = np.argmax(confidence_level)
            prediction = predicted_label[idx_max]
            confidence = confidence_level[idx_max]
            font_col = "green"
            if prediction != "positive":
                font_col = "red"
            result = f'<h3 style="color: {font_col};">{prediction.title()} ({confidence}%)</h2>'
        else:
            raise Exception(predicted_label)
    except Exception as e:
        result = f'Error: Details at the bottom'
        error = e
    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)

from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
import signal
import logging
import requests
import numpy as np

flask_logger = logging.getLogger()
# create a formatter object
logFormatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

# add file handler to the root logger
fileHandler = logging.FileHandler("flasklogs.log")
fileHandler.setFormatter(logFormatter)
flask_logger.addHandler(fileHandler)

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/stopServer', methods=['GET'])
@cross_origin()
def stopServer():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({ "success": True, "message": "Server is shutting down..." })


def decodeData(data) -> np.array:
    decodedFeatures = {}
    for key, val in data.items():
        if key in ['Age','RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
            val = [float(val[0])]
        elif key in ["CryoSleep", "VIP"]:
            val = [val[0]]
        decodedFeatures[key] = val
    flask_logger.info("data was decoded")
    return decodedFeatures

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        data = request.json['data']
        decodedFeatures = decodeData(data)
        # print(decodedFeatures)
        prediction = requests.post("http://torchserve-mar:8080/predictions/spaceship", data=decodedFeatures)
        print(prediction)
        prediction = prediction.json()
        flask_logger.info("Successfully result is responded to UI")
        return jsonify(prediction)
    except Exception as e:
        # print(e)
        flask_logger.info(f"Exception is raised {e}")
        return jsonify([{'transported': "Error processing"}])
    
if __name__ == "__main__":

    # app.run(host='127.0.0.1', port=5000, debug=True) #local host
    app.run(host='0.0.0.0', port=8085) #local host
    # # app.run(host='0.0.0.0', port=8080) #for AWS

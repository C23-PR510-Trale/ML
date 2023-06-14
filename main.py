from helper import make_prediction
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/", methods=["POST"])
def index():
    try:
        data = request.get_json()
        if data:
            category = data.get("category")
            budget = data.get("budget")
            num_of_recom = data.get("num_of_recom")

            prediction = make_prediction(category, budget, num_of_recom)

            return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
app.run()

from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        text = request.form["jobtext"]
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]
        result = "🚨 SCAM JOB" if prediction == "scam" else "✅ LEGIT JOB"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

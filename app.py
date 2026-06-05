from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    try:
        sentence = request.form.get('sentence')

        print("Input sentence:", sentence, flush=True)

        predict_pipeline = PredictPipeline()
        pred, pred_emotion_class = predict_pipeline.initiate_predictor(None, [sentence])

        print("Raw prediction:", pred, flush=True)
        print("Predicted emotion:", pred_emotion_class, flush=True)

        return render_template(
            'home.html',
            prediction=pred_emotion_class,
            sentence=sentence
        )

    except Exception as e:
        print("Error:", e, flush=True)

        return render_template(
            'home.html',
            prediction=f"Error: {e}",
            sentence=sentence
        )


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
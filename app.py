from flask import Flask, render_template_string, request
import joblib
from transformers import pipeline

# ====== Load classical model ======
tfidf = joblib.load("tfidf.joblib")
clf = joblib.load("logreg.joblib")

# ====== Load transformer models ======
# Sentiment classifier (contextual model)
trf_model = pipeline("sentiment-analysis")

# Generative model – creates a natural language explanation
gen_model = pipeline("text-generation", model="distilgpt2")

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Smartwatch Sentiment Analyzer</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: radial-gradient(circle at top, #1f2933, #020617);
            color: #fff;
            text-align: center;
            padding: 40px 10px;
        }
        h1 {
            font-size: 42px;
            margin-bottom: 6px;
        }
        h2 {
            margin-top: 0;
        }
        p {
            font-size: 16px;
            opacity: 0.9;
        }
        .box {
            background: rgba(15,23,42,0.85);
            padding: 30px;
            width: 75%;
            margin: 25px auto;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.4);
            box-shadow: 0 18px 40px rgba(15,23,42,0.8);
        }
        textarea {
            width: 92%;
            height: 140px;
            padding: 14px;
            margin-top: 18px;
            border-radius: 14px;
            border: 1px solid #0f172a;
            outline: none;
            font-size: 15px;
            resize: vertical;
        }
        textarea:focus {
            box-shadow: 0 0 0 2px #22d3ee;
        }
        button {
            margin-top: 18px;
            padding: 11px 26px;
            font-size: 16px;
            border-radius: 999px;
            border: none;
            background: linear-gradient(90deg,#22c55e,#22d3ee);
            cursor: pointer;
            font-weight: 600;
            color: #020617;
        }
        button:hover {
            filter: brightness(1.07);
        }
        .card {
            margin: 25px auto;
            padding: 22px;
            border-radius: 16px;
            background: rgba(15,23,42,0.9);
            width: 75%;
            border: 1px solid rgba(148,163,184,0.5);
        }
        .pos {
            color: #22c55e;
            font-weight: bold;
            font-size: 19px;
        }
        .neg {
            color: #fb7185;
            font-weight: bold;
            font-size: 19px;
        }
        .tag {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            background: rgba(15,23,42,0.9);
            border: 1px solid rgba(148,163,184,0.7);
            margin: 0 4px;
        }
        .ai {
            color: #22d3ee;
        }
    </style>
</head>
<body>

    <h1>🕒 Smartwatch Sentiment Analyzer</h1>
    <p>
        <span class="tag">Domain: Wearable Tech</span>
        <span class="tag ai">Gen-AI + Transformer</span>
    </p>

    <form method="post">
        <div class="box">
            <h2>Enter a smartwatch review</h2>
            <textarea name="review" placeholder="Example: The battery life of this smartwatch is amazing and the screen is bright!">{{review}}</textarea><br>
            <button type="submit">Analyze with AI</button>
        </div>
    </form>

    {% if ml_pred %}
    <div class="card">
        <h2>🔍 Model Outputs</h2>
        <p>
            <b>Classical ML (TF-IDF + Logistic Regression):</b>
            <span class="{{ 'pos' if ml_pred=='positive' else 'neg' }}">{{ ml_pred }}</span>
        </p>
        <p>
            <b>Transformer (Contextual Sentiment Model):</b>
            <span class="{{ 'pos' if trf_pred=='positive' else 'neg' }}">{{ trf_pred }}</span>
        </p>

        {% if ai_expl %}
        <hr style="border: none; border-top: 1px solid rgba(148,163,184,0.5); margin: 18px 0;">
        <h2>🤖 Gen-AI Explanation</h2>
        <p>{{ ai_expl }}</p>
        {% endif %}

        <p style="margin-top:16px; font-size:14px; opacity:0.85;">
            Classical ML looks only at word frequencies (bag-of-words).<br>
            The transformer and generative model understand the context of the sentence,
            which is why they perform much better on real smartwatch reviews.
        </p>
    </div>
    {% endif %}

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    review = ""
    ml_pred = None
    trf_pred = None
    ai_expl = None

    if request.method == "POST":
        review = request.form.get("review", "")
        if review.strip():
            # ----- Classical prediction -----
            vec = tfidf.transform([review])
            ml_pred = clf.predict(vec)[0]

            # ----- Transformer sentiment prediction -----
            out = trf_model(review, truncation=True)[0]
            trf_pred = "positive" if out["label"].upper().startswith("POS") else "negative"

            # ----- Generative AI explanation -----
            prompt = (
                f"In one short sentence, explain why the following smartwatch review is "
                f"{trf_pred}: \"{review}\""
            )
            gen = gen_model(prompt, max_length=60, num_return_sequences=1)[0]["generated_text"]
            # remove the prompt part, keep only explanation after colon if present
            ai_expl = gen.split(":")[-1].strip()

    return render_template_string(
        HTML,
        review=review,
        ml_pred=ml_pred,
        trf_pred=trf_pred,
        ai_expl=ai_expl
    )

if __name__ == "__main__":
    app.run(debug=True)

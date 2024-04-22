from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = FastAPI()

model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

label_dict = {
    0: 'Style&Fashion',
    1: 'Sensitive Topics',
    2: 'Religion&Spirituality',
    3: 'Science',
    4: 'Viral Articles',
    5: 'Beauty',
    6: 'Tech&Computing',
    7: 'Healthy Living',
    8: 'Sports',
    9: 'Culture',
    10: 'Business&Finance',
    11: 'Education',
    12: 'Astrology',
    13: 'Family&Relationships',
    14: 'Politics',
    15: 'Food&Drink',
    16: 'Pop Culture',
    17: 'Automotive',
    18: 'Attractions',
    19: 'Home&Garden',
    20: 'Travel'
}

@app.post("/predict/rf")
async def predict(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if 'url' not in data:
        raise HTTPException(status_code=400, detail="Missing url in request")

    try:
        vectorized_text = vectorizer.transform([data['url']])
        prediction = model.predict(vectorized_text)

        predicted_label_name = label_dict.get(prediction[0])
        return JSONResponse(content={'prediction with Random Forest for URL provided': predicted_label_name})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

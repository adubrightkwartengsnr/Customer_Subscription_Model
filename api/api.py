
import os
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import joblib
import uvicorn
import numpy as np
import __main__

def safe_log1p(X):
    X = np.where(X <= 0, 1e-9, X)
    return np.log1p(X)

__main__.safe_log1p = safe_log1p




# create an instance of FASTAPI
app = FastAPI(
        title="Customer Subscription Prediction API",
        description= "Predicts whether a customer subscribes to a deposit term or not",
        version="1.0.0",
        contact={
            "name": "API Support",
            "email": "adubrightkwarrteng11@gmail.com"
        }
)


def load_ml_components():
    model_path = os.path.join(os.path.dirname(__file__), "models", "model_components.joblib")
    with open(model_path, "rb") as file:
        model_components = joblib.load(file)
    return model_components


# call the model components
model_components = load_ml_components()

# get api status
@app.get("/")
def get_status():
    return {"status": "API is running"}


# # Create CustomerSubscriptionFeatures class from pydantic BaseModel
class CustomerSubscriptionFeatures(BaseModel):
        age: int
        job: str            
        marital: str         
        education: str       
        default: str         
        balance: int      
        housing: str         
        loan: str           
        contact: str         
        day: int            
        month: str          
        duration: int     
        campaign: int       
        pdays: int         
        previous: int      
        poutcome:str  

# # Create Endpoint for the XGB Classifier Pipeline
@app.post("/predict_subscription/xgb_classifier")
async def subcription_prediction(data:CustomerSubscriptionFeatures):
    try:
        # create dataframe from sepssis data
        df = pd.DataFrame([data.model_dump()])
        # call the xgb_classifier_pipeline and encoder from the ml model
        xgb_classifier_pipeline = model_components["xgb_classifier"]
        encoder = model_components["encoder"]
        # make prediction
        prediction = xgb_classifier_pipeline.predict(df)
        prediction = int(prediction[0])
        decoded_prediction = encoder.inverse_transform([prediction])[0]
        prediction_proba = xgb_classifier_pipeline.predict_proba(df)[0].tolist()
        response = {"model_used":"XGB Classifier",
                    "prediction": decoded_prediction,
                    "prediction_probability":
                    {"No":round(prediction_proba[0],2),
                     "Yes":round(prediction_proba[1],2)}
                    }
        
        return response
    except Exception as e:
        return {"error": str(e)}

# # Create Endpoint for the Gradient Boost Model
@app.post("/predict_subscription/gradient_boost")
async def subcription_prediction(data:CustomerSubscriptionFeatures):
    try:
        # create dataframe from sepssis data
        df = pd.DataFrame([data.model_dump()])
        # call the random_forest_pipeline and encoder from the ml model
        gradient_boost_pipeline = model_components["gradient_boost"]
        encoder = model_components["encoder"]
        # make prediction
        prediction = gradient_boost_pipeline.predict(df)
        prediction = int(prediction[0])
        decoded_prediction = encoder.inverse_transform([prediction])[0]
        prediction_proba = gradient_boost_pipeline.predict_proba(df)[0].tolist()

        response =  {"model_used":"Gradient Boosting Classifier",
                    "prediction": decoded_prediction,
                    "prediction_probability":
                    {"No":round(prediction_proba[0],2),
                     "Yes":round(prediction_proba[1],2)}
                    }
        
        return response
    except Exception as e:
        return {"error": str(e)}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
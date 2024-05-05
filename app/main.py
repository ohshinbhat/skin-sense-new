from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
import os
import shutil  # Import shutil to resolve "shutil is not defined" error

app = FastAPI()

# Get the absolute path to the model file
model_path = os.path.abspath("app/model/best.pt")

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")

# Load the YOLO model
model = YOLO(model_path)

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    print('here first')
    try:
        with open("temp.jpg", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        results = model("temp.jpg")
        names_dict = results[0].names
        print('here')
        probs = results[0].probs.data.tolist()
        predictions = names_dict[np.argmax(probs)]
        print(predictions)

        return {"predictions": predictions}
    

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
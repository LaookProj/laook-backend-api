import os
import random
from typing import List, Union, Optional
import numpy as np
import pandas as pd

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Pydantic v2 compatibility
try:
    from pydantic import model_validator
    PYDANTIC_V2 = True
except ImportError:
    PYDANTIC_V2 = False

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecognizeIngredientsResponse(BaseModel):
    ingredients: List[str]

class SuggestMenusRequest(BaseModel):
    ingredients: List[str]

class Menu(BaseModel):
    name: str
    description: str
    image_url: str
    ingredients: List[str]
    steps: List[str]

    model_config = {"arbitrary_types_allowed": True}

class SuggestMenusResponse(BaseModel):
    menus: List[Menu]

# Load the machine learning model for ingredient recognition
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(_BASE_DIR, 'path', 'model.h5'))

def standard_response(status: str, message: str, data: Union[dict, list, None] = None):
    """
    Helper function to standardize JSON responses.
    """
    return {
        "status": status,
        "message": message,
        "data": data
    }

@app.get("/")
async def root():
    return standard_response("success", "App is running")

@app.post("/recognize_ingredients", response_model=RecognizeIngredientsResponse)
async def recognize_ingredients(image: UploadFile = File(...)):
    # Save the uploaded image temporarily
    with open("temp_image.jpg", "wb") as temp_image:
        temp_image.write(await image.read())

    # Preprocess the image and perform ingredient recognition using the machine learning model
    img = tf.keras.preprocessing.image.load_img('temp_image.jpg', target_size=(320, 320))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_tensor = tf.expand_dims(img_array, axis=0)

    # Perform prediction on the preprocessed image tensor
    predictions = model.predict(img_tensor)

    # Convert the predictions to ingredient names (modify based on model output format)
    recognized_ingredients = [str(label) for label in np.argmax(predictions, axis=1)]  # Modify as per model's output

    # Return the list of recognized ingredients
    return standard_response("success", "Ingredients recognized successfully", {"ingredients": recognized_ingredients})

@app.post("/suggest_menus", response_model=SuggestMenusResponse)
async def suggest_menus(request: SuggestMenusRequest):
    ingredients = request.ingredients

    # Call the function or method to suggest menus based on the provided ingredients
    suggested_menus = suggest_menus_based_on_ingredients(ingredients)

    # Return the list of suggested menus
    return standard_response("success", "Menus suggested successfully", {"menus": suggested_menus})

def suggest_menus_based_on_ingredients(ingredients: List[str]) -> List[Menu]:
    # Implement your own logic to suggest menus based on the provided ingredients
    # Return a list of Menu objects representing the suggested menus
    suggested_menus = [
        Menu(
            name=f"Menu {i}",
            description=f"This is menu {i}",
            image_url=f"https://example.com/menu_{i}.jpg",
            ingredients=random.choices(ingredients, k=random.randint(1, len(ingredients))),
            steps=[f"Step {j}" for j in range(1, random.randint(5, 15))]
        )
        for i in range(10)
    ]

    return suggested_menus

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

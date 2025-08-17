# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from io import BytesIO
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import img_to_array
# from LastYear import model, label_encoder

# app = FastAPI()

# import uvicorn
# import socket
# from fastapi import HTTPException

# def is_port_in_use(port: int) -> bool:
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         return s.connect_ex(('localhost', port)) == 0

# if __name__ == "__main__":
#     port = 8000
#     if is_port_in_use(port):
#         print(f"Port {port} is already in use, trying port 8001")
#         port = 8001

#     try:
#         uvicorn.run(app, host="0.0.0.0", port=port)
#     except Exception as e:
#         print(f"Failed to start server: {e}")

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Helper function to preprocess image
# def preprocess_image(image: Image.Image):
#     image = image.resize((224, 224))
#     img_array = img_to_array(image)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
#     return img_array

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Verify file is an image
#         if not file.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="File must be an image")

#         img = Image.open(BytesIO(await file.read()))
#         img_array = preprocess_image(img)

#         # Run prediction
#         predictions = model.predict(img_array)
#         predicted_class = np.argmax(predictions)
#         predicted_label = label_encoder.inverse_transform([predicted_class])[0]

#         return {
#             "animal": predicted_label,
#             "confidence": float(np.max(predictions))
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from LastYear import model, label_encoder

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper function to preprocess image
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Verify file is an image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        img = Image.open(BytesIO(await file.read()))
        img_array = preprocess_image(img)

        # Run prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]

        return {
            "animal": predicted_label,
            "confidence": float(np.max(predictions))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

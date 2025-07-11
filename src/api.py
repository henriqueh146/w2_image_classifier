import io
import torch

from PIL import Image

from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.models.model import BinaryImageClassifier

app = FastAPI()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

model = BinaryImageClassifier.load_from_checkpoint("models/horse_human_classifier.ckpt")
model.eval()

CLASS_NAMES = ["horse", "human"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred = (output > 0.5).int().item()

    predicted_class = CLASS_NAMES[pred]
    return {"prediction": f"Is a {predicted_class}"}

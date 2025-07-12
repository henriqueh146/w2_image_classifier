import pandas as pd
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

from models.model import BinaryImageClassifier

def evaluate_model():
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    
    dataset = ImageFolder("data/validation", transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    
    model = BinaryImageClassifier.load_from_checkpoint("models/horse_human_classifier.ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device)
        with torch.no_grad():
            preds = model(x)
            preds = (preds > 0.91).int().cpu()
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())

    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=dataset.classes))

    
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df.to_csv("predictions.csv", index=False)
    print("\nPrevisões salvas em predictions.csv")


if __name__ == "__main__":
    evaluate_model()
    
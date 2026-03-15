import torch
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
"akiec","bcc","bkl","df","mel","nv","vasc"
]

RISK_MAP = {
"mel":"Critical",
"bcc":"High",
"akiec":"Medium",
"bkl":"Low",
"df":"Low",
"nv":"Low",
"vasc":"Low"
}

MODEL_PATH = "models/skin_model.pth"

model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features,7)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(
[0.485,0.456,0.406],
[0.229,0.224,0.225]
)
])

def predict_image(path):

    img = Image.open(path).convert("RGB")

    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(img)

        probs = torch.softmax(outputs,1)

        top2 = torch.topk(probs,2)

    results = []

    for i in range(2):

        idx = top2.indices[0][i].item()

        disease = CLASSES[idx]

        confidence = float(top2.values[0][i]) * 100

        results.append({
        "disease": disease,
        "confidence": round(confidence,2),
        "risk": RISK_MAP[disease]
        })

    return results
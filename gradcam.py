import os
import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "skin_model.pth")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "heatmaps")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model once at module level (same as predict.py) ─────────────
_model = models.resnet50()
_model.fc = torch.nn.Linear(_model.fc.in_features, 7)
_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
_model.to(DEVICE)
_model.eval()

_target_layer = _model.layer4[-1]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def generate_gradcam(image_path):
    """
    Generate a GradCAM heatmap for the given image.
    Returns the relative URL path to the saved heatmap image.
    """
    features   = []
    gradients  = []

    def forward_hook(module, inp, out):
        features.clear()
        features.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.clear()
        gradients.append(grad_out[0].detach())

    fwd_handle = _target_layer.register_forward_hook(forward_hook)
    bwd_handle = _target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0].detach()) or gradients.__setitem__(0, go[0].detach())
    )
    # Use simpler approach — register_backward_hook is deprecated, use this pattern:
    fwd_handle.remove()

    # Re-register properly
    features.clear()
    gradients.clear()

    fwd_handle = _target_layer.register_forward_hook(forward_hook)
    bwd_handle = _target_layer.register_backward_hook(
        lambda m, gi, go: (gradients.clear(), gradients.append(go[0].detach()))
    )

    try:
        # Load and preprocess image
        img_pil    = Image.open(image_path).convert("RGB")
        inp_tensor = _transform(img_pil).unsqueeze(0).to(DEVICE)

        # Forward pass
        output    = _model(inp_tensor)
        class_idx = int(torch.argmax(output, dim=1).item())

        # Backward pass
        _model.zero_grad()
        output[0, class_idx].backward()

        # Build CAM
        if not gradients or not features:
            raise ValueError("Hooks did not capture gradients/features")

        grads   = gradients[0].cpu().numpy()[0]   # (C, H, W)
        fmap    = features[0].cpu().numpy()[0]     # (C, H, W)
        weights = np.mean(grads, axis=(1, 2))      # (C,)

        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        for w, f in zip(weights, fmap):
            cam += w * f

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (224, 224))

        # Overlay on original
        original = cv2.imread(image_path)
        if original is None:
            # Fallback: convert PIL to cv2
            original = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        original = cv2.resize(original, (224, 224))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        result  = np.clip(heatmap * 0.4 + original, 0, 255).astype(np.uint8)

        # Save
        out_filename = f"heatmap_{os.path.basename(image_path)}"
        out_path     = os.path.join(OUTPUT_DIR, out_filename)
        cv2.imwrite(out_path, result)

        # Return relative URL path
        return f"static/heatmaps/{out_filename}"

    finally:
        fwd_handle.remove()
        bwd_handle.remove()
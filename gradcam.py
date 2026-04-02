try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch/Torchvision not found. Grad-CAM will be disabled.")

import cv2
import numpy as np
from PIL import Image
import io, base64

if TORCH_AVAILABLE:
    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
            
            # Register hooks
            self.target_layer.register_forward_hook(self.save_activation)
            self.target_layer.register_backward_hook(self.save_gradient)

        def save_activation(self, module, input, output):
            self.activations = output

        def save_gradient(self, module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def generate(self, image_tensor, target_class=None):
            # Forward pass
            logit = self.model(image_tensor)
            
            if target_class is None:
                target_class = torch.argmax(logit, dim=1).item()
                
            # Backward pass
            self.model.zero_grad()
            logit[0, target_class].backward()
            
            # Weight the channels by the gradients
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            
            # ReLU to keep only positive influence
            cam = F.relu(cam)
            
            # Normalize
            cam = cam - torch.min(cam)
            cam = cam / (torch.max(cam) + 1e-7)
            
            return cam.squeeze().cpu().detach().numpy()

    _GCAM = None
    _TRANSFORMS = None
    _DEVICE = torch.device("cpu") if TORCH_AVAILABLE else None

    def _get_gcam():
        global _GCAM, _TRANSFORMS
        if not TORCH_AVAILABLE: return None, None
        if _GCAM is None:
            try:
                print("INFO: Lazy-loading ResNet localization backbone...")
                model = models.resnet50(pretrained=True).to(_DEVICE).eval()
                target_layer = model.layer4[-1]
                _GCAM = GradCAM(model, target_layer)
                _TRANSFORMS = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            except Exception as e:
                print(f"ERROR: Failed to load localization model: {e}")
        return _GCAM, _TRANSFORMS
else:
    # Mocks for when torch is missing
    _GCAM = None
    _TRANSFORMS = None
    _DEVICE = None
    def _get_gcam(): return None, None

def generate_gradcam_overlay(image_bytes: bytes):
    """
    Analyzes image using ResNet backbone and returns a base64 encoded Grad-CAM overlay.
    """
    gcam, trans = _get_gcam()
    if gcam is None or trans is None:
        return ""
    try:
        # Load and transform image
        original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = original_img.size
        
        input_tensor = trans(original_img).unsqueeze(0).to(_DEVICE).requires_grad_(True)
        
        # Generate Heatmap
        heatmap = gcam.generate(input_tensor)
        
        # Resize heatmap back to original size
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Apply Jet colormap
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Overlay
        original_arr = np.array(original_img)
        overlay = cv2.addWeighted(original_arr, 0.6, heatmap_color, 0.4, 0)
        
        # Encode back to base64 for web delivery
        res_img = Image.fromarray(overlay)
        buf = io.BytesIO()
        res_img.save(buf, format="JPEG", quality=85)
        
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
        
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        return ""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import json
import timm
import os
import cv2
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="PEFT-CancerX",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Light UI Theme with your color scheme
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        padding: 2rem;
        background-color: #ffffff;
    }
    
    /* Button Styles */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background: light blue;
        color: white;
        font-weight: 500;
        border: none;
        margin: 10px 0;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .stButton>button:hover {
        background: light green;
        transform: translateY(-1px);
    }
    
    /* Result Box Styles */
    .success-box {
        background: #f8fafc;
        padding: 24px;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
        margin: 20px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Header Container Styles */
    .header-container {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
    }
    
    .header-container h1 {
        color: white !important;
        font-size: 2.8em !important;
        font-weight: 700 !important;
        margin-bottom: 0.5em !important;
        letter-spacing: -0.5px;
    }
    
    .header-container p {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 1.2em !important;
        line-height: 1.6 !important;
    }
    
    /* Feature Card Styles */
    .feature-card {
        background: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 12px 0;
        transition: all 0.2s ease;
        border: 1px solid #f1f5f9;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Dark Feature Card for Key Features */
    .dark-feature-card {
        background: #1e293b;
        padding: 2rem;
        border-radius: 12px;
        margin: 20px 0;
        color: white;
    }
    
    .dark-feature-card h3 {
        color: #60a5fa !important;
        margin-bottom: 1rem !important;
    }
    
    .dark-feature-card ul {
        color: #cbd5e1 !important;
        line-height: 2 !important;
    }
    
    .dark-feature-card li {
        margin-bottom: 0.5rem;
    }
    
    /* Typography Styles */
    h1 {
        color: #1e293b;
        font-size: 2.5em !important;
        font-weight: 600 !important;
        margin-bottom: 0.5em !important;
    }
    h3 {
        color: #334155 !important;
        font-size: 1.5em !important;
        font-weight: 500 !important;
    }
    p {
        color: #64748b !important;
        line-height: 1.6 !important;
        font-size: 1rem !important;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Image Display Styles */
    .stImage {
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* Hide fullscreen button on images - multiple selectors */
    button[title="View fullscreen"] {
        display: none !important;
    }

    [data-testid="StyledFullScreenButton"] {
        display: none !important;
    }

    .stImage button {
        display: none !important;
    }

    div[data-testid="stImage"] button {
        display: none !important;
    }

    /* Footer Styles */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
""", unsafe_allow_html=True)

# ==================== MODEL CLASSES ====================
class LoConLayer(nn.Module):
    """LoCon (LoRA for Convolutions) implementation for Conv2d layers"""
    def __init__(self, original_layer, r=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        
        if isinstance(original_layer, nn.Conv2d):
            self.out_channels = original_layer.out_channels
            self.in_channels = original_layer.in_channels
            self.kernel_size = original_layer.kernel_size
            self.stride = original_layer.stride
            self.padding = original_layer.padding
            self.groups = original_layer.groups
        else:
            raise ValueError("LoCon can only be applied to Conv2d layers")
        
        self.lora_down = nn.Conv2d(
            self.in_channels, r, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.lora_up = nn.Conv2d(
            r, self.out_channels, kernel_size=self.kernel_size, 
            stride=self.stride, padding=self.padding, groups=1, bias=False
        )
        
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)
        self.scaling = alpha / r
        
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        original_out = self.original_layer(x)
        lora_out = self.lora_up(self.lora_down(x)) * self.scaling
        return original_out + lora_out


class LoConLinear(nn.Module):
    """LoCon implementation for Linear layers (for transformer blocks)"""
    def __init__(self, original_layer, r=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        
        if isinstance(original_layer, nn.Linear):
            self.in_features = original_layer.in_features
            self.out_features = original_layer.out_features
        else:
            raise ValueError("LoConLinear can only be applied to Linear layers")
        
        self.lora_down = nn.Linear(self.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, self.out_features, bias=False)
        
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)
        self.scaling = alpha / r
        
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        original_out = self.original_layer(x)
        lora_out = self.lora_up(self.lora_down(x)) * self.scaling
        return original_out + lora_out


def apply_locon_to_mobilevit(model, target_conv_modules=["conv", "depthwise", "pointwise", "attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"], 
                             locon_r=8, locon_alpha=16):
    """Apply LoCon to MobileViT model (both Conv2d and Linear layers)"""
    locon_modules = {}
    modules_to_replace = {}
    
    for name, module in model.named_modules():
        should_apply_locon = any(target in name for target in target_conv_modules)
        
        if should_apply_locon:
            if not any(skip in name for skip in ['classifier', 'head', 'norm']):
                if isinstance(module, nn.Conv2d):
                    modules_to_replace[name] = ('locon_conv', module)
                elif isinstance(module, nn.Linear):
                    modules_to_replace[name] = ('locon_linear', module)
    
    for name, (method, original_module) in modules_to_replace.items():
        try:
            if method == 'locon_conv':
                locon_layer = LoConLayer(original_module, r=locon_r, alpha=locon_alpha)
            elif method == 'locon_linear':
                locon_layer = LoConLinear(original_module, r=locon_r, alpha=locon_alpha)
            else:
                continue
                
            locon_modules[name] = locon_layer
            
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name == '':
                setattr(model, child_name, locon_layer)
            else:
                parent_module = model
                for part in parent_name.split('.'):
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, child_name, locon_layer)
        except Exception as e:
            pass
    
    return model, locon_modules


class EnhancedMobileViTWithLoCon(nn.Module):
    def __init__(self, num_classes=5, model_name='mobilevitv2_100', pretrained=True, 
                 use_locon=True, locon_r=8, locon_alpha=16, image_size=(224, 224)):
        super().__init__()
        
        try:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception as e:
            self.backbone = timm.create_model('mobilevitv2_050', pretrained=pretrained, num_classes=0)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size[0], image_size[1])
            features = self.backbone(dummy_input)
            in_features = features.shape[1]
        
        if use_locon:
            self.backbone, self.locon_modules = apply_locon_to_mobilevit(
                self.backbone, locon_r=locon_r, locon_alpha=locon_alpha
            )
        else:
            self.locon_modules = {}
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ==================== LOADING FUNCTIONS ====================
def load_model_keras_pytorch(filepath, device='cpu'):
    """Load PyTorch model from HDF5 format"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        metadata = {}
        if 'metadata' in f:
            metadata_group = f['metadata']
            for key in metadata_group.attrs.keys():
                value = metadata_group.attrs[key]
                if isinstance(value, str):
                    try:
                        metadata[key] = json.loads(value)
                    except:
                        metadata[key] = value
                else:
                    metadata[key] = value
        
        weights_group = f['model_weights']
        state_dict = {}
        
        for name in weights_group.keys():
            param_data = weights_group[name][...]
            state_dict[name] = torch.from_numpy(param_data)
    
    return state_dict, metadata


@st.cache_resource
def load_cancer_model():
    """Load the cancer classification model (cached)"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = "models/best_model.h5"
        
        state_dict, metadata = load_model_keras_pytorch(model_path, device=device)
        
        num_classes = metadata.get('num_classes', 5)
        class_names = metadata.get('class_names', ['lung_aca', 'lung_n', 'lung_scc', 'colon_aca', 'colon_n'])
        
        model = EnhancedMobileViTWithLoCon(
            num_classes=num_classes,
            model_name='mobilevitv2_100',
            pretrained=False,
            use_locon=True,
            locon_r=8,
            locon_alpha=16,
            image_size=(224, 224)
        )
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        return model, class_names, device, metadata
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for PyTorch model"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


def predict_cancer(model, image, class_names, device):
    """Make prediction on cancer image"""
    img_tensor = preprocess_image(image, target_size=(224, 224))
    img_tensor = img_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()
        
        all_probs = probabilities[0].cpu().numpy()
        all_probabilities = {class_names[i]: float(all_probs[i]) for i in range(len(class_names))}
    
    return predicted_class, confidence_score, all_probabilities


def generate_gradcam(model, image, target_size=(224, 224)):
    """Generate GradCAM visualization - more robust version"""
    try:
        model.eval()
        img_tensor = preprocess_image(image, target_size)
        
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        img_tensor.requires_grad = True
        
        # Find the last convolutional layer in backbone (not wrapped in LoConLayer)
        target_layer = None
        target_layer_name = None
        
        for name, module in model.backbone.named_modules():
            # Look for Conv2d layers, skip LoConLayer wrappers
            if isinstance(module, nn.Conv2d):
                # Skip if it's inside a LoConLayer
                if 'lora' not in name.lower() and 'original_layer' not in name.lower():
                    target_layer = module
                    target_layer_name = name
            # Also check if module is a LoConLayer and get its original layer
            elif hasattr(module, 'original_layer') and isinstance(module.original_layer, nn.Conv2d):
                target_layer = module.original_layer
                target_layer_name = name + '.original_layer'
        
        if target_layer is None:
            st.warning("No suitable convolutional layer found for GradCAM")
            return None
        
        # Storage for activations and gradients
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())
        
        # Register hooks
        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_full_backward_hook(backward_hook)
        
        # Forward pass
        outputs = model(img_tensor)
        pred_class = outputs.argmax(dim=1).item()
        
        # Backward pass
        model.zero_grad()
        
        # Create one-hot output for the predicted class
        one_hot_output = torch.zeros_like(outputs)
        one_hot_output[0, pred_class] = 1
        
        # Backward pass
        outputs.backward(gradient=one_hot_output, retain_graph=True)
        
        # Remove hooks
        handle_forward.remove()
        handle_backward.remove()
        
        # Check if we got gradients
        if len(gradients) == 0 or len(activations) == 0:
            st.warning("Failed to capture gradients or activations")
            return None
        
        # Get activations and gradients
        activation = activations[0].cpu()
        gradient = gradients[0].cpu()
        
        # Global average pooling of gradients
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = (weights * activation).sum(dim=1, keepdim=True)
        
        # Apply ReLU
        cam = F.relu(cam)
        cam = cam.squeeze().numpy()
        
        # Check if CAM is valid
        if cam.max() == 0:
            st.warning("GradCAM produced empty heatmap")
            return None
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to match input
        cam = cv2.resize(cam, target_size)
        
        return cam
        
    except Exception as e:
        st.error(f"Error in GradCAM generation: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def apply_gradcam_overlay(image, cam, alpha=0.5):
    """Apply GradCAM overlay on image"""
    # Convert PIL to numpy
    img_array = np.array(image.resize((224, 224)))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = heatmap * alpha + img_array * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay


# ==================== SIDEBAR NAVIGATION ====================
with st.sidebar:
    st.title("🔬 PEFT-CancerX")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["Home", "Evaluate Model", "Sample Images", "How to Use", "Privacy Policy"],
        label_visibility="collapsed"
    )

# ==================== PAGE: HOME ====================
if page == "Home":
    st.markdown("""
        <div class="header-container">
            <h1>PEFT-CancerX</h1>
            <p>Precise lung and colon cancer histopathological image analysis powered by deep learning</p>
        
        </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown("""
        <div class="dark-feature-card">
            <h3>Key Features</h3>
            <ul>
                <li><strong>Evaluate Model:</strong> Multi-class cancer classification (Lung Adenocarcinoma, Lung Squamous Cell Carcinoma, Colon Adenocarcinoma, Benign Tissue)</li>
                <li><strong>Sample Images:</strong> View sample histopathological images used for testing and evaluate the model</li>
                <li><strong>How to Use:</strong> Learn how to use the system effectively</li>
                <li><strong>Privacy Policy:</strong> Understand how your data is handled</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # System Requirements
    st.markdown("### System Requirements")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>✅ Requirements</h3>
                <ul>
                    <li>Modern browser (Chrome/Firefox/Safari/Edge)</li>
                    <li>Internet for model loading</li>
                    <li>Image formats: PNG, JPG, JPEG</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>🎯 Navigation</h3>
                <p>Use the sidebar to navigate:</p>
                <ul>
                    <li>Evaluate Model</li>
                    <li>Sample Images</li>
                    <li>How to Use</li>
                    <li>Privacy Policy</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>© 2025, PEFT-CancerX v1.0, Research Led by Dr. Amith khandakar</p>
            <p>Advanced Histopathological Image Classification</p>
            <p>Contact: nafisa21@iut-dhaka.edu</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== PAGE: EVALUATE MODEL ====================
elif page == "Evaluate Model":
    st.markdown("""
        <div class="header-container">
            <h1>🔍 Evaluate Model</h1>
            <p>Upload histopathological images for cancer classification analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, class_names, device, metadata = load_cancer_model()
    
    if model is None:
        st.error("Failed to load model. Please check if the model file exists.")
        st.stop()
    
    # Model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="feature-card">
                <h3>Model Status</h3>
                <p>✅ Loaded Successfully</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="feature-card">
                <h3>Test Accuracy</h3>
                <p>{metadata.get('test_accuracy', 0):.2%}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="feature-card">
                <h3>Device</h3>
                <p>{str(device).upper()}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File upload
    st.markdown("""
        <div class="feature-card">
            <h3>📷 Upload Histopathological Image</h3>
            <p>Upload a histopathological image of lung or colon tissue for classification</p>
        </div>
    """, unsafe_allow_html=True)
    
    test_image = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if test_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📷 Uploaded Image")
            image = Image.open(test_image)
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Classification Results")
            
            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, all_probs = predict_cancer(model, image, class_names, device)
                    
                    st.markdown(f"""
                        <div class="success-box">
                            <h3>Prediction Result</h3>
                            <p style="font-size: 1.4em; color: #1e293b !important; font-weight: 600;">
                                {predicted_class.replace('_', ' ').title()}
                            </p>
                            <p style="font-size: 1.2em; color: #334155 !important;">
                                Confidence: {confidence:.2%}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### Class Probabilities")
                    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                    
                    for class_name, prob in sorted_probs:
                        st.progress(prob, text=f"{class_name.replace('_', ' ').title()}: {prob:.2%}")
        
        # GradCAM Analysis
        st.markdown("---")
        st.markdown("### 🎨 GradCAM Analysis")
        
        if st.button("Generate GradCAM Visualization", use_container_width=True):
            with st.spinner("Generating GradCAM..."):
                try:
                    cam = generate_gradcam(model, image)
                    
                    if cam is not None:
                        overlay = apply_gradcam_overlay(image, cam)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("#### Original Image")
                            st.image(image.resize((224, 224)), use_container_width=True)
                        
                        with col2:
                            st.markdown("#### Heatmap")
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(cam, cmap='jet')
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close()
                        
                        with col3:
                            st.markdown("#### Overlay")
                            st.image(overlay, use_container_width=True)
                    else:
                        st.warning("Could not generate GradCAM visualization")
                except Exception as e:
                    st.error(f"Error generating GradCAM: {e}")

# ==================== PAGE: SAMPLE IMAGES ====================
elif page == "Sample Images":
    st.markdown("""
        <div class="header-container">
            <h1>📸 Sample Images</h1>
            <p>Explore representative histopathological images across cancer classes</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Define sample categories
    categories = {
        "Lung Adenocarcinoma": "lung_aca",
        "Lung Squamous Cell Carcinoma": "lung_scc",
        "Lung Benign Tissue": "lung_n",
        "Colon Adenocarcinoma": "colon_aca",
        "Colon Benign Tissue": "colon_n"
    }
    
    # Check if sample_images folder exists
    sample_folder = "sample_images"
    if os.path.exists(sample_folder):
        for category_name, category_code in categories.items():
            st.markdown(f"### {category_name}")
            
            category_path = os.path.join(sample_folder, category_code)
            if os.path.exists(category_path):
                image_files = [f for f in os.listdir(category_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                
                if image_files:
                    cols = st.columns(4)  # 4 images per row
                    for idx, img_file in enumerate(image_files[:8]):  # Show max 8 images per category
                        with cols[idx % 4]:
                            img_path = os.path.join(category_path, img_file)
                            st.image(img_path, caption=category_name, width=200)  # Fixed width
                else:
                    st.info(f"No images found for {category_name}")
            else:
                st.info(f"Folder not found: {category_path}")
            
            st.markdown("---")
    else:
        st.warning(f"Sample images folder '{sample_folder}' not found. Please create it and add sample images.")
        
        st.markdown("""
            <div class="feature-card">
                <h3>📁 Folder Structure</h3>
                <pre>
sample_images/
├── lung_aca/
│   └── (lung adenocarcinoma images)
├── lung_scc/
│   └── (lung squamous cell carcinoma images)
├── lung_n/
│   └── (lung benign tissue images)
├── colon_aca/
│   └── (colon adenocarcinoma images)
└── colon_n/
    └── (colon benign tissue images)
                </pre>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="footer">
            <p>© 2025 ,PEFT-CancerX v1.0, Research Led by Dr. Amith Khandakar</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== PAGE: HOW TO USE ====================
elif page == "How to Use":
    st.markdown("""
        <div class="header-container">
            <h1>📖 How to Use PEFT-CancerX</h1>
            <p>Step-by-step guide to using the cancer classification system</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-card">
            <h3>🎥 Video Tutorial</h3>
            <p>Watch this short video to understand the complete workflow:</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Placeholder for video
    st.video("https://youtu.be/UPiA_pBv2BI")  # Replace with actual video URL
    
    st.markdown("---")
    
    # Step-by-step guide
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>Step 1: Upload Image</h3>
                <p>Navigate to <strong>Evaluate Model</strong> page</p>
                <ul>
                    <li>Click on the upload button</li>
                    <li>Select a histopathological image (PNG, JPG, JPEG)</li>
                    <li>Image should be clear and well-focused</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h3>Step 3: Review Results</h3>
                <ul>
                    <li>Check the predicted cancer type</li>
                    <li>Review confidence score</li>
                    <li>Examine all class probabilities</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>Step 2: Analyze</h3>
                <p>Click the <strong>"Analyze Image"</strong> button</p>
                <ul>
                    <li>Model will process the image</li>
                    <li>Classification results will appear</li>
                    <li>Wait for processing to complete</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h3>Step 4: GradCAM (Optional)</h3>
                <ul>
                    <li>Click "Generate GradCAM Visualization"</li>
                    <li>See which regions influenced the prediction</li>
                    <li>Review heatmap and overlay</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Best Practices
    st.markdown("""
        <div class="feature-card">
            <h3>✅ Best Practices</h3>
            <ul>
                <li><strong>Image Quality:</strong> Use high-resolution, clear histopathological images</li>
                <li><strong>Image Format:</strong> Ensure images are in PNG, JPG, or JPEG format</li>
                <li><strong>Proper Lighting:</strong> Images should be well-lit and properly stained</li>
                <li><strong>Multiple Views:</strong> For better confidence, test multiple images from the same sample</li>
                <li><strong>Professional Use:</strong> This tool is for research purposes - always consult medical professionals</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="footer">
            <p>© 2025, PEFT-CancerX v1.0, Research Led by Dr. Amith Khandakar</p>
            <p>Contact: nafisa21@iut-dhaka.edu</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== PAGE: PRIVACY POLICY ====================
elif page == "Privacy Policy":
    st.markdown("""
        <div class="header-container">
            <h1>🔒 Privacy Policy</h1>
            <p>Understanding data handling and system limitations</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Intended Use
    st.markdown("""
        <div class="feature-card">
            <h3>🎯 Intended Use</h3>
            <p>PEFT-CancerX is designed for <strong>research and educational purposes</strong> to assist in image-based histopathological analysis. It is <strong>not a substitute for professional medical judgment</strong>. It is still under development and is not yet ready for clinical use. A rigorous validation process is ongoing to ensure the accuracy and reliability of the model.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Approved Environments
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>✅ Approved Environments</h3>
                <p><strong>Can be used in:</strong></p>
                <ul>
                    <li>Research laboratories</li>
                    <li>Academic settings</li>
                    <li>Controlled clinical evaluations with appropriate oversight by medical professionals</li>
                    <li>Educational demonstrations</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>❌ Not For</h3>
                <p><strong>Should NOT be used for:</strong></p>
                <ul>
                    <li>Primary diagnosis</li>
                    <li>Autonomous decision-making</li>
                    <li>Unsupervised clinical deployment</li>
                    <li>Clinical use without medical oversight</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Data Policy
    st.markdown("""
        <div class="feature-card">
            <h3>🔐 Data Policy</h3>
            <ul>
                <li><strong>Image Processing:</strong> Uploaded images are processed in-memory for inference and are <strong>not persisted</strong> by the application</li>
                <li><strong>No PHI:</strong> No Protected Health Information (PHI) should be uploaded</li>
                <li><strong>User Responsibility:</strong> Users remain responsible for data compliance with local regulations</li>
                <li><strong>No Storage:</strong> Images are automatically deleted after processing</li>
                <li><strong>No Sharing:</strong> No data is shared with third parties</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Model Limitations
    st.markdown("""
        <div class="feature-card">
            <h3>⚠️ Model Limitations</h3>
            <ul>
                <li><strong>Research Tool:</strong> This is a research prototype, not a diagnostic device</li>
                <li><strong>Validation Required:</strong> Results should always be validated by trained pathologists</li>
                <li><strong>Dataset Bias:</strong> Model trained on specific datasets (LC25000) - may not generalize to all populations</li>
                <li><strong>Image Quality:</strong> Performance depends on image quality and proper staining</li>
                <li><strong>Not FDA Approved:</strong> This system is not FDA approved or cleared for clinical use</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Rights and Contact
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>📜 Rights</h3>
                <p>All rights reserved by the Research Team. The team reserves the right to change features, models, and policies without prior notice.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>📧 Contact</h3>
                <p>For questions, concerns, or feedback:</p>
                <p><strong>Email:</strong> nafisa21@iut-dhaka.edu</p>
                
            </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
        <div class="dark-feature-card">
            <h3>⚖️ Medical Disclaimer</h3>
            <p><strong>IMPORTANT:</strong> This tool is provided for research and educational purposes only. It is NOT intended to diagnose, treat, cure, or prevent any disease. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of information provided by this tool.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p><strong>© 2025 | Research Led by Dr. Amith Khandakar</strong></p>
            <p>PEFT-CancerX v1.0 | Advanced Histopathological Image Classification</p>
            <p>Last Updated: November 2025</p>
            <p>Contact: nafisa21@iut-dhaka.edu</p>
        </div>
    """, unsafe_allow_html=True)
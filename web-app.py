import gradio as gr
import torch
import mimetypes
from PIL import Image
import cv2
from torchvision.models import efficientnet_b0
from torchvision import transforms
import os

# === Load Model ===


def load_model():
    model = efficientnet_b0()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model_path = "models/best_model-v3.pt"
    if not os.path.exists(model_path):
        model_path = "/models/best_model-v3.pt"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


model = load_model()

# === Preprocessing ===
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Inference Logic ===


def predict_file(file_obj):
    if file_obj is None:
        return "No file selected", "0%", None, gr.update(visible=False)

    path = file_obj.name if hasattr(file_obj, 'name') else file_obj
    mime, _ = mimetypes.guess_type(path)

    if mime and mime.startswith("image"):
        img = Image.open(path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)

        is_real = pred.item() == 0
        label = "REAL" if is_real else "DEEPFAKE"
        color = "#22c55e" if is_real else "#ef4444"
        icon = "✓" if is_real else "⚠"

        return label, f"{conf.item()*100:.1f}%", img, gr.update(visible=True, value=f"""
            <div style="
                background: {'linear-gradient(135deg, #064e3b 0%, #065f46 100%)' if is_real else 'linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%)'};
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                border: 2px solid {color};
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            ">
                <div style="font-size: 48px; margin-bottom: 10px;">{icon}</div>
                <div style="font-size: 32px; font-weight: bold; color: white; margin-bottom: 5px;">{label}</div>
                <div style="font-size: 18px; color: rgba(255,255,255,0.8);">Confidence: {conf.item()*100:.1f}%</div>
            </div>
        """)

    elif mime and mime.startswith("video"):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "Error", "0%", None, gr.update(visible=True, value="""
                <div style="background: #374151; border-radius: 12px; padding: 20px; text-align: center; border: 2px solid #ef4444;">
                    <div style="font-size: 24px; color: #ef4444;">Error reading video</div>
                </div>
            """)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)

        is_real = pred.item() == 0
        label = "REAL" if is_real else "DEEPFAKE"
        color = "#22c55e" if is_real else "#ef4444"
        icon = "✓" if is_real else "⚠"

        return label, f"{conf.item()*100:.1f}%", img, gr.update(visible=True, value=f"""
            <div style="
                background: {'linear-gradient(135deg, #064e3b 0%, #065f46 100%)' if is_real else 'linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%)'};
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                border: 2px solid {color};
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            ">
                <div style="font-size: 48px; margin-bottom: 10px;">{icon}</div>
                <div style="font-size: 32px; font-weight: bold; color: white; margin-bottom: 5px;">{label}</div>
                <div style="font-size: 14px; color: rgba(255,255,255,0.7); margin-bottom: 5px;">(First Frame Analysis)</div>
                <div style="font-size: 18px; color: rgba(255,255,255,0.8);">Confidence: {conf.item()*100:.1f}%</div>
            </div>
        """)

    else:
        return "Unsupported", "0%", None, gr.update(visible=True, value="""
            <div style="background: #374151; border-radius: 12px; padding: 20px; text-align: center; border: 2px solid #f59e0b;">
                <div style="font-size: 24px; color: #f59e0b;">Unsupported file type</div>
            </div>
        """)


# === Custom CSS ===
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-bg: #0f172a;
    --secondary-bg: #1e293b;
    --accent: #3b82f6;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
}

body {
    font-family: 'Inter', sans-serif !important;
    background: var(--primary-bg) !important;
}

.gradio-container {
    background: var(--primary-bg) !important;
    max-width: 900px !important;
}

/* Header */
.header {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
    border-radius: 16px;
    margin-bottom: 30px;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.header h1 {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px 0;
}

.header p {
    font-size: 16px;
    color: var(--text-secondary);
    margin: 0;
}

/* File Upload */
.file-upload {
    border: 2px dashed rgba(59, 130, 246, 0.5) !important;
    border-radius: 16px !important;
    background: rgba(30, 41, 59, 0.5) !important;
    padding: 40px !important;
    transition: all 0.3s ease !important;
}

.file-upload:hover {
    border-color: #3b82f6 !important;
    background: rgba(59, 130, 246, 0.1) !important;
}

/* Loading Spinner */
.loading {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 30px;
    color: var(--text-secondary);
}

.spinner {
    width: 24px;
    height: 24px;
    border: 3px solid rgba(59, 130, 246, 0.3);
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Result Card */
.result-card {
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}

/* Preview Image */
.preview-container img {
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.1);
}

/* Footer */
.footer {
    text-align: center;
    padding: 20px;
    color: var(--text-secondary);
    font-size: 12px;
    margin-top: 30px;
}
"""

# === Gradio UI ===
with gr.Blocks(title="Deepfake Detector", css=custom_css) as demo:

    # Header
    gr.HTML("""
        <div class="header">
            <h1>Deepfake Detector</h1>
            <p>AI-powered detection for images and videos using EfficientNet-B0</p>
        </div>
    """)

    # File Upload
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload Image or Video",
                file_types=[".jpg", ".jpeg", ".png", ".mp4", ".mov"],
                elem_classes=["file-upload"]
            )

            gr.Markdown("""
                <div style="text-align: center; color: #64748b; font-size: 13px; margin-top: 10px;">
                    Supported formats: JPG, PNG, MP4, MOV
                </div>
            """)

    # Loading indicator
    loading = gr.HTML("""
        <div id="loading" style="display: none;" class="loading">
            <div class="spinner"></div>
            <span>Analyzing...</span>
        </div>
    """, visible=False)

    # Results
    result_html = gr.HTML(visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            preview = gr.Image(
                label="Preview",
                interactive=False,
                elem_classes=["preview-container"]
            )

    # Hidden outputs for state
    prediction = gr.Textbox(visible=False)
    confidence = gr.Textbox(visible=False)

    # Footer
    gr.HTML("""
        <div class="footer">
            Powered by PyTorch & EfficientNet-B0
        </div>
    """)

    # Event handlers
    def on_file_change(file_obj):
        if file_obj is None:
            return [gr.update(visible=False)] * 5
        return [gr.update(visible=True), gr.update(visible=False), None, None, gr.update(visible=False)]

    def handle_input(file_obj):
        pred, conf, img, result = predict_file(file_obj)
        return pred, conf, img, gr.update(visible=False), result

    file_input.change(
        fn=handle_input,
        inputs=file_input,
        outputs=[prediction, confidence, preview, loading, result_html]
    )

demo.launch()

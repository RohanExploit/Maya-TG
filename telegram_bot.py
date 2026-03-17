import asyncio
import logging
import os
import tempfile
from io import BytesIO
import base64

import torch
import cv2
import numpy as np
from PIL import Image, ExifTags
from torchvision.models import efficientnet_b0
from torchvision import transforms
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai
from scipy import fftpack
from skimage import feature, measure
import warnings
warnings.filterwarnings('ignore')

# Bot token
TOKEN = "5614405588:AAEtmjQNR8cppePAxUxRIlcYzDOK4Y11ghc"

# Gemini API
GEMINI_API_KEY = "AIzaSyAbkpCTBDcTqBdY-aF6Qbo7wPwd8Rw4KNU"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model with low latency settings
generation_config = {
    "temperature": 0.1,
    "top_p": 0.8,
    "top_k": 20,
    "max_output_tokens": 256,
}

gemini_model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config=generation_config
)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model


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

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🧠 *Deepfake Detector Bot*\n\n"
        "Send me an image or video and I'll analyze if it's real or fake.\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - How to use",
        parse_mode='Markdown'
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📸 *Send any image or video*\n\n"
        "I'll analyze it using EfficientNet-B0 and tell you if it's:\n"
        "✅ Real\n"
        "❌ Deepfake\n\n"
        "Supported formats: JPG, PNG, MP4, MOV",
        parse_mode='Markdown'
    )


# === ENHANCED DETECTION FUNCTIONS ===

def analyze_error_level_analysis(image_path):
    """
    Error Level Analysis (ELA) - detects manipulated regions
    Different compression levels indicate tampering
    """
    try:
        img = Image.open(image_path)
        # Save at known quality and compare
        temp_buffer = BytesIO()
        img.save(temp_buffer, format='JPEG', quality=90)
        temp_buffer.seek(0)
        recompressed = Image.open(temp_buffer)

        # Calculate difference
        original_array = np.array(img).astype(float)
        recompressed_array = np.array(recompressed).astype(float)

        if original_array.shape != recompressed_array.shape:
            return {"ela_score": 0, "suspicious": False}

        # Error level
        ela = np.abs(original_array - recompressed_array)
        ela_score = np.mean(ela)

        # High ELA indicates manipulation
        return {
            "ela_score": ela_score,
            "suspicious": ela_score > 15.0
        }
    except:
        return {"ela_score": 0, "suspicious": False}


def analyze_noise_pattern(image_path):
    """
    Analyze noise patterns - AI images often have different noise characteristics
    """
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract noise using high-pass filter
        noise = cv2.subtract(gray, cv2.GaussianBlur(gray, (3, 3), 0))

        # Calculate noise statistics
        noise_mean = np.mean(noise)
        noise_std = np.std(noise)

        # AI images often have uniform or patterned noise
        noise_uniformity = np.std(noise[::4, ::4])  # Sample every 4th pixel

        # Check for unnatural noise patterns
        suspicious = (
            noise_std < 2.0 or  # Too little noise
            noise_std > 25.0 or  # Too much noise
            noise_uniformity < 0.5  # Too uniform
        )

        return {
            "noise_std": noise_std,
            "noise_uniformity": noise_uniformity,
            "suspicious": suspicious
        }
    except:
        return {"noise_std": 0, "noise_uniformity": 0, "suspicious": False}


def analyze_metadata(image_path):
    """
    Check image metadata for signs of AI generation or manipulation
    """
    try:
        img = Image.open(image_path)
        metadata = {}

        # Check for EXIF data
        try:
            exif = img._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    metadata[tag] = value
        except:
            pass

        # Check for software tags indicating AI tools
        ai_software_keywords = [
            'midjourney', 'dalle', 'dall-e', 'stable diffusion',
            'firefly', 'leonardo', 'bing', 'craiyon', 'nightcafe'
        ]

        software_info = str(metadata.get('Software', '')).lower()
        maker_info = str(metadata.get('Make', '')).lower()

        for keyword in ai_software_keywords:
            if keyword in software_info or keyword in maker_info:
                return {"has_ai_metadata": True, "software": metadata.get('Software', 'Unknown')}

        # Check if image lacks camera metadata (common in AI images)
        has_camera_info = any(k in metadata for k in [
                              'Make', 'Model', 'LensModel', 'FNumber'])

        return {
            "has_ai_metadata": False,
            "has_camera_info": has_camera_info,
            "suspicious": not has_camera_info  # No camera info is suspicious
        }
    except:
        return {"has_ai_metadata": False, "suspicious": False}


def analyze_frequency_domain(image_path):
    """
    FFT analysis - AI images often have different frequency characteristics
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply FFT
        f_transform = fftpack.fft2(img)
        f_shift = fftpack.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Analyze frequency distribution
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        # Low vs high frequency ratio
        low_freq = np.mean(
            magnitude[center_h-10:center_h+10, center_w-10:center_w+10])
        high_freq = np.mean(magnitude) - low_freq

        ratio = low_freq / (high_freq + 1e-7)

        # AI images often have different frequency distributions
        suspicious = ratio > 5.0 or ratio < 0.5

        return {
            "freq_ratio": ratio,
            "suspicious": suspicious
        }
    except:
        return {"freq_ratio": 1.0, "suspicious": False}


async def detect_ai_icon_async(image_path):
    """
    Detect AI tool icons/logos in image using Gemini Vision (ASYNC for low latency)
    Returns: (has_icon, icon_name)
    """
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()

        prompt = """
        Quick check: Does this image have ANY AI tool logo/icon? 
        Look for: Midjourney, DALL-E, Stable Diffusion, Firefly, Leonardo, Bing, or AI badges.
        Reply ONLY with JSON: {"has_icon": true/false, "icon_name": "name"}
        """

        # Run Gemini in executor for async behavior
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": image_data}]
            )
        )

        try:
            import json
            result_text = response.text.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text.strip())
            has_icon = result.get("has_icon", False)
            icon_name = result.get("icon_name", "")

            return has_icon, icon_name
        except:
            return False, ""
    except Exception as e:
        return False, ""


async def analyze_with_gemini_async(image_path):
    """
    Use Gemini Vision API to analyze image for deepfake indicators (ASYNC)
    """
    try:
        # Load and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # Create SHORT prompt for low latency
        prompt = """
        Quick check: Is this image AI-generated or manipulated?
        Look for: unnatural skin, weird eyes/teeth, inconsistent lighting, blurry background.
        Reply ONLY with JSON: {"is_fake": true/false, "confidence": 0-100, "indicators": ["issue1", "issue2"]}
        """

        # Generate response (async for low latency)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": image_data}]
            )
        )

        # Parse response
        try:
            import json
            result_text = response.text.strip()
            # Extract JSON if wrapped in markdown
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text.strip())
            return {
                "is_fake": result.get("is_fake", False),
                "confidence": result.get("confidence", 50),
                "indicators": result.get("indicators", [])
            }
        except:
            return {"is_fake": False, "confidence": 50, "indicators": ["Analysis error"]}
    except Exception as e:
        return {"is_fake": False, "confidence": 50, "indicators": [f"Error: {str(e)}"]}


def detect_face_swap_opencv(image_path):
    """
    Face swap detection using OpenCV:
    1. Face detection quality
    2. Edge artifacts around face
    3. Color consistency
    """
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load classifiers
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_count = len(faces)

        if face_count == 0:
            return {"face_count": 0, "suspicious": False, "details": "No face detected"}

        suspicious_indicators = []

        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = img[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]

            # Check 1: Face edge sharpness (swapped faces often have blur at edges)
            edges = cv2.Canny(face_gray, 100, 200)
            edge_ratio = np.sum(edges > 0) / (w * h)
            if edge_ratio < 0.05:
                suspicious_indicators.append("unnatural_face_edges")

            # Check 2: Color consistency in face vs background
            face_mean = np.mean(face_roi, axis=(0, 1))
            bg_roi = img[max(0, y-20):y, max(0, x-20):x+w+20]
            if bg_roi.size > 0:
                bg_mean = np.mean(bg_roi, axis=(0, 1))
                color_diff = np.abs(face_mean - bg_mean)
                if np.mean(color_diff) > 60:
                    suspicious_indicators.append("color_mismatch")

            # Check 3: Face aspect ratio (unnatural proportions)
            aspect_ratio = w / h
            if aspect_ratio < 0.7 or aspect_ratio > 0.95:
                suspicious_indicators.append("unnatural_proportions")

            # Check 4: Face pattern irregularities (texture analysis)
            # Detect unnatural skin patterns common in deepfakes
            face_gray_float = face_gray.astype(np.float32) / 255.0

            # Local Binary Pattern (LBP) for texture analysis
            from scipy import ndimage
            lbp = np.zeros_like(face_gray_float)
            for i in range(1, face_gray_float.shape[0]-1):
                for j in range(1, face_gray_float.shape[1]-1):
                    center = face_gray_float[i, j]
                    binary = (
                        (face_gray_float[i-1, j-1] >= center) << 7 |
                        (face_gray_float[i-1, j] >= center) << 6 |
                        (face_gray_float[i-1, j+1] >= center) << 5 |
                        (face_gray_float[i, j+1] >= center) << 4 |
                        (face_gray_float[i+1, j+1] >= center) << 3 |
                        (face_gray_float[i+1, j] >= center) << 2 |
                        (face_gray_float[i+1, j-1] >= center) << 1 |
                        (face_gray_float[i, j-1] >= center)
                    )
                    lbp[i, j] = binary

            # Check texture uniformity (deepfakes often have overly smooth skin)
            lbp_hist = np.histogram(
                lbp[1:-1, 1:-1], bins=256, range=(0, 256))[0]
            lbp_entropy = -np.sum((lbp_hist / np.sum(lbp_hist))
                                  * np.log2(lbp_hist / np.sum(lbp_hist) + 1e-7))

            # Low entropy indicates overly uniform texture (unnatural)
            if lbp_entropy < 4.5:
                suspicious_indicators.append("unnatural_skin_texture")

            # Check 5: Facial symmetry analysis
            face_width = w
            left_half = face_gray[:, :face_width//2]
            right_half = face_gray[:, -face_width//2:]

            # Resize to same dimensions for comparison
            min_h = min(left_half.shape[0], right_half.shape[0])
            left_half = left_half[:min_h, :]
            right_half = right_half[:min_h, :]

            # Flip right half for comparison
            right_half_flipped = np.fliplr(right_half)

            # Calculate symmetry score
            symmetry_score = np.corrcoef(
                left_half.flatten(), right_half_flipped.flatten())[0, 1]

            # Perfect symmetry is suspicious (real faces are asymmetric)
            if symmetry_score > 0.95:
                suspicious_indicators.append("unnatural_symmetry")
            elif symmetry_score < 0.6:
                suspicious_indicators.append("asymmetric_irregularities")

            # Check 6: Eye region analysis
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)

            if len(eyes) >= 2:
                # Check if eyes are at similar heights (unnatural if perfectly aligned)
                eye_y_positions = [eye[1] for eye in eyes[:2]]
                y_diff = abs(eye_y_positions[0] - eye_y_positions[1])
                if y_diff < 2:  # Too perfectly aligned
                    suspicious_indicators.append("unnatural_eye_alignment")

        is_suspicious = len(suspicious_indicators) >= 2

        return {
            "face_count": face_count,
            "suspicious": is_suspicious,
            "indicators": suspicious_indicators,
            "details": ", ".join(suspicious_indicators) if suspicious_indicators else "Normal"
        }
    except Exception as e:
        return {"face_count": 0, "suspicious": False, "details": f"Error: {str(e)}"}


async def analyze_image_async(image_path):
    """Analyze image with ENHANCED multi-layer approach (ASYNC for low latency)"""

    # Layer 0: AI Icon detection (highest priority) - async
    has_icon, icon_name = await detect_ai_icon_async(image_path)
    if has_icon:
        return "AI ICON DETECTED ❌", 95.0, 0, f"Icon detected: '{icon_name}'"

    # ENHANCED: Run all technical analyses in parallel
    ela_result = analyze_error_level_analysis(image_path)
    noise_result = analyze_noise_pattern(image_path)
    metadata_result = analyze_metadata(image_path)
    freq_result = analyze_frequency_domain(image_path)

    # Check enhanced detections first
    if metadata_result.get("has_ai_metadata"):
        return "AI METADATA DETECTED ❌", 98.0, 0, f"AI software: {metadata_result.get('software', 'Unknown')}"

    if ela_result["suspicious"] and noise_result["suspicious"]:
        return "MANIPULATION DETECTED ❌", 90.0, 0, "ELA + Noise anomalies"

    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)

    # Layer 1: Model prediction (fast, synchronous)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)

    model_confidence = conf.item()
    model_label = "FAKE" if pred.item() == 1 else "REAL"

    # Layer 2 & 3: Run face swap and Gemini in parallel for low latency
    face_analysis = detect_face_swap_opencv(image_path)
    gemini_result = await analyze_with_gemini_async(image_path)

    gemini_fake = gemini_result["is_fake"]
    gemini_conf = gemini_result["confidence"] / 100.0
    gemini_indicators = gemini_result["indicators"]

    # ENHANCED: Build comprehensive reason
    enhanced_indicators = []
    if ela_result["suspicious"]:
        enhanced_indicators.append("compression_anomaly")
    if noise_result["suspicious"]:
        enhanced_indicators.append("unnatural_noise")
    if freq_result["suspicious"]:
        enhanced_indicators.append("frequency_anomaly")
    if not metadata_result.get("has_camera_info", True):
        enhanced_indicators.append("no_camera_metadata")

    # Multi-layer decision with enhanced detection
    if model_confidence > 0.85 and model_label == "FAKE":
        final_label = "DEEPFAKE ❌"
        reason = "High confidence AI detection"
    elif gemini_fake and gemini_conf > 0.7:
        final_label = "AI-GENERATED ❌"
        reason = f"Gemini detected: {', '.join(gemini_indicators[:3])}"
    elif face_analysis["suspicious"] and model_label == "FAKE":
        final_label = "FACE SWAP ❌"
        reason = f"Face swap detected: {face_analysis['details']}"
    elif len(enhanced_indicators) >= 2:
        final_label = "MANIPULATED ❌"
        reason = f"Technical: {', '.join(enhanced_indicators[:2])}"
    elif ela_result["suspicious"]:
        final_label = "SUSPICIOUS ⚠️"
        reason = "Error level analysis indicates tampering"
    elif model_confidence > 0.7 and model_label == "FAKE":
        final_label = "SUSPICIOUS ⚠️"
        reason = "Possible manipulation"
    elif face_analysis["suspicious"]:
        final_label = "SUSPICIOUS ⚠️"
        reason = f"Anomalies: {face_analysis['details']}"
    elif gemini_fake:
        final_label = "SUSPICIOUS ⚠️"
        reason = f"Gemini flags: {', '.join(gemini_indicators[:2])}"
    else:
        final_label = "REAL ✅"
        reason = "No manipulation detected"

    confidence = model_confidence * 100

    return final_label, confidence, face_analysis["face_count"], reason


def analyze_video(video_path):
    """Analyze first frame of video"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Error reading video", 0

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)

    is_real = pred.item() == 0
    label = "REAL ✅ (1st frame)" if is_real else "DEEPFAKE ❌ (1st frame)"
    confidence = conf.item() * 100

    return label, confidence


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle image uploads with low latency async analysis"""
    await update.message.reply_text("🔍 Analyzing image...")

    # Get photo
    photo = update.message.photo[-1]  # Get highest quality
    file = await context.bot.get_file(photo.file_id)

    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    try:
        # Analyze with async multi-layer detection (LOW LATENCY)
        label, confidence, face_count, reason = await analyze_image_async(tmp_path)

        # Send result
        result_text = (
            f"🧠 *Analysis Result*\n\n"
            f"📊 Prediction: *{label}*\n"
            f"🎯 Confidence: *{confidence:.1f}%*\n"
            f"👤 Faces: *{face_count}*\n"
            f"📝 Reason: {reason}\n\n"
            f"Multi-layer: EfficientNet + Gemini + Face Analysis"
        )
        await update.message.reply_text(result_text, parse_mode='Markdown')

    finally:
        os.unlink(tmp_path)


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle video uploads"""
    await update.message.reply_text("🔍 Analyzing video (first frame)...")

    # Get video
    video = update.message.video or update.message.document
    file = await context.bot.get_file(video.file_id)

    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    try:
        # Analyze
        label, confidence = analyze_video(tmp_path)

        # Send result
        result_text = (
            f"🧠 *Video Analysis*\n\n"
            f"📊 Prediction: *{label}*\n"
            f"🎯 Confidence: *{confidence:.1f}%*\n\n"
            f"⚠️ Note: Only first frame analyzed\n"
            f"Powered by EfficientNet-B0"
        )
        await update.message.reply_text(result_text, parse_mode='Markdown')

    finally:
        os.unlink(tmp_path)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle document uploads (images/videos as documents)"""
    doc = update.message.document
    mime_type = doc.mime_type or ""

    if mime_type.startswith("image"):
        await update.message.reply_text("🔍 Analyzing image...")

        file = await context.bot.get_file(doc.file_id)
        ext = ".jpg" if "jpeg" in mime_type else ".png"

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        try:
            label, confidence = analyze_image(tmp_path)
            result_text = (
                f"🧠 *Analysis Result*\n\n"
                f"📊 Prediction: *{label}*\n"
                f"🎯 Confidence: *{confidence:.1f}%*\n\n"
                f"Powered by EfficientNet-B0"
            )
            await update.message.reply_text(result_text, parse_mode='Markdown')
        finally:
            os.unlink(tmp_path)

    elif mime_type.startswith("video"):
        await handle_video(update, context)
    else:
        await update.message.reply_text("❌ Unsupported file type. Send images (JPG, PNG) or videos (MP4).")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.message:
        await update.message.reply_text("❌ An error occurred. Please try again.")


def main():
    # Create application
    application = Application.builder().token(TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(MessageHandler(
        filters.Document.ALL, handle_document))
    application.add_error_handler(error_handler)

    # Run
    logger.info("Bot started!")
    application.run_polling()


if __name__ == "__main__":
    main()

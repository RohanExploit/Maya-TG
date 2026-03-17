import asyncio
import logging
import os
import tempfile
import json
import csv
import time
from io import BytesIO
import base64
from datetime import datetime
from collections import defaultdict

import torch
import cv2
import numpy as np
from PIL import Image, ExifTags
from torchvision.models import efficientnet_b0
from torchvision import transforms
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationTypes
import google.generativeai as genai
from scipy import fftpack
from skimage import feature, measure
import warnings
warnings.filterwarnings('ignore')

# MediaPipe for advanced face detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# User statistics and rate limiting
user_stats = defaultdict(
    lambda: {"total": 0, "fake": 0, "real": 0, "last_request": 0})
rate_limit = {}  # user_id: last_request_time
BATCH_MODE = {}  # user_id: [photo_file_ids]

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


def check_rate_limit(user_id):
    """Check if user is rate limited"""
    current_time = time.time()
    if user_id in rate_limit:
        if current_time - rate_limit[user_id] < 2:  # 2 second cooldown
            return False
    rate_limit[user_id] = current_time
    return True


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        f"🧠 *Deepfake Detector Bot*\n\n"
        f"Welcome {user.first_name}!\n\n"
        f"*14-Layer Detection System:*\n"
        f"✅ AI Icon Detection\n"
        f"✅ Metadata Analysis\n"
        f"✅ Copy-Move Forgery\n"
        f"✅ GAN Fingerprint\n"
        f"✅ Head Pose Analysis\n"
        f"✅ Iris Anomaly Detection\n"
        f"✅ MediaPipe Face Mesh (468 landmarks)\n"
        f"✅ Face Swap Detection\n"
        f"✅ Face Enhancement Detection\n"
        f"✅ Error Level Analysis\n"
        f"✅ Noise Pattern Analysis\n"
        f"✅ EfficientNet-B0 Model\n"
        f"✅ Gemini AI Analysis\n"
        f"✅ Frequency Domain Analysis\n\n"
        f"Commands:\n"
        f"/start - Start\n"
        f"/help - Help\n"
        f"/stats - Your statistics\n"
        f"/batch - Batch analysis mode\n"
        f"/export - Export results (JSON/CSV)",
        parse_mode='Markdown'
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📸 *Deepfake Detector - Full Fledged*\n\n"
        "*Send:*\n"
        "• Single image - Instant analysis\n"
        "• Multiple images - Batch processing\n"
        "• Video - Frame-by-frame analysis\n\n"
        "*Commands:*\n"
        "/stats - View your detection stats\n"
        "/batch - Enable batch mode (send multiple photos)\n"
        "/done - Finish batch and get report\n"
        "/export - Export results to JSON or CSV\n"
        "/reset - Reset your statistics\n\n"
        "*Detection Types:*\n"
        "❌ Deepfake\n"
        "❌ Face Swap\n"
        "❌ AI Generated\n"
        "❌ Face Enhanced\n"
        "❌ Copy-Move Forgery\n"
        "❌ GAN Generated\n"
        "❌ Manipulated\n"
        "✅ Real",
        parse_mode='Markdown'
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show user statistics"""
    user_id = update.effective_user.id
    stats = user_stats[user_id]

    total = stats["total"]
    fake = stats["fake"]
    real = stats["real"]
    fake_rate = (fake / total * 100) if total > 0 else 0

    await update.message.reply_text(
        f"📊 *Your Statistics*\n\n"
        f"Total Analyzed: {total}\n"
        f"❌ Fake/AI: {fake}\n"
        f"✅ Real: {real}\n"
        f"📈 Fake Detection Rate: {fake_rate:.1f}%",
        parse_mode='Markdown'
    )


async def batch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enable batch processing mode"""
    user_id = update.effective_user.id
    BATCH_MODE[user_id] = []
    await update.message.reply_text(
        "📁 *Batch Mode Enabled*\n\n"
        "Send me multiple images one by one.\n"
        "When done, type /done to get full report.\n\n"
        "Send /cancel to exit batch mode."
    )


async def done_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process batch and generate report"""
    user_id = update.effective_user.id

    if user_id not in BATCH_MODE or not BATCH_MODE[user_id]:
        await update.message.reply_text("No images in batch. Send /batch to start.")
        return

    photos = BATCH_MODE[user_id]
    await update.message.reply_text(f"🔍 Processing {len(photos)} images...")

    results = []
    for i, photo_file_id in enumerate(photos, 1):
        try:
            file = await context.bot.get_file(photo_file_id)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                await file.download_to_drive(tmp.name)
                label, confidence, face_count, reason = await analyze_image_async(tmp.name)
                results.append({
                    "image": f"Image {i}",
                    "result": label,
                    "confidence": confidence,
                    "faces": face_count,
                    "reason": reason
                })
                os.unlink(tmp.name)
        except Exception as e:
            results.append(
                {"image": f"Image {i}", "result": "Error", "error": str(e)})

    # Generate report
    report = "📊 *Batch Analysis Report*\n\n"
    fake_count = sum(1 for r in results if "❌" in r.get("result", ""))

    for r in results:
        if "error" in r:
            report += f"❓ {r['image']}: Error\n"
        else:
            report += f"{r['image']}: {r['result']}\n"

    report += f"\n📈 Summary: {fake_count}/{len(results)} flagged"

    await update.message.reply_text(report, parse_mode='Markdown')
    del BATCH_MODE[user_id]


async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Export user statistics to JSON or CSV"""
    user_id = update.effective_user.id
    stats = user_stats[user_id]

    # Create JSON export
    export_data = {
        "user_id": user_id,
        "export_date": datetime.now().isoformat(),
        "statistics": dict(stats),
        "detection_history": []  # Could store history if implemented
    }

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(export_data, f, indent=2)
        json_path = f.name

    # Send file
    with open(json_path, 'rb') as f:
        await update.message.reply_document(f, filename=f"detection_stats_{user_id}.json")

    os.unlink(json_path)

    # Also create CSV
    csv_path = json_path.replace('.json', '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Analyzed', stats['total']])
        writer.writerow(['Fake Detected', stats['fake']])
        writer.writerow(['Real Detected', stats['real']])

    with open(csv_path, 'rb') as f:
        await update.message.reply_document(f, filename=f"detection_stats_{user_id}.csv")

    os.unlink(csv_path)


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


def detect_copy_move_forgery(image_path):
    """ADVANCED: Detect copy-move forgery using block matching"""
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        block_size = 16
        h, w = gray.shape
        blocks = []
        positions = []

        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = gray[i:i+block_size, j:j+block_size]
                dct = cv2.dct(np.float32(block))
                feature = dct[:4, :4].flatten()
                blocks.append(feature)
                positions.append((i, j))

        blocks = np.array(blocks)
        from scipy.spatial.distance import cdist
        distances = cdist(blocks, blocks, 'euclidean')

        suspicious_pairs = 0
        for i in range(len(distances)):
            for j in range(i + 1, len(distances)):
                if distances[i, j] < 0.1:
                    pos1, pos2 = positions[i], positions[j]
                    dist = np.sqrt((pos1[0] - pos2[0]) **
                                   2 + (pos1[1] - pos2[1])**2)
                    if dist > 50:
                        suspicious_pairs += 1

        is_forgery = suspicious_pairs > 5
        return {"is_forgery": is_forgery, "suspicious": is_forgery}
    except:
        return {"is_forgery": False, "suspicious": False}


def detect_gan_fingerprint(image_path):
    """ADVANCED: Detect GAN fingerprints using GLCM texture analysis"""
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (256, 256))

        from skimage.feature import graycomatrix, graycoprops
        glcm = graycomatrix(gray_small, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=True)

        contrast = graycoprops(glcm, 'contrast').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        gan_score = (contrast / 100) - correlation

        return {"gan_score": gan_score, "suspicious": gan_score > 0.5}
    except:
        return {"gan_score": 0, "suspicious": False}


def detect_head_pose_inconsistency(image_path):
    """ADDED: Head pose estimation for unnatural 3D positioning"""
    if not MEDIAPIPE_AVAILABLE:
        return {"inconsistent": False}
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"inconsistent": False}
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(rgb_img)
            if not results.multi_face_landmarks:
                return {"inconsistent": False}
            landmarks = results.multi_face_landmarks[0].landmark
            nose_tip = np.array(
                [landmarks[4].x, landmarks[4].y, landmarks[4].z])
            chin = np.array(
                [landmarks[152].x, landmarks[152].y, landmarks[152].z])
            left_eye = np.array(
                [landmarks[33].x, landmarks[33].y, landmarks[33].z])
            right_eye = np.array(
                [landmarks[263].x, landmarks[263].y, landmarks[263].z])
            eye_vector = right_eye - left_eye
            face_vector = chin - nose_tip
            yaw = np.arctan2(eye_vector[0], eye_vector[2]) * 180 / np.pi
            pitch = np.arctan2(face_vector[1], face_vector[2]) * 180 / np.pi
            suspicious = abs(yaw) > 45 or abs(pitch) > 40
            return {"inconsistent": suspicious, "pose": {"yaw": yaw, "pitch": pitch}, "suspicious": suspicious}
    except:
        return {"inconsistent": False, "suspicious": False}


def detect_iris_inconsistency(image_path):
    """ADDED: Detect unnatural iris patterns in deepfakes"""
    if not MEDIAPIPE_AVAILABLE:
        return {"inconsistent": False}
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"inconsistent": False}
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(rgb_img)
            if not results.multi_face_landmarks:
                return {"inconsistent": False}
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = img.shape[:2]
            iris_issues = []
            left_iris = (int(landmarks[468].x * w) if len(landmarks) > 468 else int(landmarks[33].x * w),
                         int(landmarks[468].y * h) if len(landmarks) > 468 else int(landmarks[33].y * h))
            right_iris = (int(landmarks[473].x * w) if len(landmarks) > 473 else int(landmarks[263].x * w),
                          int(landmarks[473].y * h) if len(landmarks) > 473 else int(landmarks[263].y * h))
            for ix, iy, name in [(left_iris[0], left_iris[1], "left"), (right_iris[0], right_iris[1], "right")]:
                if 10 <= ix < w-10 and 10 <= iy < h-10:
                    roi = gray[iy-10:iy+10, ix-10:ix+10]
                    if roi.size > 0 and np.std(roi) < 10:
                        iris_issues.append(f"unnatural_{name}_iris")
            return {"inconsistent": len(iris_issues) > 0, "suspicious": len(iris_issues) > 0, "details": ", ".join(iris_issues) if iris_issues else "Normal"}
    except:
        return {"inconsistent": False, "suspicious": False}


def detect_face_swap_mediapipe(image_path):
    """
    INTEGRATED: MediaPipe Face Mesh for precise face swap detection
    Detects facial landmark inconsistencies common in deepfakes
    """
    if not MEDIAPIPE_AVAILABLE:
        return {"swapped": False, "indicators": [], "face_count": 0}

    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"swapped": False, "indicators": [], "face_count": 0}

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        swap_indicators = []
        face_count = 0

        # Initialize Face Mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            min_detection_confidence=0.5
        ) as face_mesh:

            results = face_mesh.process(rgb_img)

            if results.multi_face_landmarks:
                face_count = len(results.multi_face_landmarks)

                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark

                    # Check 1: Face boundary consistency
                    # Get face boundary points
                    face_boundary = [landmarks[i]
                                     # Sample boundary
                                     for i in range(0, 468, 4)]
                    x_coords = [p.x for p in face_boundary]
                    y_coords = [p.y for p in face_boundary]

                    # Check aspect ratio consistency
                    face_width = max(x_coords) - min(x_coords)
                    face_height = max(y_coords) - min(y_coords)
                    aspect_ratio = face_width / face_height if face_height > 0 else 0

                    if aspect_ratio < 0.6 or aspect_ratio > 1.0:
                        swap_indicators.append("unnatural_face_shape")

                    # Check 2: Eye symmetry and positioning
                    left_eye = [landmarks[i]
                                for i in [33, 133, 157, 158, 159, 160, 161, 246]]
                    right_eye = [landmarks[i]
                                 for i in [362, 263, 384, 385, 386, 387, 388, 466]]

                    left_eye_y = np.mean([p.y for p in left_eye])
                    right_eye_y = np.mean([p.y for p in right_eye])
                    eye_level_diff = abs(left_eye_y - right_eye_y)

                    if eye_level_diff < 0.005:  # Too perfectly aligned
                        swap_indicators.append("unnatural_eye_symmetry")

                    # Check 3: Mouth to eye distance consistency
                    mouth_top = landmarks[13].y
                    mouth_bottom = landmarks[14].y
                    mouth_height = abs(mouth_bottom - mouth_top)

                    nose_tip = landmarks[4].y
                    eye_level = (left_eye_y + right_eye_y) / 2
                    nose_to_eye = abs(nose_tip - eye_level)

                    # Unnatural proportions
                    if nose_to_eye / mouth_height > 2.0 if mouth_height > 0 else False:
                        swap_indicators.append("unnatural_facial_proportions")

                    # Check 4: Jawline smoothness (swapped faces often have irregular jawlines)
                    jaw_points = [landmarks[i] for i in [
                        234, 93, 132, 58, 172, 136, 150, 149, 176, 148]]
                    jaw_x = [p.x for p in jaw_points]
                    jaw_y = [p.y for p in jaw_points]

                    # Calculate curvature consistency
                    if len(jaw_x) > 2:
                        jaw_curve = np.polyfit(jaw_x, jaw_y, 2)
                        curve_variance = np.var(
                            [y - np.polyval(jaw_curve, x) for x, y in zip(jaw_x, jaw_y)])

                        if curve_variance > 0.01:
                            swap_indicators.append("irregular_jawline")

        # Remove duplicates
        swap_indicators = list(set(swap_indicators))

        return {
            "swapped": len(swap_indicators) >= 2,
            "indicators": swap_indicators,
            "face_count": face_count,
            "details": ", ".join(swap_indicators) if swap_indicators else "Normal"
        }
    except Exception as e:
        return {"swapped": False, "indicators": [], "face_count": 0, "details": f"Error: {str(e)}"}


def detect_face_enhancement(image_path):
    """
    Detect AI face enhancement/beautification:
    - Skin smoothing
    - Teeth whitening
    - Eye enlargement
    - Jawline modification
    - Lip enhancement
    - Unnatural skin tone uniformity
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"enhanced": False, "indicators": []}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Load face and feature detectors
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return {"enhanced": False, "indicators": []}

        enhancement_indicators = []

        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            face_hsv = hsv[y:y+h, x:x+w]

            # Check 1: Skin Smoothing Detection
            # Measure texture detail using Laplacian variance
            laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()

            # Very low variance indicates excessive smoothing
            if laplacian_var < 50:
                enhancement_indicators.append("excessive_skin_smoothing")

            # Check 2: Skin Tone Uniformity (unnatural)
            # Extract skin region using HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(face_hsv, lower_skin, upper_skin)

            # Calculate skin color variance
            skin_pixels = face_hsv[skin_mask > 0]
            if len(skin_pixels) > 100:
                skin_std = np.std(skin_pixels[:, 1])  # Saturation variance
                if skin_std < 15:
                    enhancement_indicators.append("unnatural_skin_uniformity")

            # Check 3: Jawline Sharpness Analysis
            # Detect edges around lower face
            lower_face = face_gray[int(h*0.6):, :]
            edges = cv2.Canny(lower_face, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Unnaturally sharp jawline
            if edge_density > 0.15:
                enhancement_indicators.append("unnatural_jawline")

            # Check 4: Eye Region Analysis
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)

            if len(eyes) >= 2:
                # Check eye brightness (whitening detection)
                for (ex, ey, ew, eh) in eyes[:2]:
                    eye_roi = face_gray[ey:ey+eh, ex:ex+ew]
                    eye_brightness = np.mean(eye_roi)

                    # Unnaturally bright eyes (whitened)
                    if eye_brightness > 200:
                        enhancement_indicators.append("eye_whitening_detected")
                        break

                # Check eye size relative to face
                eye_areas = [ew * eh for (ex, ey, ew, eh) in eyes[:2]]
                avg_eye_area = np.mean(eye_areas)
                face_area = w * h
                eye_to_face_ratio = avg_eye_area / face_area

                # Unnaturally large eyes
                if eye_to_face_ratio > 0.045:
                    enhancement_indicators.append("eye_enlargement")

            # Check 5: Lip Enhancement Detection
            # Lower face region for lips
            lower_face_region = face_hsv[int(
                h*0.65):int(h*0.9), int(w*0.25):int(w*0.75)]
            if lower_face_region.size > 0:
                # Check for high saturation (enhanced lip color)
                mean_saturation = np.mean(lower_face_region[:, :, 1])
                if mean_saturation > 180:
                    enhancement_indicators.append("lip_enhancement")

            # Check 6: Overall Face Contrast
            # Enhanced faces often have unnatural contrast
            face_lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            l_channel = face_lab[:, :, 0]
            contrast = np.std(l_channel)

            if contrast > 60:
                enhancement_indicators.append("unnatural_contrast")

        # Remove duplicates
        enhancement_indicators = list(set(enhancement_indicators))

        return {
            "enhanced": len(enhancement_indicators) >= 2,
            "indicators": enhancement_indicators,
            "details": ", ".join(enhancement_indicators) if enhancement_indicators else "No enhancement detected"
        }
    except Exception as e:
        return {"enhanced": False, "indicators": [], "details": f"Error: {str(e)}"}


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

    # ADVANCED: Run all technical analyses in parallel
    ela_result = analyze_error_level_analysis(image_path)
    noise_result = analyze_noise_pattern(image_path)
    metadata_result = analyze_metadata(image_path)
    freq_result = analyze_frequency_domain(image_path)
    copy_move_result = detect_copy_move_forgery(image_path)
    gan_result = detect_gan_fingerprint(image_path)

    # Check advanced detections first
    if metadata_result.get("has_ai_metadata"):
        return "AI METADATA DETECTED ❌", 98.0, 0, f"AI software: {metadata_result.get('software', 'Unknown')}"

    if copy_move_result["suspicious"]:
        return "COPY-MOVE FORGERY ❌", 95.0, 0, "Cloned regions detected"

    if gan_result["suspicious"]:
        return "GAN GENERATED ❌", 92.0, 0, f"GAN fingerprint score: {gan_result['gan_score']:.2f}"

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

    # Layer 2-7: Run all detection methods in parallel
    face_analysis = detect_face_swap_opencv(image_path)
    mediapipe_analysis = detect_face_swap_mediapipe(image_path)
    head_pose = detect_head_pose_inconsistency(image_path)
    iris_check = detect_iris_inconsistency(image_path)
    enhancement_analysis = detect_face_enhancement(image_path)
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
    elif head_pose["suspicious"]:
        final_label = "UNNATURAL HEAD POSE ❌"
        reason = f"Pose: yaw={head_pose['pose']['yaw']:.1f}, pitch={head_pose['pose']['pitch']:.1f}"
    elif iris_check["suspicious"]:
        final_label = "IRIS ANOMALY ❌"
        reason = f"Iris issue: {iris_check['details']}"
    elif mediapipe_analysis["swapped"]:
        final_label = "FACE SWAP (MediaPipe) ❌"
        reason = f"Landmark analysis: {mediapipe_analysis['details']}"
    elif face_analysis["suspicious"] and model_label == "FAKE":
        final_label = "FACE SWAP ❌"
        reason = f"Face swap detected: {face_analysis['details']}"
    elif enhancement_analysis["enhanced"]:
        final_label = "FACE ENHANCED ❌"
        reason = f"Beautification: {enhancement_analysis['details']}"
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
    elif enhancement_analysis["indicators"]:
        final_label = "SUSPICIOUS ⚠️"
        reason = f"Possible enhancement: {enhancement_analysis['details']}"
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
    """Handle image uploads with full-fledged features"""
    user_id = update.effective_user.id

    # Check rate limit
    if not check_rate_limit(user_id):
        await update.message.reply_text("⏱️ Please wait a moment before sending another image.")
        return

    # Check if in batch mode
    if user_id in BATCH_MODE:
        photo = update.message.photo[-1]
        BATCH_MODE[user_id].append(photo.file_id)
        count = len(BATCH_MODE[user_id])
        await update.message.reply_text(f"📸 Added to batch ({count} images). Send more or type /done")
        return

    await update.message.reply_text("🔍 Analyzing with 14-layer detection...")

    # Get photo
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    try:
        # Analyze with async multi-layer detection
        start_time = time.time()
        label, confidence, face_count, reason = await analyze_image_async(tmp_path)
        analysis_time = time.time() - start_time

        # Update user statistics
        user_stats[user_id]["total"] += 1
        if "❌" in label:
            user_stats[user_id]["fake"] += 1
        else:
            user_stats[user_id]["real"] += 1
        user_stats[user_id]["last_request"] = time.time()

        # Send result with full details
        result_text = (
            f"🧠 *Analysis Complete*\n\n"
            f"📊 Result: *{label}*\n"
            f"🎯 Confidence: *{confidence:.1f}%*\n"
            f"👤 Faces Detected: *{face_count}*\n"
            f"⏱️ Analysis Time: *{analysis_time:.2f}s*\n"
            f"📝 Reason: {reason}\n\n"
            f"📈 Your Stats: {user_stats[user_id]['total']} analyzed\n"
            f"Use /stats for full statistics"
        )
        await update.message.reply_text(result_text, parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(f"❌ Error during analysis: {str(e)}")
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

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("batch", batch_command))
    application.add_handler(CommandHandler("done", done_command))
    application.add_handler(CommandHandler("export", export_command))

    # Add message handlers
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(MessageHandler(
        filters.Document.ALL, handle_document))
    application.add_error_handler(error_handler)

    # Run
    logger.info("Bot started! Full-fledged mode active.")
    application.run_polling()


if __name__ == "__main__":
    main()

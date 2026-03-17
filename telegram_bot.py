import asyncio
import logging
import os
import tempfile
from io import BytesIO

import torch
import cv2
from PIL import Image
from torchvision.models import efficientnet_b0
from torchvision import transforms
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Bot token
TOKEN = "5614405588:AAEtmjQNR8cppePAxUxRIlcYzDOK4Y11ghc"

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


def analyze_image(image_path):
    """Analyze image for deepfake detection"""
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)

    is_real = pred.item() == 0
    label = "REAL ✅" if is_real else "DEEPFAKE ❌"
    confidence = conf.item() * 100

    return label, confidence


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
    """Handle image uploads"""
    await update.message.reply_text("🔍 Analyzing image...")

    # Get photo
    photo = update.message.photo[-1]  # Get highest quality
    file = await context.bot.get_file(photo.file_id)

    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    try:
        # Analyze
        label, confidence = analyze_image(tmp_path)

        # Send result
        result_text = (
            f"🧠 *Analysis Result*\n\n"
            f"📊 Prediction: *{label}*\n"
            f"🎯 Confidence: *{confidence:.1f}%*\n\n"
            f"Powered by EfficientNet-B0"
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

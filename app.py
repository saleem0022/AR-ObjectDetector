import base64
import io
import logging
import time
import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import torch
import clip
from openai import OpenAI

# ======================
# INITIALIZATION
# ======================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories for debugging
os.makedirs('received_images', exist_ok=True)
os.makedirs('processed_images', exist_ok=True)

# ======================
# MODEL SETUP
# ======================

def initialize_models():
    """Initialize and verify all AI models"""
    try:
        # Initialize CLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing CLIP on {device}")
        
        model, preprocess = clip.load("ViT-B/32", device=device)
        logger.info("CLIP model loaded successfully")
        
        # Verify CLIP works
        test_tensor = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model.encode_image(test_tensor)
            logger.info(f"CLIP test output shape: {output.shape}")
        
        # Initialize OpenAI
       # paste your key here
        logger.info("OpenAI client initialized")
        
        return device, model, preprocess, openai_client
    
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

# ======================
# IMAGE PROCESSING
# ======================

def process_image(image_bytes, save_path=None):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if save_path:
            image.save(f"received_images/{save_path}.jpg")

        logger.info(f"Original image: {image.size}, mode: {image.mode}")
        img_array = np.array(image)
        logger.info(f"Pixel range: {img_array.min()} - {img_array.max()}")

        processed = preprocess(image).unsqueeze(0).to(device)

        logger.info(f"Processed tensor shape: {processed.shape}")
        logger.info(f"Processed range: {processed.min().item():.2f} - {processed.max().item():.2f}")

        # Save processed version for debugging
        if save_path:
            vis_array = (processed.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            processed_img = Image.fromarray(vis_array)
            processed_img.save(f"processed_images/{save_path}.jpg")

        return processed

    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise


# ======================
# LABELS & INFERENCE
# ======================

# Enhanced label set with common objects
LABELS = [
    # Kitchen items
    "pan", "pot", "plate", "bowl", "cup", "glass", "fork", "knife", "spoon",
    "cutting board", "microwave", "refrigerator", "oven", "toaster", "kettle",
    "blender", "mug", "stove", "frying pan", "kitchen sink", "dish rack",
    "coffee maker", "bottle", "measuring cup", "colander", "grater", "can opener",

    # Office items
    "laptop", "keyboard", "mouse", "monitor", "phone", "tablet", 
    "notebook", "pen", "pencil", "book", "folder", "desk lamp", 
    "stapler", "tape dispenser", "printer", "scanner", "calculator", "paper clip",
    "whiteboard", "marker", "calendar", "file cabinet", "envelope",

    # Furniture
    "chair", "table", "desk", "sofa", "bed", "shelf", "cabinet", 
    "bench", "dresser", "nightstand", "coffee table", "bookshelf",
    "wardrobe", "tv stand", "cushion", "mirror",

    # Electronics
    "television", "remote", "speaker", "headphones", "camera",
    "smartphone", "smartwatch", "router", "light switch", "fan",
    "air conditioner", "game controller", "VR headset", "electric bulb",

    # Bathroom items
    "toilet", "sink", "mirror", "toothbrush", "toothpaste", "soap",
    "shampoo", "towel", "razor", "hair dryer", "shower head", "bath mat",

    # Personal accessories
    "glasses", "wallet", "bag", "backpack", "hat", "shoe", "jacket",
    "watch", "scarf", "ring", "bracelet", "necklace", "umbrella",

    # Tools & Hardware
    "hammer", "screwdriver", "wrench", "pliers", "drill", "saw",
    "tape measure", "ladder", "toolbox", "nail", "bolt", "paintbrush",

    # Food items
    "apple", "banana", "orange", "grape", "watermelon", "bread",
    "egg", "cheese", "milk", "butter", "yogurt", "pizza", "burger",
    "sandwich", "cake", "cookie", "chocolate", "carrot", "broccoli",

    # Outdoor & nature
    "tree", "flower", "grass", "rock", "sky", "cloud", "sun",
    "bicycle", "car", "truck", "motorcycle", "bus", "traffic light",
    "bench", "sign", "fence", "building", "road", "sidewalk",

    # Animals (common)
    "dog", "cat", "bird", "cow", "horse", "sheep", "goat", "duck",
    "chicken", "rabbit", "fish", "squirrel", "pig", "turtle",

    # Miscellaneous
    "plant", "lamp", "clock", "picture", "window", "door", "curtain",
    "person", "ball", "broom", "bucket", "suitcase", "shopping bag",
    "glove", "mask", "bottle", "can", "box", "gift", "tissue box"
]


def run_inference(image_tensor):
    """Run CLIP inference with enhanced validation"""
    try:
        # Prepare text inputs
        text_inputs = torch.cat([clip.tokenize(label) for label in LABELS]).to(device)
        
        # Run model
        with torch.no_grad():
            logits_per_image, _ = model(image_tensor, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
        # Get detailed predictions
        predictions = sorted(zip(LABELS, probs), key=lambda x: -x[1])
        
        # Log all predictions
        logger.info("All predictions:")
        for i, (label, prob) in enumerate(predictions[:10]):  # Top 10
            logger.info(f"{i+1:2d}. {label:15s}: {prob:.4f}")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise

# ======================
# FLASK APP
# ======================

device, model, preprocess, openai_client = initialize_models()
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

@app.route('/explain', methods=['POST'])
def explain():
    """Main endpoint for image explanation"""
    start_time = time.time()
    request_id = int(time.time() * 1000)
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        logger.info(f"\n=== Processing request {request_id} ===")
        
        # Process image
        try:
            image_bytes = base64.b64decode(data['image'])
            image_tensor = process_image(image_bytes, save_path=f"req_{request_id}")
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return jsonify({"error": "Invalid image data"}), 400
        
        # Run inference
        try:
            predictions = run_inference(image_tensor)
            top_label, top_confidence = predictions[0]
            
            logger.info(f"Top prediction: {top_label} ({top_confidence:.2%})")
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return jsonify({"error": "Model inference failed"}), 500
        
        # Generate explanation
        try:
            prompt = (
                f"The image shows a {top_label}. "
                f"Explain what this is and its typical uses in simple terms. "
                f"Keep the response under 100 words."
            )
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content
            logger.info("Explanation generated successfully")
        except Exception as e:
            logger.warning(f"OpenAI failed, using fallback explanation: {str(e)}")
            explanation = f"This appears to be a {top_label}."
        
        # Prepare response
        response_data = {
            "request_id": request_id,
            "label": top_label,
            "confidence": float(top_confidence),
            "explanation": explanation,
            "processing_time": round(time.time() - start_time, 2),
            "alternative_predictions": [
                {"label": label, "confidence": float(conf)}
                for label, conf in predictions[1:5]  # Next 4 predictions
            ]
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "CLIP ViT-B/32",
        "device": device,
        "labels_loaded": len(LABELS)
    })

if __name__ == '__main__':
    try:
        logger.info(" Starting server with HTTPS...")
        app.run(
            host='192.168.0.103',
            port=8000,
            ssl_context=('cert.pem', 'key.pem'),
            threaded=True
        )
    except Exception as e:
        logger.error(f"Server start failed: {str(e)}")
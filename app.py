import os
import json
import uuid
import traceback
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────────────
# Flask setup
# ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
HEATMAP_FOLDER= os.path.join(BASE_DIR, "static", "heatmaps")
HISTORY_FILE  = os.path.join(BASE_DIR, "scan_history.json")
ALLOWED_EXT   = {"png", "jpg", "jpeg", "heic", "heif", "webp"}

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB

# ─────────────────────────────────────────────────────────────────────
# Treatment data  (inline — no external file dependency)
# ─────────────────────────────────────────────────────────────────────
TREATMENTS = {
    "mel":   ["Immediate dermatologist referral", "Surgical excision",
              "Sentinel lymph node biopsy", "Immunotherapy if advanced"],
    "bcc":   ["Mohs micrographic surgery", "Surgical excision",
              "Topical imiquimod", "Photodynamic therapy"],
    "akiec": ["Cryotherapy", "Topical 5-fluorouracil",
              "Photodynamic therapy", "Regular monitoring"],
    "bkl":   ["Usually no treatment needed", "Cryotherapy if bothersome",
              "Curettage", "Laser therapy"],
    "df":    ["Observation only", "Surgical excision if symptomatic",
              "Cryotherapy", "Laser resurfacing"],
    "nv":    ["Regular monitoring for changes", "Dermoscopy follow-up",
              "Excision if atypical features", "Sun protection advised"],
    "vasc":  ["Observation", "Laser therapy",
              "Sclerotherapy", "Surgical removal if required"],
}

# Try to load from treatment.json if it exists — merge/override inline data
_json_path = os.path.join(BASE_DIR, "treatment.json")
if os.path.exists(_json_path):
    try:
        with open(_json_path) as _f:
            _json_data = json.load(_f)
        # treatment.json may store {"mel": [...]} or {"mel": {"treatment": [...]}}
        for _k, _v in _json_data.items():
            if isinstance(_v, list):
                TREATMENTS[_k] = _v
            elif isinstance(_v, dict) and "treatment" in _v:
                TREATMENTS[_k] = _v["treatment"]
    except Exception as _e:
        print(f"[WARN] Could not load treatment.json: {_e} — using built-in data")

MALIGNANT = {"mel", "bcc", "akiec"}

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def allowed_file(filename, mimetype=""):
    # Check by extension
    if "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()
        if ext in ALLOWED_EXT:
            return True
    # Accept HEIC/HEIF by MIME (iPhone)
    if mimetype in ("image/heic", "image/heif"):
        return True
    # Accept jpg/png/webp by MIME
    if mimetype in ("image/jpeg", "image/jpg", "image/png", "image/webp"):
        return True
    return False

def load_history():
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []

def save_history(entry):
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def make_url_path(abs_or_rel_path):
    """Convert any file path to a URL-safe relative path."""
    path = str(abs_or_rel_path).replace("\\", "/")
    # Strip leading drive letters / slashes
    if "static/" in path:
        path = path[path.index("static/"):]
    elif "uploads/" in path:
        path = path[path.index("uploads/"):]
    return path.lstrip("/")

# ─────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/static/<path:filename>")
def serve_static_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)


@app.route("/predict", methods=["POST"])
def predict():

    # ── 1. Validate upload ───────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]

    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename, file.mimetype or ""):
        return jsonify({"error": "Only JPG, PNG, HEIC and WebP files are accepted"}), 400

    # ── 2. Save file ─────────────────────────────────────────────────
    # Get extension — default to jpg if missing (camera uploads)
    if "." in file.filename:
        ext = file.filename.rsplit(".", 1)[1].lower()
        # Normalize jfif to jpg (PIL handles it fine)
        if ext in ("jfif", "jpe"):
            ext = "jpg"
    else:
        ext = "jpg"
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # ── 3. Run model ─────────────────────────────────────────────────
    try:
        from predict import predict_image
        raw_preds = predict_image(filepath)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # raw_preds = [{"disease": "mel", "confidence": 92.5, "risk": "Critical"}, ...]

    # ── 4. Build response ────────────────────────────────────────────
    top_predictions = []
    for p in raw_preds:
        cls  = p.get("disease") or p.get("class") or "unknown"
        conf = float(p.get("confidence", 0))

        # Normalize: predict.py returns percentage (e.g. 92.5), we need 0–1
        if conf > 1.5:
            conf = conf / 100.0

        treatment_list = TREATMENTS.get(cls, ["Consult a dermatologist"])

        top_predictions.append({
            "class":      cls,
            "confidence": round(conf, 4),
            "treatment":  {"treatment": treatment_list}
        })

    # ── 5. GradCAM ───────────────────────────────────────────────────
    heatmap_url = None
    try:
        from gradcam import generate_gradcam
        heatmap_path = generate_gradcam(filepath)
        heatmap_url  = make_url_path(heatmap_path)
    except Exception as e:
        traceback.print_exc()
        print(f"[WARN] GradCAM failed: {e} — continuing without heatmap")

    # ── 6. Warnings ──────────────────────────────────────────────────
    warnings = []
    if top_predictions:
        top_cls  = top_predictions[0]["class"]
        top_conf = top_predictions[0]["confidence"]

        if top_conf < 0.60:
            warnings.append(
                f"Low confidence ({round(top_conf*100, 1)}%) — "
                "image may be unclear. Consider retaking."
            )
        if top_cls in MALIGNANT and top_conf > 0.40:
            warnings.append(
                f"{top_cls.upper()} detected with {round(top_conf*100,1)}% confidence. "
                "Seek immediate dermatologist consultation."
            )

    # ── 7. Save history ──────────────────────────────────────────────
    scan_id = str(uuid.uuid4())
    top_cls  = top_predictions[0]["class"]  if top_predictions else "unknown"
    top_conf = top_predictions[0]["confidence"] if top_predictions else 0.0

    save_history({
        "id":             scan_id,
        "timestamp":      str(datetime.now()),
        "image":          make_url_path(filepath),
        "heatmap":        heatmap_url,
        "top_prediction": top_cls,
        "confidence":     top_conf,
        "warnings":       warnings,
    })

    # ── 8. Return ────────────────────────────────────────────────────
    return jsonify({
        "id":              scan_id,
        "top_predictions": top_predictions,
        "heatmap":         heatmap_url,
        "warnings":        warnings,
        "disclaimer":      (
            "This AI tool provides informational analysis only "
            "and is not a substitute for professional medical diagnosis."
        )
    })


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(load_history())


@app.route("/history/all", methods=["DELETE"])
def delete_all_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)
    return jsonify({"success": True})


@app.route("/history/<scan_id>", methods=["DELETE"])
def delete_one(scan_id):
    data     = load_history()
    filtered = [s for s in data if str(s.get("id", "")) != str(scan_id)]
    if len(filtered) == len(data):
        return jsonify({"error": "Scan not found"}), 404
    with open(HISTORY_FILE, "w") as f:
        json.dump(filtered, f, indent=4)
    return jsonify({"success": True})


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n✓ SkinAid backend starting...")
    print(f"  Base dir    : {BASE_DIR}")
    print(f"  Upload dir  : {UPLOAD_FOLDER}")
    print(f"  Heatmap dir : {HEATMAP_FOLDER}")
    print(f"  History file: {HISTORY_FILE}")
    print(f"  Treatment   : {len(TREATMENTS)} conditions loaded\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
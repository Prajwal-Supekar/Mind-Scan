"""
MindScan — Depression & Mental Health Analysis System
Fixed app.py:
  BUG 1: model.load_weights('D:/depresion level/model.h5') → relative path
  BUG 2: Emotion detection used server webcam → now browser-sends frames via fetch
  BUG 3: All heavy models lazy-loaded so app starts even if model files missing
  BUG 4: SocketIO set to eventlet for gunicorn compatibility
  BUG 5: Added global error handlers returning HTML error page
  BUG 6: Sentiment result now stored for result page display
"""

import os, random, hashlib, base64, warnings
import sqlite3, pickle
import numpy as np
from collections import deque
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'mindscan-2024-secret')
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins='*')

# ── Global session state ──────────────────────────────────────────────────────
loggedin_user   = {}
depression_result = None
sentiment_result  = None
emotion_result    = None

# ── Therapy questions ─────────────────────────────────────────────────────────
QUESTIONS = [
    "How are you feeling today?",
    "What is something that made you happy recently?",
    "Can you describe a recent challenge you faced?",
    "What are your goals for the future?",
    "Tell me about your favorite hobby.",
    "What do you do when you're feeling stressed?",
    "What is something you are grateful for?",
    "Describe a memorable moment from this week.",
    "How would you describe your energy levels lately?",
    "What helps you feel calm or at peace?",
]

# ── Lazy model loaders (app starts even if model files are missing) ────────────
_depression_model = None
_scaler           = None
_sentiment_pipe   = None
_emotion_model    = None
_face_cascade     = None


def get_depression_model():
    global _depression_model, _scaler
    if _depression_model is None:
        with open("depression_model.pkl", "rb") as f:
            _depression_model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            _scaler = pickle.load(f)
    return _depression_model, _scaler


def get_sentiment_pipeline():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        from transformers import pipeline
        _sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _sentiment_pipe


def get_emotion_model():
    """
    BUG FIX: Load model with relative path ('model.h5') and only ONCE.
    Original code rebuilt the entire Sequential model on every webcam frame request.
    """
    global _emotion_model, _face_cascade
    if _emotion_model is None:
        try:
            import cv2
            import tensorflow as tf
            from tensorflow import keras

            model = keras.Sequential([
                keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
                keras.layers.Conv2D(64, (3,3), activation='relu'),
                keras.layers.MaxPooling2D(2,2),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(128, (3,3), activation='relu'),
                keras.layers.MaxPooling2D(2,2),
                keras.layers.Conv2D(128, (3,3), activation='relu'),
                keras.layers.MaxPooling2D(2,2),
                keras.layers.Dropout(0.25),
                keras.layers.Flatten(),
                keras.layers.Dense(1024, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(7, activation='softmax'),
            ])
            # BUG FIX: was 'D:/depresion level/model.h5' — now relative
            model.load_weights('model.h5')
            _emotion_model = model

            _face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("[MindScan] Emotion model loaded.")
        except Exception as e:
            print(f"[MindScan] Emotion model unavailable: {e}")
    return _emotion_model, _face_cascade


# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('users.db')
    conn.cursor().execute('''
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT UNIQUE,
            password_hash TEXT,
            email        VARCHAR,
            phone_no     INTEGER,
            R_address    VARCHAR(255),
            gender       VARCHAR,
            age          INTEGER,
            dob          DATE
        )''')
    conn.commit()
    conn.close()

init_db()


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', message="Page not found.", code=404), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message="Internal server error.", code=500), 500


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def login():
    global loggedin_user
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['U_password']
            hashed   = hashlib.sha256(password.encode()).hexdigest()

            conn = sqlite3.connect('users.db')
            c    = conn.cursor()
            c.execute("SELECT * FROM users WHERE username=? AND password_hash=?",
                      (username, hashed))
            user = c.fetchone()
            conn.close()

            if user:
                loggedin_user = {
                    'id': user[0], 'username': user[1],
                    'email': user[3], 'phone_no': user[4],
                    'address': user[5], 'gender': user[6],
                    'age': user[7], 'dob': user[8],
                }
                return redirect('/index')
            return render_template('login1.html', error='Invalid username or password.')
        except Exception as e:
            return render_template('error.html', message=str(e), code=500)
    return render_template('login1.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            pw_hash = hashlib.sha256(request.form['U_password'].encode()).hexdigest()
            conn = sqlite3.connect('users.db')
            conn.cursor().execute(
                '''INSERT INTO users (username, password_hash, email, phone_no,
                   R_address, gender, age, dob) VALUES (?,?,?,?,?,?,?,?)''',
                (request.form['username'], pw_hash, request.form['email'],
                 request.form['phone_no'], request.form['R_address'],
                 request.form['gender'], request.form['age'], request.form['dob'])
            )
            conn.commit()
            conn.close()
            return "Registration successful!", 200
        except sqlite3.IntegrityError:
            return "Username already exists.", 409
        except Exception as e:
            return f"Error: {str(e)}", 500
    return render_template('register1.html')


@app.route('/logout')
def logout():
    global loggedin_user, depression_result, sentiment_result, emotion_result
    loggedin_user = {}
    depression_result = sentiment_result = emotion_result = None
    return redirect(url_for('login'))


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route('/index')
def index():
    return render_template('index.html', user=loggedin_user)


# ── Depression ────────────────────────────────────────────────────────────────
@app.route('/depression')
def depression():
    return render_template('depression.html')


@app.route('/predict', methods=['POST'])
def predict():
    global depression_result
    try:
        data = [
            float(request.form['age']),
            int(request.form['gender']),
            int(request.form['education']),
            int(request.form['employment']),
            int(request.form['marital_status']),
            int(request.form['history_of_depression']),
            int(request.form['family_history']),
            int(request.form['past_treatments']),
            float(request.form['sleep_duration']),
            float(request.form['physical_activity']),
            int(request.form['diet']),
            int(request.form['substance_use']),
            int(request.form['stress_level']),
            int(request.form['anxiety_level']),
            int(request.form['mood_variations']),
            int(request.form['social_support']),
            int(request.form['isolation_frequency']),
            int(request.form['relationship_satisfaction']),
        ]
        mdl, scl = get_depression_model()
        pred = mdl.predict(scl.transform([data]))[0]
        result = "At Risk of Depression" if pred == 1 else "Not At Risk of Depression"
        depression_result = result
        return render_template('depression.html', prediction_text=result)
    except Exception as e:
        return render_template('depression.html', error_text=str(e))


# ── Sentiment ─────────────────────────────────────────────────────────────────
@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html', question=random.choice(QUESTIONS))


@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    global sentiment_result
    text = request.form.get('text', '').strip()
    if not text:
        return jsonify([{'label': 'NEUTRAL', 'score': 0.5}])
    try:
        pipe   = get_sentiment_pipeline()
        result = pipe(text)
        sentiment_result = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_next_question')
def get_next_question():
    return jsonify({'question': random.choice(QUESTIONS)})


# ── Emotion ───────────────────────────────────────────────────────────────────
@app.route('/emotion')
def emotion():
    return render_template('emotion.html')


@app.route('/detect_emotion_frame', methods=['POST'])
def detect_emotion_frame():
    """
    BUG FIX: Original used server webcam (cv2.VideoCapture(0)) which
    never works on a remote server. This endpoint receives a single frame
    as base64 from the user's browser webcam, processes it, and returns
    the detected emotion. Works on any deployment including Render.
    """
    global emotion_result
    try:
        import cv2
        data_json = request.get_json(silent=True) or {}
        frame_b64 = data_json.get('frame', '')
        if not frame_b64:
            return jsonify({'emotion': 'No frame', 'confidence': 0, 'faces': []})

        # Strip data URL prefix if present
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',', 1)[1]

        img_bytes = base64.b64decode(frame_b64)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'emotion': 'Invalid frame', 'confidence': 0, 'faces': []})

        mdl, cascade = get_emotion_model()
        if mdl is None or cascade is None:
            return jsonify({'emotion': 'Model unavailable', 'confidence': 0, 'faces': []})

        EMOTION_DICT = {0:"Angry",1:"Disgusted",2:"Fearful",
                        3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        face_results   = []
        top_emotion    = "No Face Detected"
        top_confidence = 0.0
        fh, fw         = frame.shape[:2]

        for (x, y, w, h) in faces:
            roi    = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            inp    = np.expand_dims(np.expand_dims(roi, -1), 0)
            preds  = mdl.predict(inp, verbose=0)
            idx    = int(np.argmax(preds))
            conf   = float(np.max(preds))
            top_emotion    = EMOTION_DICT[idx]
            top_confidence = conf
            face_results.append({
                'x': x/fw, 'y': y/fh, 'w': w/fw, 'h': h/fh,
                'emotion': top_emotion, 'confidence': round(conf*100, 1)
            })

        emotion_result = top_emotion
        return jsonify({
            'emotion':    top_emotion,
            'confidence': round(top_confidence * 100, 1),
            'faces':      face_results,
        })

    except Exception as e:
        app.logger.error(f"Emotion frame error: {e}", exc_info=True)
        return jsonify({'emotion': 'Error', 'confidence': 0,
                        'faces': [], 'error': str(e)})


# ── Result ────────────────────────────────────────────────────────────────────
@app.route('/result')
def result():
    return render_template('result.html',
                           user=loggedin_user,
                           depression=depression_result,
                           sentiment=sentiment_result,
                           emotion=emotion_result)


if __name__ == '__main__':
    socketio.run(app, debug=True)

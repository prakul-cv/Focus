import cv2
import time
import sys
import json
import os
import threading
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from collections import deque
from datetime import datetime

app = Flask(__name__)

# ─── Load Cascades ───────────────────────────────────────────────
cascade_path = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(cascade_path + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cascade_path + "haarcascade_eye.xml")

if face_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load cascade classifiers.")
    sys.exit(1)

# ─── Settings ────────────────────────────────────────────────────
settings = {
    "show_face_detection": True,
    "show_eye_detection":  True,
    "show_overlay_info":   True,
    "fps_limit":           15,
    "notify_when_away":    True,
    "away_threshold":      15,
    "theme":               "dark",
}

# ─── Global Session State ────────────────────────────────────────
session = {
    "running":           False,
    "start_time":        None,
    "started_at_str":    None,
    "total_frames":      0,
    "focused_frames":    0,
    "distracted_frames": 0,
    "away_frames":       0,
    "focus_state":       "Offline",
    "focus_score":       0.0,
    "longest_streak":    0,
    "current_streak":    0,
    "last_focus_time":   time.time(),
    "history":           deque(maxlen=150),
    "recent_activity":   deque(maxlen=20),
    "frame_times":       deque(maxlen=30),
    "avg_fps":           0.0,
}
lock = threading.Lock()
cam  = None

# ─── Session History (persisted in memory) ───────────────────────
HISTORY_FILE = "session_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return []

def save_session_to_history(sess_data):
    history = load_history()
    history.insert(0, sess_data)
    history = history[:50]  # keep last 50 sessions
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)
    except:
        pass

# ─── Blank offline frame ─────────────────────────────────────────
def make_blank_frame():
    blank = np.ones((480, 640, 3), dtype=np.uint8) * 18
    cv2.putText(blank, "CAMERA OFFLINE", (155, 220),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (60, 100, 140), 2)
    cv2.putText(blank, "Press START SESSION to begin", (130, 260),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (40, 70, 100), 1)
    _, buf = cv2.imencode('.jpg', blank)
    return buf.tobytes()

# ─── MJPEG Frame Generator ───────────────────────────────────────
def generate_frames():
    global cam
    while True:
        with lock:
            running = session["running"]

        if not running or cam is None:
            frame_bytes = make_blank_frame()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + frame_bytes + b'\r\n')
            time.sleep(0.1)
            continue

        with lock:
            cam_ref = cam
            fps_limit = settings["fps_limit"]
            show_face = settings["show_face_detection"]
            show_eye  = settings["show_eye_detection"]
            show_overlay = settings["show_overlay_info"]

        if cam_ref is None:
            time.sleep(0.05)
            continue

        frame_start = time.time()
        ret, frame = cam_ref.read()
        if not ret:
            time.sleep(0.05)
            continue

        try:
            frame = cv2.resize(frame, (640, 480))
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            focus_state = "Away"
            spark_val   = 0

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    if show_face:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 80, 255), 2)
                    roi_gray  = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7)

                    if len(eyes) > 0:
                        focus_state = "Focused"
                        spark_val   = 1
                        if show_eye:
                            for (ex, ey, ew, eh) in eyes:
                                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 130), 1)
                    else:
                        focus_state = "Distracted"
                        spark_val   = 0.5
            else:
                focus_state = "Away"
                spark_val   = 0

            # Update session stats
            now_str = datetime.now().strftime("%I:%M:%S %p")
            frame_end = time.time()
            frame_time = frame_end - frame_start

            with lock:
                session["total_frames"] += 1
                session["focus_state"]   = focus_state
                session["history"].append(spark_val)
                session["frame_times"].append(frame_time)

                if len(session["frame_times"]) > 1:
                    avg_t = sum(session["frame_times"]) / len(session["frame_times"])
                    session["avg_fps"] = round(1.0 / avg_t, 1) if avg_t > 0 else 0

                prev_state = session["recent_activity"][-1]["state"] if session["recent_activity"] else None
                if focus_state != prev_state:
                    session["recent_activity"].append({
                        "time":  now_str,
                        "state": focus_state,
                    })

                if focus_state == "Focused":
                    session["focused_frames"]    += 1
                    session["last_focus_time"]    = time.time()
                    session["current_streak"]    += 1
                    if session["current_streak"] > session["longest_streak"]:
                        session["longest_streak"] = session["current_streak"]
                elif focus_state == "Distracted":
                    session["distracted_frames"] += 1
                    session["current_streak"]     = 0
                else:
                    session["away_frames"]        += 1
                    session["current_streak"]     = 0

                total = session["total_frames"] or 1
                session["focus_score"] = (session["focused_frames"] / total) * 100

            # ── Overlay ───────────────────────────────────────────
            if show_overlay:
                color = ((0, 255, 130)  if focus_state == "Focused"
                         else (0, 200, 255) if focus_state == "Distracted"
                         else (60, 60, 255))

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (640, 48), (8, 12, 28), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                elapsed = int(time.time() - session["start_time"]) if session["start_time"] else 0
                score   = session["focus_score"]

                cv2.putText(frame, f"{focus_state.upper()}", (12, 32),
                            cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2)
                cv2.putText(frame, f"SCORE: {score:.1f}%", (420, 32),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 200, 255), 2)
                cv2.putText(frame, f"T: {elapsed//60:02d}:{elapsed%60:02d}", (268, 32),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (140, 140, 200), 1)

            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')

            # FPS throttle
            elapsed_frame = time.time() - frame_start
            sleep_time = (1.0 / fps_limit) - elapsed_frame
            if sleep_time > 0:
                time.sleep(sleep_time)

        except Exception as e:
            print(f"[Frame Error] {e}")
            time.sleep(0.05)
            continue


# ─── Routes ──────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/start', methods=['POST'])
def start():
    global cam
    try:
        with lock:
            if session["running"]:
                return jsonify({"status": "already_running"})

            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                cam = None
                return jsonify({
                    "status": "error",
                    "message": "Cannot open camera. Make sure it is connected and not in use."
                }), 500

            now = datetime.now()
            session["running"]           = True
            session["start_time"]        = time.time()
            session["started_at_str"]    = now.strftime("%I:%M:%S %p")
            session["total_frames"]      = 0
            session["focused_frames"]    = 0
            session["distracted_frames"] = 0
            session["away_frames"]       = 0
            session["focus_state"]       = "Initializing"
            session["focus_score"]       = 0.0
            session["longest_streak"]    = 0
            session["current_streak"]    = 0
            session["last_focus_time"]   = time.time()
            session["history"]           = deque(maxlen=150)
            session["recent_activity"]   = deque(maxlen=20)
            session["frame_times"]       = deque(maxlen=30)
            session["avg_fps"]           = 0.0

        return jsonify({"status": "started"})

    except Exception as e:
        print(f"[Start Error] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop():
    global cam
    try:
        with lock:
            if not session["running"]:
                return jsonify({"status": "not_running"})

            session["running"]     = False
            session["focus_state"] = "Offline"

            # Build summary for history
            elapsed = int(time.time() - session["start_time"]) if session["start_time"] else 0
            hrs,  rem  = divmod(elapsed, 3600)
            mins, secs = divmod(rem, 60)
            duration_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
            total = session["total_frames"] or 1

            # Longest streak in mm:ss (estimate: streak * ~0.1s per frame / 60)
            streak_secs = int(session["longest_streak"] * (1.0 / max(session["avg_fps"], 1)))
            sm, ss = divmod(streak_secs, 60)
            streak_str = f"{sm:02d}:{ss:02d}"

            summary = {
                "date":     datetime.now().strftime("%b %d, %Y"),
                "duration": duration_str,
                "score":    round(session["focus_score"], 1),
                "focused":  round((session["focused_frames"] / total) * 100, 1),
                "distracted": round((session["distracted_frames"] / total) * 100, 1),
                "away":     round((session["away_frames"] / total) * 100, 1),
                "longest":  streak_str,
                "total_frames": session["total_frames"],
                "avg_fps":  session["avg_fps"],
                "started_at": session["started_at_str"],
            }

            if cam is not None:
                cam.release()
                cam = None

        save_session_to_history(summary)
        return jsonify({"status": "stopped", "summary": summary})

    except Exception as e:
        print(f"[Stop Error] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stats')
def stats():
    try:
        with lock:
            total      = session["total_frames"] or 1
            elapsed    = int(time.time() - session["start_time"]) if session["start_time"] else 0
            hrs,  rem  = divmod(elapsed, 3600)
            mins, secs = divmod(rem, 60)
            duration   = f"{hrs:02d}:{mins:02d}:{secs:02d}"

            return jsonify({
                "running":    session["running"],
                "state":      session["focus_state"],
                "score":      round(session["focus_score"], 1),
                "duration":   duration,
                "started_at": session["started_at_str"],
                "total":      session["total_frames"],
                "focused":    session["focused_frames"],
                "distracted": session["distracted_frames"],
                "away":       session["away_frames"],
                "f_pct":      round((session["focused_frames"]    / total) * 100, 1),
                "d_pct":      round((session["distracted_frames"] / total) * 100, 1),
                "a_pct":      round((session["away_frames"]       / total) * 100, 1),
                "longest":    session["longest_streak"],
                "history":    list(session["history"]),
                "activity":   list(session["recent_activity"]),
                "avg_fps":    session["avg_fps"],
            })
    except Exception as e:
        print(f"[Stats Error] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/history')
def get_history():
    try:
        page = int(request.args.get("page", 1))
        per_page = 8
        all_history = load_history()
        start = (page - 1) * per_page
        end   = start + per_page
        return jsonify({
            "sessions": all_history[start:end],
            "total":    len(all_history),
            "page":     page,
            "has_more": end < len(all_history),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/settings', methods=['GET'])
def get_settings():
    return jsonify(settings)


@app.route('/settings', methods=['POST'])
def update_settings():
    try:
        data = request.get_json()
        with lock:
            for key in settings:
                if key in data:
                    settings[key] = data[key]
        return jsonify({"status": "ok", "settings": settings})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/settings/reset', methods=['POST'])
def reset_settings():
    global settings
    with lock:
        settings = {
            "show_face_detection": True,
            "show_eye_detection":  True,
            "show_overlay_info":   True,
            "fps_limit":           15,
            "notify_when_away":    True,
            "away_threshold":      15,
            "theme":               "dark",
        }
    return jsonify({"status": "ok", "settings": settings})


# ─── Run ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 45)
    print("  Focus Monitor running at:")
    print("  http://localhost:5000")
    print("  Press CTRL+C to stop")
    print("=" * 45)
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)

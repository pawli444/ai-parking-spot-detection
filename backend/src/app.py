import os
import threading
import uuid
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import shutil
import subprocess

from process_video import process_video

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

DEFAULT_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'yolov8n.pt'))
SPOTS_JSON = os.path.join(os.path.dirname(__file__), 'miejsca_parkingowe.json')

# job store: job_id -> {status, progress, output, error}
JOBS = {}


def _worker_job(job_id, in_path, out_path, model_path):
    JOBS[job_id]['status'] = 'processing'

    def progress_cb(pct):
        JOBS[job_id]['progress'] = int(pct)
        # print occasional progress to server console
        if int(pct) % 10 == 0:
            print(f"[job {job_id}] progress: {int(pct)}%")

    print(f"[job {job_id}] started. input={in_path} output={out_path}")
    try:
        result = process_video(in_path, spots_json_path=SPOTS_JSON, model_path=model_path, output_path=out_path, progress_callback=progress_cb)
        # attempt to transcode to browser-friendly H.264 MP4 using ffmpeg
        transcoded = None
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            trans_path = os.path.splitext(result)[0] + '_web.mp4'
            cmd = [ffmpeg_path, '-y', '-i', result, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', '-b:a', '128k', trans_path]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                transcoded = trans_path
                print(f"[job {job_id}] transcoded to {transcoded}")
            except Exception as e:
                print(f"[job {job_id}] ffmpeg failed: {e}")
        else:
            print(f"[job {job_id}] ffmpeg not found; skipping transcode")

        JOBS[job_id]['status'] = 'done'
        JOBS[job_id]['output'] = transcoded or result
        JOBS[job_id]['progress'] = 100
        print(f"[job {job_id}] finished. output={JOBS[job_id]['output']}")
    except Exception as e:
        JOBS[job_id]['status'] = 'error'
        JOBS[job_id]['error'] = str(e)
        print(f"[job {job_id}] error: {e}")


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    f = request.files['video']
    if f.filename == '':
        return jsonify({'error': 'no selected file'}), 400
    filename = secure_filename(f.filename)
    in_path = os.path.join(UPLOAD_DIR, filename)
    f.save(in_path)

    out_name = 'out_' + filename.rsplit('.', 1)[0] + '.mp4'
    out_path = os.path.join(UPLOAD_DIR, out_name)

    model_path = request.form.get('model') or DEFAULT_MODEL

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {'status': 'queued', 'progress': 0, 'output': None, 'error': None}

    t = threading.Thread(target=_worker_job, args=(job_id, in_path, out_path, model_path), daemon=True)
    t.start()

    return jsonify({'job_id': job_id}), 202


@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'not found'}), 404
    return jsonify(job)


@app.route('/result/<job_id>', methods=['GET'])
def result(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'not found'}), 404
    if job.get('status') != 'done':
        return jsonify({'error': 'not ready', 'status': job.get('status')}), 400
    return send_file(job['output'], mimetype='video/mp4')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

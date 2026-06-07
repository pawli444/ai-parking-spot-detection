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

# model leci z tej sciezki jak nie podasz innej z frontu
DEFAULT_MODEL = r"C:\Users\Mateusz\PycharmProjects\Parking_spot_detector\backend\src\runs\detect\doszkoleniev2\weights\best.pt"

# tu trzymamy stan zadan
JOBS = {}


# dodalismy spots_path jako argument
def _worker_job(job_id, in_path, out_path, model_path, spots_path, conf, imgsz, margin, iou, device, save_flag):
    JOBS[job_id]['status'] = 'processing'

    def progress_cb(pct):
        JOBS[job_id]['progress'] = int(pct)
        if int(pct) % 10 == 0:
            print(f"[job {job_id}] progress: {int(pct)}%")

    print(f"[job {job_id}] started. input={in_path} output={out_path}")
    try:
        # podajemy spots_path pobrane od uzytkownika zamiast globalnego jsona
        result = process_video(in_path, spots_json_path=spots_path, model_path=model_path,
                               output_path=out_path, conf=conf, imgsz=imgsz, margin=margin,
                               iou=iou, device=device, save_flag=save_flag,
                               progress_callback=progress_cb)

        # konwersja do mp4 na www
        transcoded = None
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            trans_path = os.path.splitext(result)[0] + '_web.mp4'
            cmd = [ffmpeg_path, '-y', '-i', result, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac',
                   '-b:a', '128k', trans_path]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                transcoded = trans_path
                print(f"[job {job_id}] transcoded to {transcoded}")
            except Exception as e:
                print(f"[job {job_id}] ffmpeg failed: {e}")
        else:
            print(f"[job {job_id}] ffmpeg not found; skipping transcode")

        base_dir = os.path.dirname(result)
        base_name = os.path.splitext(os.path.basename(result))[0]
        csv_path = os.path.join(base_dir, f"{base_name}_log.csv")
        plot_path = os.path.join(base_dir, f"{base_name}_plot.png")

        JOBS[job_id]['status'] = 'done'
        JOBS[job_id]['output'] = transcoded or result
        JOBS[job_id]['csv'] = csv_path
        JOBS[job_id]['plot'] = plot_path
        JOBS[job_id]['progress'] = 100
        print(f"[job {job_id}] finished.")

    except Exception as e:
        JOBS[job_id]['status'] = 'error'
        JOBS[job_id]['error'] = str(e)
        print(f"[job {job_id}] error: {e}")


@app.route('/upload', methods=['POST'])
def upload():
    # sprawdzamy czy przyszlo wideo
    if 'video' not in request.files:
        return jsonify({'error': 'no video file'}), 400
    f_vid = request.files['video']
    if f_vid.filename == '':
        return jsonify({'error': 'no selected video'}), 400

    # sprawdzamy czy przyszedl json z miejscami
    if 'spots' not in request.files:
        return jsonify({'error': 'no spots json file'}), 400
    f_spots = request.files['spots']
    if f_spots.filename == '':
        return jsonify({'error': 'no selected spots file'}), 400

    job_id = str(uuid.uuid4())

    vid_filename = secure_filename(f_vid.filename)
    in_path = os.path.join(UPLOAD_DIR, vid_filename)
    f_vid.save(in_path)

    spots_filename = f"{job_id}_{secure_filename(f_spots.filename)}"
    spots_path = os.path.join(UPLOAD_DIR, spots_filename)
    f_spots.save(spots_path)

    out_name = 'out_' + vid_filename.rsplit('.', 1)[0] + '.mp4'
    out_path = os.path.join(UPLOAD_DIR, out_name)

    model_path = request.form.get('model') or DEFAULT_MODEL

    def _parse_float(name, default):
        v = request.form.get(name)
        return float(v) if v else default

    def _parse_int(name, default):
        v = request.form.get(name)
        return int(v) if v else default

    conf = _parse_float('conf', 0.2)
    imgsz = _parse_int('imgsz', 960)
    margin = _parse_int('margin', 15)
    iou = _parse_float('iou', 0.6)
    device = request.form.get('device', '0')
    save_flag = request.form.get('save', 'false').lower() == 'true'

    JOBS[job_id] = {'status': 'queued', 'progress': 0, 'output': None, 'csv': None, 'plot': None, 'error': None}

    t = threading.Thread(
        target=_worker_job,
        args=(job_id, in_path, out_path, model_path, spots_path, conf, imgsz, margin, iou, device, save_flag),
        daemon=True
    )
    t.start()

    return jsonify({'job_id': job_id}), 202


@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'not found'}), 404
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'error': job['error']
    })


@app.route('/result/<job_id>/<file_type>', methods=['GET'])
def result_file(job_id, file_type):
    job = JOBS.get(job_id)
    if not job or job.get('status') != 'done':
        return jsonify({'error': 'not ready'}), 400

    if file_type == 'video' and job.get('output'):
        return send_file(job['output'], mimetype='video/mp4')
    elif file_type == 'csv' and job.get('csv'):
        return send_file(job['csv'], mimetype='text/csv', as_attachment=True, download_name='log_zajetosci.csv')
    elif file_type == 'plot' and job.get('plot'):
        return send_file(job['plot'], mimetype='image/png')

    return jsonify({'error': 'invalid file type or file missing'}), 400


@app.route('/result/<job_id>', methods=['GET'])
def result_video_only(job_id):
    return result_file(job_id, 'video')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
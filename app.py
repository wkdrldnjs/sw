from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, Response
import cv2
import torch
from werkzeug.utils import secure_filename
import os
from collections import defaultdict
import uuid
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import scipy.io.wavfile
from diffusers import AudioLDMPipeline, AudioLDM2Pipeline
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import requests
import io
import os
import moviepy.editor as mp
import math

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024
# 오디오 파일 저장 경로
app.config['SAVED_FOLDER'] = 'saved_audios'

if not os.path.exists(app.config['SAVED_FOLDER']):
    os.makedirs(app.config['SAVED_FOLDER'])

# GPU 사용 가능 여부 확인 및 디바이스 설정
audio_device = 'cuda' if torch.cuda.is_available() else 'cpu'
other_device = 'cpu'  # 나머지 연산은 CPU에서 수행

# YOLO 모델 로드 및 CPU 설정
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='oldfilm.pt').to(other_device)

# kosmos-2 모델 로드 및 CPU 설정
caption_model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224").to(other_device)
caption_processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

# AudioLDMPipeline 로드 및 GPU 설정
audio_pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-m-full", torch_dtype=torch.float16).to(audio_device)
audio_file_counts = defaultdict(int)
id_mapping = {}

@app.route('/', methods=['GET'])
def index():
    # 메인 페이지 렌더링
    return render_template('index.html')

@app.route('/upload/video', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일 부분이 없습니다'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '선택된 파일이 없습니다'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)

        video_url = url_for('uploaded_file', filename=filename)
        
        return jsonify({'original_video_url': video_url, 'filename': filename})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process/labels', methods=['POST'])
def process_labels():
    try:
        data = request.json
        # 클라이언트로부터 original_video_url 받기
        filename = data['filename']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # 비디오에서 객체 탐지 및 라벨링
        labeled_video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'labeled_{filename}')
        class_durations, capture_images = perform_detection_and_labeling(file_path, labeled_video_path)

        # 처리 결과를 JSON 형태로 반환
        return jsonify({
            "labeled_video_url": url_for('uploaded_file', filename=f'labeled_{filename}'),
            "labeled_video_path": labeled_video_path,
            "class_durations": class_durations,
            "capture_images": capture_images
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/process/audio', methods=['POST'])
def process_audio():
    try:
        data = request.json

        capture_images = data['capture_images']

        # 오디오 클립 정보를 저장할 리스트
        audio_clips_info = []
        for class_name in capture_images.keys():
            for capture in capture_images[class_name]:
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], capture['capture_filename'])
                caption = generate_caption_for_image(image_path)
                duration = capture['interval'][1] - capture['interval'][0]
                sound_wav, actual_duration = generate_sound_from_caption(caption, duration)
                #sound_filename = f"sound_{uuid.uuid4().hex}.wav"
                
                sound_filename = f"{class_name}_{audio_file_counts[class_name]}.wav"
                audio_file_counts[class_name] += 1
                sound_path = os.path.join(app.config['UPLOAD_FOLDER'], sound_filename)
                #sound_path = sound_path.replace('\\', '/')
                scipy.io.wavfile.write(sound_path, rate=16000, data=sound_wav)

                # 오디오 클립 정보 추가
                audio_clips_info.append({
                    'class': class_name, # 추가
                    'path': sound_path,
                    'start_time': capture['interval'][0],
                    'duration': min(actual_duration, duration),  # 실제 오디오 길이와 요청된 길이 중 작은 값 사용
                })

                capture['caption'] = caption
                capture['sound_url'] = url_for('uploaded_file', filename=sound_filename)

        return jsonify({
            'audio_clips_info': audio_clips_info,
            'capture_images': capture_images
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/finalize/video', methods=['POST'])
def finalize_video():
    try:
        data = request.json
        labeled_video_path = data['labeled_video_path']
        print('final_label: ', labeled_video_path)
        audio_clips_info = data['audio_clips_info']
        print('final_audio: ', audio_clips_info)
        final_video_path = add_audio_to_video(labeled_video_path, audio_clips_info)

        return jsonify({
            'final_video_url': url_for('uploaded_file', filename=os.path.basename(final_video_path))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/getList')
def getList():
    api_key = ''

    # 검색할 단어 설정
    search_words = list(audio_file_counts.keys())

    audios = []
    for word in search_words:
        print(f'Searching for sounds with word: {word}')
        # Freesound API를 통해 검색 진행
        url = f"https://freesound.org/apiv2/search/text/?query={word}&fields=id,name,previews&token={api_key}&page_size=5"
        response = requests.get(url)
        data = response.json()

        # 결과에서 오디오 파일의 이름, URL, 및 ID 추출
        for idx, sound in enumerate(data['results']):
            custom_id = f"{word}-{idx}"
            audios.append({
                'id': custom_id,  # 사용자 정의 ID 사용
                'name': sound['name'],
                'url': sound['previews']['preview-hq-mp3']
            })
            # 사용자 정의 ID와 원래의 Freesound ID 매핑 저장
            id_mapping[custom_id] = sound['id']

    print(f'Found {len(audios)} sounds')
    # 데이터를 JSON 형식으로 반환
    return jsonify(audios)

@app.route('/download')
def download_sound():
    api_key = 'EX2F14a3aOQbQeiqwu9YSbknz7IURVzBUtaeOTIB'

    # 쿼리 매개변수에서 custom_id 추출
    custom_id = request.args.get('soundId')

    # 매핑을 사용하여 원래의 sound_id 찾기
    sound_id = id_mapping.get(custom_id)
    if not sound_id:
        return f'Error: sound ID not found for {custom_id}', 404

    # Freesound API 다운로드 URL
    download_url = f'https://freesound.org/apiv2/sounds/{sound_id}/?token={api_key}'

    # 파일 다운로드 요청
    response = requests.get(download_url, allow_redirects=True)
    sound_info = response.json()

    if response.status_code == 200:
        download_url = sound_info['previews']['preview-lq-mp3']
        download_response = requests.get(download_url)
        # 'effects' 폴더에 파일 저장
        save_folder = 'effects'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        file_extension = download_url.split('.')[-1]
        file_path = os.path.join(save_folder, f'{custom_id}.{file_extension}')
        spath =  os.path.join(f'{custom_id}.{file_extension}')
        with open(file_path, 'wb') as file:
            file.write(download_response.content)
        
        print('패뚜: ', spath)
        return jsonify(spath)
    else:
        return f'Error downloading file: {response.reason}', response.status_code

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    # 오디오 생성 모델 초기화
    pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # 사용자 입력 받기
    text = request.form['text']
    duration = float(request.form['duration'])

    # 오디오 생성
    audio = pipe(text, num_inference_steps=200, audio_length_in_s=duration).audios[0]

    # 오디오를 BytesIO 객체에 저장
    byte_io = io.BytesIO()
    scipy.io.wavfile.write(byte_io, rate=16000, data=audio)
    byte_io.seek(0)  # 파일 읽기 위치를 시작점으로 이동

    return Response(byte_io.getvalue(), mimetype='audio/wav')

@app.route('/save_audio', methods=['POST'])
def save_audio():

    print('app[saved_folder]: ', app.config['SAVED_FOLDER'])

    # 오디오 데이터와 파일 이름 받기
    audio_data = request.files['audio_data']
    filename = request.form['filename']

    # 오디오 파일 저장
    filepath = os.path.join(app.config['SAVED_FOLDER'], filename)
    audio_data.save(filepath)

    # 저장된 오디오 파일 목록 반환
    path = os.path.join(app.config['SAVED_FOLDER'])
    mp3_files = [f for f in os.listdir(path) if f.endswith('.wav')]
    print('mp3_files: ', mp3_files)

    return jsonify(mp3_files)

def generate_caption_for_image(image_path):
    # 이미지로부터 캡션 생성
    image = Image.open(image_path)
    prompt = "<grounding>"
    inputs = caption_processor(text=prompt, images=image, return_tensors="pt").to(other_device)
    generated_ids = caption_model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=40,
    )
    generated_text = caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processed_text, _ = caption_processor.post_process_generation(generated_text)
    return processed_text

def merge_intervals(intervals, threshold=0.7):
    # 인터벌을 병합하는 함수 (threshold: 인접 인터벌 병합 임계값)
    merged = []
    for start, end in sorted(intervals, key=lambda x: x[0]):
        if merged and start - merged[-1][1] <= threshold:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def capture_at_intervals(video_path, intervals, class_name, upload_folder, ):
    # 비디오에서 특정 시간 간격의 이미지를 캡처
    cap = cv2.VideoCapture(video_path)
    capture_file_counts = defaultdict(int)
    if not cap.isOpened():
        raise ValueError("Could not open the video file")
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    for interval in intervals:
        mid_time = (interval[0] + interval[1]) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
        ret, frame = cap.read()
        if ret:
            #capture_filename = f"capture_{class_name}_{uuid.uuid4().hex}.jpg"
            capture_filename = f"capture_{class_name}_{capture_file_counts[class_name]}.jpg"            
            capture_path = os.path.join(upload_folder, capture_filename)
            cv2.imwrite(capture_path, frame)  # 프레임 저장
            caption = generate_caption_for_image(capture_path)
            yield {
                'time': mid_time,
                'interval': interval,
                'capture_url': url_for('uploaded_file', filename=capture_filename),
                'caption': caption,
                'capture_filename': capture_filename
            }
        capture_file_counts[class_name] += 1    
    cap.release()

def perform_detection_and_labeling(video_path, output_path):
    # 비디오에서 객체 탐지 및 라벨링 수행
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file")
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 'mp4v' 사용

    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
    if not out.isOpened():
        raise ValueError("Could not open the video writer")

    class_intervals = defaultdict(list)
    previous_detections = defaultdict(lambda: {'time': 0, 'last_seen': -1})
    capture_images = defaultdict(list)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        frame_rendered = results.render()[0]
        out.write(frame_rendered)

        time_stamp = frame_id / frame_rate
        current_detections = defaultdict(lambda: {'time': 0, 'last_seen': -1})

        for *xyxy, conf, cls in results.xyxy[0]:
            class_name = yolo_model.names[int(cls)]
            current_detections[class_name]['time'] = time_stamp
            current_detections[class_name]['last_seen'] = frame_id

        for class_name, info in current_detections.items():
            if class_name in previous_detections and info['last_seen'] - previous_detections[class_name]['last_seen'] <= 0.1 * frame_rate:
                class_intervals[class_name][-1][1] = time_stamp
            else:
                class_intervals[class_name].append([time_stamp, time_stamp])

        previous_detections = current_detections
        frame_id += 1

    cap.release()
    out.release()

    # 캡처 이미지 및 캡션 생성 (1초 이상 지속되는 인터벌에 대해서만)
    for class_name, intervals in class_intervals.items():
        merged_intervals = merge_intervals(intervals)
        long_intervals = [interval for interval in merged_intervals if interval[1] - interval[0] >= 1]  # 1초 이상 지속되는 인터벌만 선택

        if long_intervals:
            captures = list(capture_at_intervals(video_path, long_intervals, class_name, app.config['UPLOAD_FOLDER']))
            capture_images[class_name].extend(captures)

        class_durations = []
        for class_name, intervals in class_intervals.items():
            merged_intervals = merge_intervals(intervals)
            long_intervals = [interval for interval in merged_intervals if interval[1] - interval[0] >= 1]

            if sum(e - s for s, e in long_intervals) >= 1:
                class_durations.append({
                    'class': class_name,
                    'intervals': long_intervals,
                    'total_duration': sum(e - s for s, e in long_intervals),
                    'captures': capture_images[class_name]
                })


    return class_durations, capture_images

def generate_sound_from_caption(caption, duration):
    # 캡션을 바탕으로 오디오 생성 - 수정된 버전
    audio = audio_pipe(caption, num_inference_steps=10, audio_length_in_s=duration).audios[0]
    actual_duration = len(audio) / 16000  # 오디오 길이 계산 (16000은 샘플링 레이트)
    return audio, actual_duration

def add_audio_to_video(video_path, audio_clips_info):
    # 비디오에 오디오 추가 - 수정된 버전
    print('add func: ', video_path, audio_clips_info)
    video_clip = VideoFileClip(video_path)
    print('???')
    audio_clips = []
    for clip_info in audio_clips_info:
        print('clip: ', clip_info)
        audio_clip = AudioFileClip(clip_info['path'])

        if audio_clip.duration > clip_info['duration']:
            audio_clip = audio_clip.subclip(0, clip_info['duration'])

        audio_clip = audio_clip.set_start(clip_info['start_time'])
        audio_clips.append(audio_clip)

    composite_audio = CompositeAudioClip(audio_clips)
    video_clip_with_audio = video_clip.set_audio(composite_audio)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"final.mp4")
    video_clip_with_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")

    return output_path

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # 업로드된 파일 제공
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/merge', methods=['POST'])
def merge_audio():
    data = request.json
    audio_files = data['audio_files']  # 오디오 파일 정보
    
    # 오디오 파일 정보 출력
    for audio in audio_files:
        print(f"File: {audio['file']}, Start: {audio['start']}, End: {audio['end']}")

    # 비디오 파일 경로 정의
    video_file = 'uploads/final.mp4'
    video_clip = mp.VideoFileClip(video_file)  # 비디오 클립 정의

    audio_clips = []  # 오디오 클립을 저장할 리스트

    for audio in audio_files:
        # 'List' 폴더와 'uploads' 폴더에서 오디오 파일 경로 찾기
        audio_path_list = [f"uploads/{audio['file']}", f"effects/{audio['file']}", f"saved_audios/{audio['file']}"]
        audio_path = next((path for path in audio_path_list if os.path.exists(path)), None)

        if audio_path is None:
            print(f"Audio file not found: {audio['file']}")
            continue

        original_audio_clip = mp.AudioFileClip(audio_path)
        audio_duration = original_audio_clip.duration

        repeat_count = math.ceil((audio['end'] - audio['start']) / audio_duration)
        repeated_audio_clip = mp.concatenate_audioclips([original_audio_clip] * repeat_count)

        audio_clip = repeated_audio_clip.subclip(0, audio['end'] - audio['start'])

        # 오디오 클립을 비디오의 지정된 시간대에 배치
        audio_clip = audio_clip.set_start(audio['start'])
        audio_clips.append(audio_clip)

    # 모든 오디오 클립을 하나의 클립으로 합성
    final_audio = mp.CompositeAudioClip(audio_clips)

    # 비디오에 오디오 추가
    final_clip = video_clip.set_audio(final_audio)

    # 결과 저장
    output_path = 'static/uploads/merged_video.mp4'
    final_clip.write_videofile(output_path)

    return jsonify({"success": True, "output": output_path})


if __name__ == '__main__':
    app.run(debug=True)
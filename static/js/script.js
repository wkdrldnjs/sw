let videoDuration = 0;
let audioDuration = 0;
let audioTimelineInfo = {};

let audioPlayers = {};


const menuItems = document.querySelectorAll('.menu');
const contents = document.querySelectorAll('.content');


function highlightMenuItem(selectedItem) {
    menuItems.forEach(item => item.classList.remove('selected'));
    selectedItem.classList.add('selected');
}

function switchContent(event) {
    contents.forEach(content => content.classList.add('hide'));

    const contentToShow = document.getElementById(event.target.id + 'Content');
    contentToShow.classList.remove('hide');

    highlightMenuItem(event.target);
}

menuItems.forEach(item => {
    item.addEventListener('click', switchContent);
});

var response = {
    audio_clips_info: null,
    capture_images: null,
    class_durations: null,
    filename: null,
    final_video_url: null,
    original_video_url: null,
    labeled_video_url: null,
    labeled_video_path: null
};

function uploadVideo() {
    var formData = new FormData(document.getElementById('uploadForm'));
    showLoader();
    $.ajax({
        url: '/upload/video',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        
        success: function(responsee) {
            response.original_video_url = responsee.original_video_url;
            response.filename = responsee.filename;
            $('#originalVideoPlayer source').attr('src', responsee.original_video_url);
            $('#originalVideoPlayer')[0].load();

            process_labels();
        },
        error: function(xhr, status, error) {
            var errorMessage = xhr.status + ': ' + xhr.statusText;
            alert('video-upload error' + errorMessage);
        }
    });
}

function process_labels() {
    var data = {
        original_video_url: response.original_video_url,
        filename: response.filename
    };
    console.log(data)
    $.ajax({
        url: '/process/labels',
        type: 'POST',
        data: JSON.stringify(data),
        contentType: 'application/json; charset=utf-8',
        dataType: 'json',

        success: function(responsee) {
            response.labeled_video_url = responsee.labeled_video_url;
            response.class_durations = responsee.class_durations;
            response.capture_images = responsee.capture_images;
            response.labeled_video_path = responsee.labeled_video_path;
            console.log(responsee.catpure_images)

            displayResults(response.class_durations);
            console.log('cature_images: ', response.capture_images)
            hideLoader();

            process_audio();
        },
        error: function(xhr, status, error) {
            var errorMessage = xhr.status + ': ' + xhr.statusText;
            alert('object-detection error' + errorMessage);
        }
    })
}

function process_audio() {
    var data = {
        capture_images: response.capture_images
    };
    $.ajax({
        url: '/process/audio',
        type: 'POST',
        data: JSON.stringify(data),
        contentType: 'application/json; charset=utf-8',
        dataType: 'json',

        success: function(responsee){
            response.audio_clips_info = responsee.audio_clips_info;
            response.capture_images = responsee.capture_images;
            console.log(response)

            displayCaptureHighlights(response.capture_images);
            hideLoader1();
            display_final_video(response.original_video_url, response);
            sound_list();
            response.audio_clips_info.forEach((clip, index) => {
                const filePath = clip.path;
                const fileName = filePath.split('/').pop().replace('.wav', '');
                loadAudioFile(filePath, fileName, clip.start_time, clip.duration, index);
            });
            let currentIndex = 0;
            response.audio_clips_info.forEach(clip => {
                clip.index = currentIndex++;
                console.log('response.audio_clip_info: ', response.audio_clips_info);
            });

            finalize_video();
        },
        error: function(xhr, status, error) {
            var errorMessage = xhr.status + ': ' + xhr.statusText;
            alert('capture & sound error' + errorMessage);
        }
    })
}

function finalize_video() {
    var data = {
        labeled_video_path: response.labeled_video_path,
        audio_clips_info: response.audio_clips_info
    };
    $.ajax({
        url: '/finalize/video',
        type: 'POST',
        data: JSON.stringify(data),
        contentType: 'application/json; charset=utf-8',
        dataType: 'json',

        success: function(responsee){
            response.final_video_url = responsee.final_video_url;
            $('#labeledVideoPlayer source').attr('src', responsee.final_video_url);
            $('#labeledVideoPlayer')[0].load();

            console.log('최종 response: ', response)
        },
        error: function(xhr, status, error) {
            var errorMessage = xhr.status + ': ' + xhr.statusText;
            alert('final error' + errorMessage);
        }
    })
}

function showLoader() {
    $(".wrapper").show();
}

function hideLoader() {
    $(".wrapper").hide();
}
function hideLoader1() {
    $(".loader").hide();
}

function sound_list() {
    var formData = new FormData();
    var filename = document.getElementById("text").values
    $.ajax({
        url: '/getList',
        type: 'GET',
        data: '',
        processData: false,
        contentType: false,
        success: function(response) {
            displayAudioList(response);
            console.log('sound_list response: ', response)
        }
    })
}

function displayResults(data) {
    var resultsElement = document.getElementById('results');
    resultsElement.innerHTML = '';

    data.forEach(function(duration) {
        var intervalsHtml = duration.intervals.map(function(interval) {
            return '<div class="interval">' + interval[0].toFixed(2) + '초 - ' + interval[1].toFixed(2) + '초</div>';
        }).join('');

        var resultHtml = `
            <div class="result-item">
                <div class="result-header">${duration.class}</div>
                <div class="intervals-container">${intervalsHtml}</div>
                <div class="result-duration">총 지속 시간: ${duration.total_duration.toFixed(2)}초</div>
            </div>
        `;
        resultsElement.innerHTML += resultHtml;
    });
}

function displayCaptureHighlights(data) {
    var highlightsElement = $('#captureHighlights');
    highlightsElement.empty();

    for (var class_name in data) {
        data[class_name].forEach(function(capture) {
            console.log('url: ', capture.capture_url)
            highlightsElement.append('<div><p>' + class_name + ': ' + parseFloat(capture.time).toFixed(2) + '초</p><img src="' + capture.capture_url + '" width="320" alt="Capture Image"><p>캡션: ' + capture.caption + '</p><audio controls><source src="' + capture.sound_url + '" type="audio/wav">해당 오디오 파일은 지원되지 않는 브라우저에서는 재생할 수 없습니다.</audio></div>');
        });
    }
}

function init() {
    contents.forEach(content => content.classList.add('hide'));

    const soundContent = document.getElementById('soundContent');
    soundContent.classList.remove('hide');

    const soundMenuItem = document.getElementById('sound');
    soundMenuItem.classList.add('selected');
}

function downloadFile(soundId) {
    fetch('/download?soundId=' + soundId)
        .then(response => response.json())
        .then(filepath => {
            console.log('filename(api): ',  filepath)
            updateAudioClipPath(filepath);
        })
        .catch(e => console.error(e));
}

function extractSubstring(str) {
    let reversedStr = str.split('').reverse().join('');

    let firstBackslashIndex = reversedStr.indexOf('\\');
    if (firstBackslashIndex === -1) {
        return "";
    }

    let secondBackslashIndex = reversedStr.indexOf('\\', firstBackslashIndex + 1);
    if (secondBackslashIndex === -1) {
        secondBackslashIndex = reversedStr.length;
    }

    return reversedStr.substring(firstBackslashIndex + 1, secondBackslashIndex).split('').reverse().join('');
}

function displayAudioList(data) {
    var listElement = document.getElementById('square11').querySelector('ul');
    listElement.innerHTML = '';

    data.forEach(function(item) {
        var listItemHTML = `
            <li>
                ${item.id}
                <audio controls>
                    <source src="${item.url}" type="audio/mpeg">
                    Your browser does not support the audio element.
                </audio>
                <button id="yas" onclick="downloadFile('${item.id}');">적용</button>
            </li>
        `;

        listElement.innerHTML += listItemHTML;
    });
}

function updateAudioClipPath(fileName) {
    console.log('비교 filename: ', fileName)
    const className = fileName.replace(/\.wav|\.mp3/g, '').replace(/[^a-zA-Z]/g, '');

    response.audio_clips_info.forEach(clip => {
        if (clip.class === className) {
            clip.path = '../effects/' + fileName;
        }
    });

    updateTimelines();
}


function updateTimelines() {
    const timelineContainer = document.querySelector('.timeline-container');
    timelineContainer.innerHTML = '';

    response.audio_clips_info.forEach((clip, index) => {
        const fileName = clip.path.split('/').pop().replace('.wav', '').replace('.mp3', '');
        addAudioTimeline(clip.duration, fileName, clip.start_time, index); // index 추가
    });
}

var audioBlob = null;

function generateAudio() {
    var form = document.getElementById('audioForm');
    var formData = new FormData(form);
    fetch('/generate_audio', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        audioBlob = blob;
        var audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = URL.createObjectURL(blob);
        audioPlayer.style.display = 'block';
        document.getElementById('saveButton').style.display = 'block';
        audioPlayer.play();
    });
}

function saveAudio() {
    var formData = new FormData();
    var filename = document.getElementById("text").value
    formData.append('audio_data', audioBlob, filename + '.wav');
    formData.append('filename', filename + '!.wav');

    fetch('/save_audio', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(filename => {
        filename.forEach(file => {
            updateAudioClipPath(file)
        })
    })
    .then(data => alert(data.message))
    .catch(error => console.error('There has been a problem with your fetch operation:', error));
}



//timeline script


function display_final_video(path, data) {
    const videoPlayer = document.getElementById('videoPlayer1');
    videoPlayer.src = path;
    videoPlayer.onloadedmetadata = () => {
        videoDuration = videoPlayer.duration;
        data.audio_clips_info.forEach(clip => {
            addAudioTimeline(clip.duration, clip.path, clip.start_time, clip.index)
        })
    };
}

function createAudioTimelines(data) {
    data.audio_clips_info.forEach(clip => {
        addAudioTimeline(clip.duration, clip.path, clip.start_time)
    })
}

function loadAudioFile(filePath, fileName, startTime, duration, index) {
    const audioPlayer = new Audio(filePath);
    audioPlayer.onloadedmetadata = () => {
        //addAudioTimeline(duration, fileName, startTime, index); // index 추가
        audioPlayers[fileName] = audioPlayer;
    };
}

function addAudioTimeline(duration, fName, startTime, index) {
    fileName = fName.split('/').pop().replace('.wav', '');
    const audioTimeline = document.createElement('div');
    audioTimeline.className = 'timeline';
    audioTimeline.style.backgroundColor = '#007bff';
    audioTimeline.style.margin = '22px 0'
    if (fileName.includes('train')) {
        audioTimeline.style.backgroundColor = '#007bff';
    } else if (fileName.includes('wheel')){
        audioTimeline.style.backgroundColor = '#d5ca01';
    } else if (fileName.includes('horse')){
        audioTimeline.style.backgroundColor = '#60451f';
    }
    // index 속성 추가
    audioTimeline.setAttribute('data-index', index);
    // 오디오 길이를 비디오 타임라인에 비례하여 설정
    const audioLengthPercent = Math.min((duration / videoDuration) * 100, 100);
    audioTimeline.style.width = audioLengthPercent + '%';

    // 오디오 시작 시간 및 종료 시간 설정
    const endTime = startTime + duration;

    // 타임라인의 위치 설정
    const startLeftPercent = (startTime / videoDuration) * 100;
    audioTimeline.style.left = startLeftPercent + '%';

    // 파일 이름 레이블 추가
    const nameLabel = document.createElement('span');
    nameLabel.textContent = fileName.substring(fileName.lastIndexOf('\\') + 1);
    nameLabel.style.color = 'black';
    nameLabel.style.padding = '5px';
    audioTimeline.appendChild(nameLabel);

    // 시작과 끝 시간 레이블 추가
    const timeLabel = document.createElement('div');
    timeLabel.innerHTML = `<span>시작: ${formatTime(startTime)}</span> - <span>끝: ${formatTime(endTime)}</span>`;
    timeLabel.style.textAlign = 'center';
    audioTimeline.appendChild(timeLabel);

    // 크기 조절 핸들 추가
    const resizeHandle = document.createElement('div');
    resizeHandle.className = 'resize-handle';
    audioTimeline.appendChild(resizeHandle);

    // 오디오 타임라인 클릭 이벤트 추가
    audioTimeline.addEventListener('click', function () {
        playAudio(fileName);
    });

    // 타임라인 컨테이너에 오디오 타임라인 추가
    document.querySelector('.timeline-container').appendChild(audioTimeline);

    // 드래그 및 크기 조절 기능 설정
    setupTimelineControls(audioTimeline);

    // 오디오 타임라인 정보 저장
    audioTimelineInfo[index] = { fileName, duration, startTime, endTime, element: audioTimeline };
}

function playAudio(fileName) {
    // response.audio_clips_info 배열에서 해당 fileName과 일치하는 항목 찾기
    if (!response || !response.audio_clips_info) {
        console.error('response or audio_clips_info is undefined');
        return;
    }
    const clipInfo = response.audio_clips_info.find(clip => clip.path.includes(fileName));
    const audioPlayer = document.getElementById('audioPlayer');

    if (clipInfo && audioPlayer) {
        // 찾은 오디오 클립의 경로로 오디오 소스 설정
        audioPlayer.src = clipInfo.path;
        audioPlayer.load(); // 오디오 플레이어 새로 고침

        const audioPlayerContainer = document.getElementById('audioPlayerContainer');
        audioPlayerContainer.style.display = 'block'; // 오디오 컨트롤러 표시
    }
}


function formatTime(seconds) {
    const date = new Date(0);
    date.setSeconds(seconds);
    return date.toISOString().substr(11, 8);
}

function setupTimelineControls(timeline) {
    // 드래그 기능 설정
    let isDragging = false;
    let dragStartX = 0;
    let originalLeft = 0;
    const fileName = timeline.querySelector('span').textContent;

    timeline.addEventListener('mousedown', (e) => {
        if (!e.target.classList.contains('resize-handle')) {
            isDragging = true;
            dragStartX = e.clientX;
            originalLeft = parseInt(window.getComputedStyle(timeline).left, 10) || 0;
            e.preventDefault();
        }
    });

    document.addEventListener('mousemove', (e) => {
        if (isDragging) {
            const deltaX = e.clientX - dragStartX;
            const newLeft = originalLeft + deltaX;
            timeline.style.left = newLeft + 'px';
            updateTimelineTimes(timeline, fileName, true);
        }
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
        }
    });

    // 크기 조절 기능 설정
    const handle = timeline.querySelector('.resize-handle');
    handle.addEventListener('mousedown', function (e) {
        e.preventDefault();
        const startX = e.clientX;
        const startWidth = timeline.offsetWidth;
        const timelineContainer = document.querySelector('.timeline-container');
        const totalWidth = timelineContainer.offsetWidth;

        function onMouseMove(e) {
            let currentX = e.clientX;
            let deltaX = currentX - startX;
            let newWidth = Math.max(startWidth + deltaX, 0);
            let maxWidth = totalWidth - parseInt(timeline.style.left, 10);
            newWidth = Math.min(newWidth, maxWidth);
            timeline.style.width = newWidth + 'px';
            updateTimelineTimes(timeline, fileName, false); // 크기 조절 시 타임라인 정보 업데이트
        }

        function onMouseUp() {
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
            updateAudioTimelineInfo(timeline, fileName); // 크기 조절 완료 시 정보 업데이트
        }

        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
    });
}

function updateAudioTimelineInfo(timeline, index) {
    const totalWidth = document.querySelector('.timeline-container').offsetWidth;
    const timelineRect = timeline.getBoundingClientRect();
    const containerRect = document.querySelector('.timeline-container').getBoundingClientRect();

    let newLeft = timelineRect.left - containerRect.left;
    let timelineWidth = timelineRect.width;

    let startTime = (newLeft / totalWidth) * videoDuration;
    let endTime = startTime + (timelineWidth / totalWidth) * videoDuration;

    if (audioTimelineInfo[index]) {
        audioTimelineInfo[index].startTime = Math.round(startTime * 100) / 100;
        audioTimelineInfo[index].endTime = Math.round(endTime * 100) / 100;

        const clipInfo = response.audio_clips_info.find(clip => clip.index === index);
        if (clipInfo) {
            clipInfo.start_time = audioTimelineInfo[index].startTime;
            clipInfo.duration = audioTimelineInfo[index].endTime - audioTimelineInfo[index].startTime;
        }
    }
}

function updateTimelineTimes(timeline, isDragging) {
    if (!timeline) return;
    const totalWidth = document.querySelector('.timeline-container').offsetWidth;
    const timelineRect = timeline.getBoundingClientRect();
    const containerRect = document.querySelector('.timeline-container').getBoundingClientRect();
    // timeline 요소에서 index 얻기
    const index = parseInt(timeline.getAttribute('data-index'));

    // 타임라인의 실제 픽셀 위치와 너비 계산
    let newLeft = timelineRect.left - containerRect.left;
    let timelineWidth = timelineRect.width;

    // 전체 비디오 길이를 사용하여 실제 시간 계산
    let startTime = (newLeft / totalWidth) * videoDuration;
    let endTime = startTime + (timelineWidth / totalWidth) * videoDuration;

    // 드래그 시 시작 시간과 끝 시간 제한
    if (isDragging) {
        if (startTime < 0) {
            startTime = 0;
            endTime = (timelineWidth / totalWidth) * videoDuration;
            newLeft = 0;
        } else if (endTime > videoDuration) {
            endTime = videoDuration;
            startTime = videoDuration - (timelineWidth / totalWidth) * videoDuration;
            newLeft = (startTime / videoDuration) * totalWidth;
        }
        timeline.style.left = newLeft + 'px';
    }

    // 크기 조절 시 끝 시간 제한
    if (!isDragging) {
        if (endTime > videoDuration) {
            endTime = videoDuration;
            timelineWidth = (endTime - startTime) / videoDuration * totalWidth;
            timeline.style.width = timelineWidth + 'px';
        }
    }

    // 소수점 둘째 자리에서 반올림
    const roundedStartTime = Math.round(startTime * 100) / 100;
    const roundedEndTime = Math.round(endTime * 100) / 100;

    const timeLabel = timeline.querySelector('div');
    timeLabel.innerHTML = `<span>시작: ${formatTime(roundedStartTime)}</span> - <span>끝: ${formatTime(roundedEndTime)}</span>`;

    if (audioTimelineInfo[index]) {
        audioTimelineInfo[index].startTime = Math.round(startTime * 100) / 100;
        audioTimelineInfo[index].endTime = Math.round(endTime * 100) / 100;

        const clipInfo = response.audio_clips_info.find(clip => clip.index === index);
        if (clipInfo) {
            clipInfo.start_time = audioTimelineInfo[index].startTime;
            clipInfo.duration = audioTimelineInfo[index].endTime - audioTimelineInfo[index].startTime;
        }
    }
}

const video = document.getElementById('videoPlayer1');
const seeker = document.querySelector('.timeline-seeker');

video.addEventListener('loadedmetadata', () => {
    video.addEventListener('timeupdate', () => {
        const percent = (video.currentTime / video.duration) * 100;
        seeker.style.left = percent + '%';
    });
});

document.getElementById('merge').addEventListener('click', function () {
    const audioFiles = response.audio_clips_info.map(clip => {
        const filePath = clip.path.replace(/\\/g, '/').split('/').pop(); // 백슬래시를 슬래시로 변환 후 파일 이름 추출
        return {
            file: filePath,
            start: clip.start_time,
            end: clip.start_time + clip.duration
        };
    });

    fetch('/merge', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ audio_files: audioFiles })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('병합 완료! 파일 위치: ' + data.output);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
});

 // 가상 데이터
 const timeOfDayData = {
    labels: ['1AM', '3AM', '5AM', '7AM', '9AM', '11AM', '1PM', '3PM', '5PM', '7PM', '9PM', '11PM'],
    datasets: [{
        label: '이용량',
        data: [ 186, 162, 103, 40, 100, 200, 150, 250, 300, 280, 250, 230 ],
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
    }]
};

const dailySoundData = {
    labels: ['월', '화', '수', '목', '금', '토', '일'],
    datasets: [{
        label: '일일 생성량',
        data: [130, 143, 170, 150, 130, 180, 200],
        fill: false,
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1
    }]
};

const subscriptionData = {
    labels: ['STARTER', 'PRO', 'PRO+'],
    datasets: [{
        label: '요금제 비율',
        data: [50, 30, 20],
        backgroundColor: [
            'rgb(255, 99, 132)',
            'rgb(54, 162, 235)',
            'rgb(255, 205, 86)'
        ],
        hoverOffset: 4
    }]
};

const apiAiUsageData = {
    labels: ['API', 'AI'],
    datasets: [{
        label: '사용 비율',
        data: [40, 60],
        backgroundColor: [
            'rgb(75, 192, 192)',
            'rgb(153, 102, 255)'
        ],
        hoverOffset: 4
    }]
};

const mostUsedSounds = ['The sound of Train', 'The sound of Horse', 'The sound of Wheel', 'The sound of Storm', 'The sound of Dog', 'The sound of Engine', 'The sound of Horn'];
const mostreviewSounds = ['The sound of Dog', 'The sound of Drum', 'The sound of Train', 'The sound of Plane', 'The sound of Explosion', 'The sound of Storm', 'The sound of Wind'];

// 차트 생성
window.onload = function() {
    new Chart(document.getElementById('graph-timeofday'), {
        type: 'line',
        data: timeOfDayData,
    });

    new Chart(document.getElementById('graph-soundcreation'), {
        type: 'line',
        data: dailySoundData,
    });

    new Chart(document.getElementById('chart-subscription'), {
        type: 'pie',
        data: subscriptionData,
        options: {
            aspectRatio: 1.2 
        }
    });

    new Chart(document.getElementById('chart-api-ai'), {
        type: 'pie',
        data: apiAiUsageData,
        options: {
            aspectRatio: 1.2 
        }
    });
    

    // 리스트 추가
    var list = document.getElementById('list-mostused');
    mostUsedSounds.forEach(function(sound, index) {
        var listItem = document.createElement('li');
        listItem.innerText = sound;
        list.appendChild(listItem);
    });

    // 리스트 추가
    var list = document.getElementById('list-review');
    mostreviewSounds.forEach(function(sound, index) {
        var listItem = document.createElement('li');
        listItem.innerText = sound;
        list.appendChild(listItem);
    });
    
    
};

window.addEventListener('load', init);

import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12";
import Delaunator from "https://cdn.jsdelivr.net/npm/delaunator@5.0.1/+esm";
import { LOCAL_CONFIG } from "./config.js";

/* ============================================================
 *  목소리 → 감정 → 표정  데모
 *  3단계: (1) 사진에서 얼굴 랜드마크 (2) 음성에서 감정 (3) 표정 워핑
 * ============================================================ */

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const faceStatus = document.getElementById("faceStatus");
const featuresEl = document.getElementById("features");
const emotionResult = document.getElementById("emotionResult");

let baseImage = null; // 원본 이미지 (HTMLImageElement)
let baseLandmarks = null; // [{x,y}] 픽셀 좌표 (원본 기준)
let triangles = null; // Delaunator triangle index array
let faceWidth = 0; // 얼굴 가로 폭(px) — 변위 스케일 기준
let landmarker = null;

/* ---------- MediaPipe FaceLandmarker 로드 ---------- */
async function initLandmarker() {
    const fileset = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
    );
    landmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        },
        runningMode: "IMAGE",
        numFaces: 1,
    });
}

/* ---------- 사진 업로드 ---------- */
document.getElementById("photoInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => detectFace(img);
    img.src = URL.createObjectURL(file);
});

async function detectFace(img) {
    faceStatus.textContent = "얼굴 분석 중...";
    // 캔버스 크기를 이미지에 맞춤 (너무 크면 축소)
    const maxW = 520;
    const scale = Math.min(1, maxW / img.width);
    canvas.width = Math.round(img.width * scale);
    canvas.height = Math.round(img.height * scale);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    baseImage = img;

    if (!landmarker) {
        try {
            await initLandmarker();
        } catch (err) {
            faceStatus.textContent = "랜드마크 모델 로드 실패 (네트워크 확인)";
            console.error(err);
            return;
        }
    }

    const result = landmarker.detect(canvas);
    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
        faceStatus.textContent = "얼굴을 못 찾았어요. 정면 사진으로 시도하세요.";
        baseLandmarks = null;
        return;
    }

    // 정규화 좌표 → 픽셀
    baseLandmarks = result.faceLandmarks[0].map((p) => ({
        x: p.x * canvas.width,
        y: p.y * canvas.height,
    }));

    // 삼각 분할 (Delaunay)
    const coords = [];
    baseLandmarks.forEach((p) => coords.push(p.x, p.y));
    triangles = new Delaunator(coords).triangles;

    // 얼굴 폭 = 좌우 광대(234, 454) 거리
    const L = baseLandmarks[234], R = baseLandmarks[454];
    faceWidth = Math.hypot(R.x - L.x, R.y - L.y) || canvas.width * 0.5;

    faceStatus.textContent = `얼굴 인식 완료 (랜드마크 ${baseLandmarks.length}개)`;
    redrawBase();
}

function redrawBase() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(baseImage, 0, 0, canvas.width, canvas.height);
}

/* ============================================================
 *  표정 변형 (Action Unit 근사)
 * ============================================================ */

// 사용할 랜드마크 인덱스 그룹 (MediaPipe 468 기준)
const IDX = {
    mouthL: 61, mouthR: 291,
    upperLip: [13, 0, 37, 267],
    lowerLip: [14, 17, 84, 314],
    browInnerL: 107, browInnerR: 336,
    browMidL: 105, browMidR: 334,
    browOuterL: 70, browOuterR: 300,
    eyeUpperL: [159, 158], eyeLowerL: [145, 153],
    eyeUpperR: [386, 385], eyeLowerR: [374, 380],
};

// 감정별 변위 정의 — 값은 faceWidth 대비 비율, dy 음수 = 위로
const EMOTION_GAIN = 1.7; // 변형 가시성 전역 배율
function emotionField(emotion, intensity) {
    const m = new Map();
    const g = intensity * EMOTION_GAIN;
    const add = (idx, dx, dy) => {
        const cur = m.get(idx) || { dx: 0, dy: 0 };
        cur.dx += dx * g;
        cur.dy += dy * g;
        m.set(idx, cur);
    };
    const addAll = (arr, dx, dy) => arr.forEach((i) => add(i, dx, dy));

    switch (emotion) {
        case "happy":
            add(IDX.mouthL, -0.03, -0.055);
            add(IDX.mouthR, 0.03, -0.055);
            addAll(IDX.lowerLip, 0, 0.012);
            addAll(IDX.eyeLowerL, 0, -0.012);
            addAll(IDX.eyeLowerR, 0, -0.012);
            break;
        case "sad":
            add(IDX.mouthL, 0.012, 0.05);
            add(IDX.mouthR, -0.012, 0.05);
            add(IDX.browInnerL, 0.012, -0.03);
            add(IDX.browInnerR, -0.012, -0.03);
            add(IDX.browOuterL, 0, 0.015);
            add(IDX.browOuterR, 0, 0.015);
            break;
        case "angry":
            add(IDX.browInnerL, 0.022, 0.035);
            add(IDX.browInnerR, -0.022, 0.035);
            add(IDX.browMidL, 0.01, 0.02);
            add(IDX.browMidR, -0.01, 0.02);
            add(IDX.mouthL, 0, 0.02);
            add(IDX.mouthR, 0, 0.02);
            addAll(IDX.upperLip, 0, 0.006);
            break;
        case "surprise":
            add(IDX.browInnerL, 0, -0.05); add(IDX.browInnerR, 0, -0.05);
            add(IDX.browMidL, 0, -0.05); add(IDX.browMidR, 0, -0.05);
            add(IDX.browOuterL, 0, -0.04); add(IDX.browOuterR, 0, -0.04);
            addAll(IDX.eyeUpperL, 0, -0.02); addAll(IDX.eyeUpperR, 0, -0.02);
            addAll(IDX.eyeLowerL, 0, 0.012); addAll(IDX.eyeLowerR, 0, 0.012);
            addAll(IDX.upperLip, 0, -0.02);
            addAll(IDX.lowerLip, 0, 0.045);
            break;
        case "fear":
            add(IDX.browInnerL, 0.015, -0.04); add(IDX.browInnerR, -0.015, -0.04);
            add(IDX.browMidL, 0, -0.03); add(IDX.browMidR, 0, -0.03);
            addAll(IDX.eyeUpperL, 0, -0.025); addAll(IDX.eyeUpperR, 0, -0.025);
            addAll(IDX.eyeLowerL, 0, 0.01); addAll(IDX.eyeLowerR, 0, 0.01);
            add(IDX.mouthL, 0.02, 0.02); add(IDX.mouthR, -0.02, 0.02);
            break;
        case "neutral":
        default:
            break;
    }
    return m;
}

// 변위 적용 → 워핑 렌더
function applyEmotion(emotion, intensity) {
    if (!baseLandmarks || !triangles) {
        faceStatus.textContent = "먼저 얼굴이 인식된 사진이 필요합니다.";
        return;
    }
    const field = emotionField(emotion, intensity);
    const dst = baseLandmarks.map((p, i) => {
        const d = field.get(i);
        if (!d) return { x: p.x, y: p.y };
        return { x: p.x + d.dx * faceWidth, y: p.y + d.dy * faceWidth };
    });

    redrawBase(); // 배경/머리카락은 원본 유지
    for (let t = 0; t < triangles.length; t += 3) {
        const a = triangles[t], b = triangles[t + 1], c = triangles[t + 2];
        drawTriangle(
            [baseLandmarks[a], baseLandmarks[b], baseLandmarks[c]],
            [dst[a], dst[b], dst[c]]
        );
    }
}

// 한 삼각형을 src→dst 아핀 변환으로 텍스처 매핑
function drawTriangle(s, d) {
    const dd = expand(d, 0.6); // 미세 확장으로 seam 완화
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(dd[0].x, dd[0].y);
    ctx.lineTo(dd[1].x, dd[1].y);
    ctx.lineTo(dd[2].x, dd[2].y);
    ctx.closePath();
    ctx.clip();

    const [s0, s1, s2] = s;
    const denom = (s1.x - s0.x) * (s2.y - s0.y) - (s2.x - s0.x) * (s1.y - s0.y);
    if (Math.abs(denom) < 1e-6) { ctx.restore(); return; }
    const a = ((d[1].x - d[0].x) * (s2.y - s0.y) - (d[2].x - d[0].x) * (s1.y - s0.y)) / denom;
    const b = ((d[1].y - d[0].y) * (s2.y - s0.y) - (d[2].y - d[0].y) * (s1.y - s0.y)) / denom;
    const c = ((d[2].x - d[0].x) * (s1.x - s0.x) - (d[1].x - d[0].x) * (s2.x - s0.x)) / denom;
    const dcoef = ((d[2].y - d[0].y) * (s1.x - s0.x) - (d[1].y - d[0].y) * (s2.x - s0.x)) / denom;
    const e = d[0].x - a * s0.x - c * s0.y;
    const f = d[0].y - b * s0.x - dcoef * s0.y;
    ctx.setTransform(a, b, c, dcoef, e, f);
    ctx.drawImage(baseImage, 0, 0, baseImage.width, baseImage.height, 0, 0, canvas.width, canvas.height);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.restore();
}

// 삼각형 중심 기준 약간 확장
function expand(tri, px) {
    const cx = (tri[0].x + tri[1].x + tri[2].x) / 3;
    const cy = (tri[0].y + tri[1].y + tri[2].y) / 3;
    return tri.map((p) => {
        const len = Math.hypot(p.x - cx, p.y - cy) || 1;
        return { x: p.x + ((p.x - cx) / len) * px, y: p.y + ((p.y - cy) / len) * px };
    });
}

/* ============================================================
 *  음성 녹음 + 특징 추출
 * ============================================================ */

const recordBtn = document.getElementById("recordBtn");
const recStatus = document.getElementById("recStatus");
const levelEl = document.getElementById("level");

let mediaRecorder = null;
let chunks = [];
let recognition = null;
let lastTranscript = "";

recordBtn.addEventListener("click", async () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        return;
    }
    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
        recStatus.textContent = "마이크 권한이 필요합니다.";
        return;
    }

    // 실시간 레벨 미터
    const ac = new AudioContext();
    const src = ac.createMediaStreamSource(stream);
    const analyser = ac.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser);
    const buf = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
        if (!mediaRecorder || mediaRecorder.state !== "recording") return;
        analyser.getByteTimeDomainData(buf);
        let sum = 0;
        for (const v of buf) sum += (v - 128) * (v - 128);
        const rms = Math.sqrt(sum / buf.length) / 128;
        levelEl.style.width = Math.min(100, rms * 250) + "%";
        requestAnimationFrame(tick);
    };

    chunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        ac.close();
        levelEl.style.width = "0%";
        recordBtn.classList.remove("recording");
        recordBtn.textContent = "● 녹음 시작";
        recStatus.textContent = "분석 중...";
        const blob = new Blob(chunks, { type: "audio/webm" });
        await analyzeAudio(blob);
    };

    // 선택: Web Speech 전사
    lastTranscript = "";
    if (document.getElementById("useTranscript")?.checked) startTranscript();

    mediaRecorder.start();
    recordBtn.classList.add("recording");
    recordBtn.textContent = "■ 녹음 중지";
    recStatus.textContent = "녹음 중... (말하고 중지)";
    tick();
});

function startTranscript() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    recognition = new SR();
    recognition.lang = "ko-KR";
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.onresult = (e) => {
        for (let i = e.resultIndex; i < e.results.length; i++) {
            lastTranscript += e.results[i][0].transcript + " ";
        }
    };
    try { recognition.start(); } catch (_) {}
}

async function analyzeAudio(blob) {
    if (recognition) { try { recognition.stop(); } catch (_) {} recognition = null; }
    const arrayBuf = await blob.arrayBuffer();
    const ac = new AudioContext();
    const audioBuf = await ac.decodeAudioData(arrayBuf);
    ac.close();
    const feats = extractFeatures(audioBuf);
    featuresEl.textContent =
        `에너지(평균): ${feats.energy.toFixed(3)}\n` +
        `피치(중앙값): ${feats.pitchMedian.toFixed(0)} Hz\n` +
        `피치 변동성: ${feats.pitchStd.toFixed(0)} Hz\n` +
        `유성음 비율: ${(feats.voicedRatio * 100).toFixed(0)}%\n` +
        `말속도(근사): ${feats.speechRate.toFixed(2)}` +
        (lastTranscript ? `\n전사: "${lastTranscript.trim()}"` : "");

    const engine = document.querySelector('input[name="engine"]:checked').value;
    recStatus.textContent = "감정 추론 중...";
    let emo;
    if (engine === "openrouter") {
        emo = await classifyWithOpenRouter(feats, lastTranscript.trim());
    } else {
        emo = classifyHeuristic(feats);
    }
    recStatus.textContent = "완료";
    showEmotion(emo);
    applyEmotion(emo.label, emo.intensity);
}

// 오디오버퍼 → 운율 특징
function extractFeatures(audioBuf) {
    const data = audioBuf.getChannelData(0);
    const sr = audioBuf.sampleRate;
    const frame = Math.floor(sr * 0.04); // 40ms
    const hop = Math.floor(sr * 0.02);
    const pitches = [];
    let energySum = 0, frames = 0, voiced = 0;

    for (let i = 0; i + frame < data.length; i += hop) {
        const seg = data.subarray(i, i + frame);
        let e = 0;
        for (let j = 0; j < seg.length; j++) e += seg[j] * seg[j];
        const rms = Math.sqrt(e / seg.length);
        energySum += rms;
        frames++;
        if (rms < 0.01) continue; // 무음
        const p = autoCorrPitch(seg, sr);
        if (p > 60 && p < 400) { pitches.push(p); voiced++; }
    }

    pitches.sort((a, b) => a - b);
    const pitchMedian = pitches.length ? pitches[Math.floor(pitches.length / 2)] : 0;
    const mean = pitches.reduce((s, v) => s + v, 0) / (pitches.length || 1);
    const pitchStd = Math.sqrt(pitches.reduce((s, v) => s + (v - mean) ** 2, 0) / (pitches.length || 1));

    return {
        energy: energySum / (frames || 1),
        pitchMedian,
        pitchStd: pitches.length ? pitchStd : 0,
        voicedRatio: voiced / (frames || 1),
        speechRate: voiced / ((data.length / sr) || 1) / 10, // 대략적
    };
}

// 자기상관 피치 추정
function autoCorrPitch(buf, sr) {
    const size = buf.length;
    let bestOffset = -1, bestCorr = 0;
    const minP = Math.floor(sr / 400), maxP = Math.floor(sr / 60);
    for (let off = minP; off <= maxP; off++) {
        let corr = 0;
        for (let i = 0; i < size - off; i++) corr += buf[i] * buf[i + off];
        corr /= size - off;
        if (corr > bestCorr) { bestCorr = corr; bestOffset = off; }
    }
    return bestOffset > 0 ? sr / bestOffset : 0;
}

/* ---------- 휴리스틱 분류 (valence/arousal 근사) ---------- */
function classifyHeuristic(f) {
    // 정규화 (대략적 범위)
    const energyN = clamp((f.energy - 0.02) / 0.12, 0, 1);
    const pitchN = clamp((f.pitchMedian - 90) / 160, 0, 1);
    const varN = clamp(f.pitchStd / 60, 0, 1);

    const arousal = clamp(0.5 * energyN + 0.3 * pitchN + 0.2 * varN, 0, 1);
    // valence는 음향만으로 매우 불확실 — 거친 추정
    const valence = clamp(0.5 + 0.4 * (pitchN - 0.5) + 0.3 * (varN - 0.4) - 0.3 * (energyN > 0.7 ? energyN : 0), 0, 1);

    let label;
    if (arousal > 0.6 && valence > 0.55) label = "happy";
    else if (arousal > 0.6 && valence < 0.45) label = "angry";
    else if (arousal > 0.65 && valence >= 0.45 && valence <= 0.55) label = "surprise";
    else if (arousal < 0.4 && valence < 0.45) label = "sad";
    else if (arousal < 0.4) label = "neutral";
    else label = valence >= 0.5 ? "happy" : "sad";

    const intensity = clamp(0.4 + arousal * 0.6, 0.3, 1);
    return { label, intensity, valence, arousal, source: "heuristic" };
}

/* ---------- OpenRouter LLM 분류 ---------- */
async function classifyWithOpenRouter(f, transcript) {
    const key = (document.getElementById("orKey").value || "").trim();
    const model = (document.getElementById("orModel").value || "").trim() || "google/gemini-2.0-flash-001";
    if (!key) {
        alert("OpenRouter API 키를 입력하세요. 휴리스틱으로 대체합니다.");
        return classifyHeuristic(f);
    }
    localStorage.setItem("ve_orKey", key);
    localStorage.setItem("ve_orModel", model);

    const sys = "너는 음향 운율 특징으로 화자의 감정을 추정하는 분석기다. " +
        "반드시 JSON만 출력한다: {\"label\":\"happy|sad|angry|surprise|fear|neutral\",\"intensity\":0~1,\"valence\":0~1,\"arousal\":0~1,\"reason\":\"한 문장\"}. " +
        "수치를 지어내지 말고 주어진 특징에 근거해 판단하라.";
    const user = `음향 특징:\n에너지=${f.energy.toFixed(3)}, 피치중앙값=${f.pitchMedian.toFixed(0)}Hz, ` +
        `피치변동=${f.pitchStd.toFixed(0)}Hz, 유성음비율=${f.voicedRatio.toFixed(2)}, 말속도=${f.speechRate.toFixed(2)}` +
        (transcript ? `\n말 내용: "${transcript}"` : "\n(말 내용 없음 — 운율만으로 판단)");

    try {
        const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
            body: JSON.stringify({
                model,
                messages: [
                    { role: "system", content: sys },
                    { role: "user", content: user },
                ],
                temperature: 0.3,
                response_format: { type: "json_object" },
            }),
        });
        const json = await res.json();
        const text = json.choices?.[0]?.message?.content || "{}";
        const parsed = JSON.parse(text.replace(/```json|```/g, "").trim());
        return {
            label: parsed.label || "neutral",
            intensity: clamp(parsed.intensity ?? 0.7, 0, 1),
            valence: parsed.valence,
            arousal: parsed.arousal,
            reason: parsed.reason,
            source: "openrouter",
        };
    } catch (err) {
        console.error(err);
        alert("OpenRouter 호출 실패 — 휴리스틱으로 대체합니다.\n" + err.message);
        return classifyHeuristic(f);
    }
}

function showEmotion(emo) {
    const names = { happy: "😊 기쁨", sad: "😢 슬픔", angry: "😠 분노", surprise: "😮 놀람", fear: "😨 공포", neutral: "😐 중립" };
    emotionResult.classList.add("active");
    emotionResult.innerHTML =
        `<div class="big">${names[emo.label] || emo.label}</div>` +
        `<div>강도 ${(emo.intensity).toFixed(2)} · 출처 ${emo.source}` +
        (emo.valence != null ? ` · valence ${(+emo.valence).toFixed(2)} · arousal ${(+emo.arousal).toFixed(2)}` : "") +
        `</div>` +
        (emo.reason ? `<div class="muted">${emo.reason}</div>` : "");
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/* ============================================================
 *  수동 테스트 + 옵션 UI
 * ============================================================ */
const intensitySlider = document.getElementById("intensity");
const intensityVal = document.getElementById("intensityVal");
intensitySlider.addEventListener("input", () => {
    intensityVal.textContent = (+intensitySlider.value).toFixed(2);
});
document.querySelectorAll(".btn.emo").forEach((btn) => {
    btn.addEventListener("click", () => {
        const emo = btn.dataset.emo;
        applyEmotion(emo, +intensitySlider.value);
        showEmotion({ label: emo, intensity: +intensitySlider.value, source: "manual" });
    });
});

// 엔진 토글
document.querySelectorAll('input[name="engine"]').forEach((r) => {
    r.addEventListener("change", () => {
        document.getElementById("orBox").classList.toggle("hidden", r.value !== "openrouter" || !r.checked);
    });
});

// 저장된/주입된 키 복원
async function restoreConfig() {
    const cfg = { ...LOCAL_CONFIG };
    // 로컬 전용 키 파일(gitignore)이 있으면 덮어씀. 배포 시엔 없으므로 조용히 무시.
    try {
        const m = await import("./config.local.js");
        Object.assign(cfg, m.LOCAL_CONFIG);
    } catch (_) {}

    const keyEl = document.getElementById("orKey");
    const modelEl = document.getElementById("orModel");
    keyEl.value = localStorage.getItem("ve_orKey") || cfg.OPENROUTER_API_KEY || "";
    modelEl.value = localStorage.getItem("ve_orModel") || cfg.OPENROUTER_MODEL || "";
    // 키가 주입돼 있으면 OpenRouter 모드를 기본 선택
    if (keyEl.value) {
        const orRadio = document.querySelector('input[name="engine"][value="openrouter"]');
        orRadio.checked = true;
        document.getElementById("orBox").classList.remove("hidden");
    }
}
restoreConfig();

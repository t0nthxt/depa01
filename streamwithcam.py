import streamlit as st
import paho.mqtt.client as mqtt
import ssl
import time
import threading
import cv2
import io
import qrcode
import numpy as np
from ultralytics import YOLO
from pathlib import Path



# ==========================================
# CONFIG
# ==========================================
MQTT_BROKER   = "3813e4fbff0844eab72b8875d7a546df.s1.eu.hivemq.cloud"
MQTT_PORT     = 8883
MQTT_USERNAME = "t0nt0n"
MQTT_PASSWORD = "1234567We"
MQTT_TOPIC    = "TP"

YOLO_EXTRA_PATH = Path(r"C:\Users\RICOH-NB145\Downloads\forest_detection_best.pt")
CAMERA_INDEX    = 0

# ==========================================
# DATA STORE
# ==========================================
@st.cache_resource
def get_data_store():
    return {
        "Temp"    : "",
        "Humidity": "",
        "frame"   : None,   # frame ล่าสุดจากกล้อง
    }

data_store = get_data_store()

# ==========================================
# MQTT
# ==========================================
def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8")
    print(f"📨 MQTT: {payload}")
    try:
        temp, rh = payload.split(",")
        data_store["Temp"]     = temp
        data_store["Humidity"] = rh
    except Exception as e:
        print(f"parse error: {e}")

@st.cache_resource
def init_mqtt():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set(tls_version=ssl.PROTOCOL_TLS)
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe(MQTT_TOPIC)
        client.loop_start()
        return client
    except Exception as e:
        st.error(f"MQTT เชื่อมต่อล้มเหลว: {e}")
        return None

# ==========================================
# YOLO THREAD
# ==========================================
@st.cache_resource
def init_yolo():
    model_extra = YOLO(YOLO_EXTRA_PATH)
    model_human = YOLO("yolo11n.pt")
    return model_extra, model_human

@st.cache_resource
def start_yolo_thread():
    def run():
        model_extra, model_human = init_yolo()
        cap = cv2.VideoCapture(CAMERA_INDEX)

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Model 1 — person only (COCO)
            results1 = model_human.predict(
                frame, conf=0.4, classes=[0], verbose=False
            )
            # Model 2 — Axes/chainsaw (custom)
            results2 = model_extra.predict(
                frame, conf=0.2, classes=[0], verbose=False
            )

            # วาด bounding box
            for result in results1 + results2:
                for box in result.boxes:
                    if box.conf[0] < 0.4:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls        = int(box.cls[0])
                    class_name = result.names[cls]
                    conf       = float(box.conf[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame, f"{class_name} {conf:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

            # แปลง BGR → RGB แล้วเก็บใน data_store
            data_store["frame"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t

# เริ่ม MQTT + YOLO
init_mqtt()
start_yolo_thread()

# ==========================================
# UI
# ==========================================
st.set_page_config(
    page_title="Forest Fire Detection",
    page_icon="🌲",
    layout="wide",
)

st.markdown("""
<style>
.card {
    background: #1e1e2e;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    border: 1px solid #2e2e3e;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.card-icon  { font-size: 42px; margin-bottom: 8px; }
.card-label { font-size: 14px; color: #888; letter-spacing: 1px;
              text-transform: uppercase; margin-bottom: 4px; }
.card-value { font-size: 52px; font-weight: 700; line-height: 1.1; }
.card-unit  { font-size: 22px; color: #aaa; margin-left: 4px; }
.card-badge { display: inline-block; margin-top: 12px; padding: 4px 14px;
              border-radius: 99px; font-size: 13px; font-weight: 500; }
.temp-value  { color: #FF6B6B; }
.humid-value { color: #4ECDC4; }
.badge-warn  { background: #3d2a00; color: #FFA94D; }
.badge-ok    { background: #0d2e1a; color: #69DB7C; }
.badge-low   { background: #1a1a3d; color: #74C0FC; }
.status-box  { border-radius: 12px; padding: 16px 24px; text-align: center;
               font-size: 20px; font-weight: 600; margin-top: 8px; }
.alert { background:#3d0000; color:#FF6B6B; border:1px solid #FF6B6B33; }
.warn  { background:#3d2a00; color:#FFA94D; border:1px solid #FFA94D33; }
.safe  { background:#0d2e1a; color:#69DB7C; border:1px solid #69DB7C33; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🌲 Forest Fire Detection")
st.caption(f"Topic: **{MQTT_TOPIC}** — auto refresh ทุก 1 วินาที")
st.divider()

# ดึงค่า sensor
try:
    temp     = float(data_store["Temp"])
    humid    = float(data_store["Humidity"])
    has_data = True
except (ValueError, TypeError):
    temp = humid = 0.0
    has_data = False

temp_badge  = ("badge-warn", "🔥 สูง")  if temp  > 35 else ("badge-ok", "✅ ปกติ")
humid_badge = ("badge-low",  "💧 ต่ำ")  if humid < 40 else ("badge-ok", "✅ ปกติ")

# แบ่ง layout — ซ้าย sensor + status / ขวา กล้อง
left, right = st.columns([1, 1.6])

with left:
    # Sensor cards
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-icon">🌡️</div>
            <div class="card-label">Temperature</div>
            <div class="card-value temp-value">
                {f"{temp:.1f}" if has_data else "--"}
                <span class="card-unit">°C</span>
            </div>
            <div class="card-badge {temp_badge[0]}">{temp_badge[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-icon">💧</div>
            <div class="card-label">Humidity</div>
            <div class="card-value humid-value">
                {f"{humid:.1f}" if has_data else "--"}
                <span class="card-unit">%</span>
            </div>
            <div class="card-badge {humid_badge[0]}">{humid_badge[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Fire status
    if not has_data:
        st.markdown('<div class="status-box warn">⏳ รอรับข้อมูล...</div>',
                    unsafe_allow_html=True)
    elif temp > 38 and humid < 35:
        st.markdown('<div class="status-box alert">🔴 FIRE ALERT — อุณหภูมิสูงมาก!</div>',
                    unsafe_allow_html=True)
    elif temp > 35 and humid < 45:
        st.markdown('<div class="status-box warn">🟠 HIGH RISK — เสี่ยงสูง</div>',
                    unsafe_allow_html=True)
    elif temp > 32 or humid < 55:
        st.markdown('<div class="status-box warn">🟡 WARNING — เฝ้าระวัง</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box safe">🟢 SAFE — ปลอดภัย</div>',
                    unsafe_allow_html=True)

with right:
    st.markdown("##### 📷 Live Camera — YOLO Detection")
    frame = data_store["frame"]
    if frame is not None:
        # st.image(frame, channels="RGB", use_container_width=True)
        st.image(frame, channels="RGB", width='stretch')
    else:
        st.info("⏳ รอกล้อง...")

qr  = qrcode.make("http://192.168.50.168:8501/")
buf = io.BytesIO()
qr.save(buf, format="PNG")
buf.seek(0)

st.image(buf, width=200)
st.caption("http://192.168.50.168:8501/")

time.sleep(1)
st.rerun()
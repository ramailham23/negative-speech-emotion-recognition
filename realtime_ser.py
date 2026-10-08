#!/usr/bin/env python3
# ================================================================
#  REAL-TIME SPEECH EMOTION RECOGNITION — Raspberry Pi 4B
#  UI     : Compact untuk LCD 3.5" (480x320)
#  Model  : CNN 1D MFCC+Δ+ΔΔ (TFLite INT8)
#  Audio  : USB microphone via PyAudio
#  Notif  : Telegram (emosi negatif non-Netral)
# ================================================================

import os, sys, time, threading, queue, logging
from datetime import datetime
import numpy as np
import librosa
import pyaudio
import tkinter as tk
import requests

# ── Konfigurasi ──────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'model', 'cnn1d_ser_int8.tflite')

TELEGRAM_TOKEN   = '8835746307:AAEbcsFZaZTURN2gkDk6gW3FOLhj2nVJN7o'
TELEGRAM_CHAT_ID = '8677436269'
NOTIFY_EMOTIONS  = ['Jijik', 'Marah', 'Sedih', 'Takut']
CONF_THRESHOLD   = 0.55
COOLDOWN_SEC     = 30

SR        = 16000
MIC_SR    = 48000
DURATION  = 3
CHUNK     = 1024
CHANNELS  = 1
FORMAT    = pyaudio.paInt16
SLIDE_SEC = 1.0
N_SAMPLES = SR * DURATION

N_MFCC   = 40
N_FFT    = 512
WIN      = int(round(0.025 * SR))
HOP      = int(round(0.0125 * SR))
PE_COEFF = 0.97
N_FEAT   = N_MFCC * 3

CLASSES = ['Jijik', 'Marah', 'Netral', 'Sedih', 'Takut']

EM_COLORS = {
    'Jijik': '#27AE60', 'Marah': '#E74C3C',
    'Netral': '#3498DB', 'Sedih': '#E67E22', 'Takut': '#8E44AD',
}
EM_EXPR = {
    'Jijik': '(>.<)', 'Marah': '(X.X)',
    'Netral': '(-_-)', 'Sedih': '(T.T)', 'Takut': '(O.O)',
}

EMOTION_PARAMS = {
    'Jijik': [
        ('f0_mean',           '≥ 256 Hz',  lambda v: v >= 256),
        ('spectral_centroid', '≥ 1460 Hz', lambda v: v >= 1460),
        ('energy_rms',        '≤ 0.088',   lambda v: v <= 0.088),
    ],
    'Marah': [
        ('energy_rms',  '≥ 0.085',   lambda v: v >= 0.085),
        ('f0_mean',     '≥ 215 Hz',  lambda v: v >= 215),
        ('speech_rate', '≥ 7.0/det', lambda v: v >= 7.0),
    ],
    'Netral': [
        ('f0_mean', '≤ 215 Hz', lambda v: v <= 215),
        ('f0_std',  '≤ 52 Hz',  lambda v: v <= 52),
        ('jitter',  '≤ 0.035',  lambda v: v <= 0.035),
    ],
    'Sedih': [
        ('f0_mean',           '180–260 Hz', lambda v: 180 <= v <= 260),
        ('f0_std',            '≤ 57 Hz',    lambda v: v <= 57),
        ('spectral_centroid', '≤ 1490 Hz',  lambda v: v <= 1490),
    ],
    'Takut': [
        ('f0_std',  '≥ 60 Hz',  lambda v: v >= 60),
        ('jitter',  '≥ 0.028',  lambda v: v >= 0.028),
        ('shimmer', '≥ 0.14',   lambda v: v >= 0.14),
    ],
}

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')


def load_tflite(path):
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=path)
    except ImportError:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


def pre_emphasis(y):
    return np.append(y[0], y[1:] - PE_COEFF * y[:-1]).astype(np.float32)


def extract_all(y):
    res = {}
    try:    y_t, _ = librosa.effects.trim(y, top_db=20)
    except: y_t = y
    n   = N_SAMPLES
    y_p = y_t[:n] if len(y_t) >= n else np.pad(y_t, (0, n - len(y_t)))
    y_p = pre_emphasis(y_p)

    mfcc = librosa.feature.mfcc(y=y_p, sr=SR, n_mfcc=N_MFCC,
               n_fft=N_FFT, hop_length=HOP, win_length=WIN, window='hamm')
    d1   = librosa.feature.delta(mfcc, order=1)
    d2   = librosa.feature.delta(mfcc, order=2)
    feat = np.hstack([np.mean(mfcc,axis=1), np.mean(d1,axis=1),
                      np.mean(d2,axis=1)])
    res['feat'] = feat.reshape(1, N_FEAT, 1).astype(np.float32)

    try:
        f0, vf, _ = librosa.pyin(y_p, fmin=75, fmax=500, sr=SR)
        f0v = f0[vf] if vf is not None else f0[~np.isnan(f0)]
        res['f0_mean'] = float(np.mean(f0v)) if len(f0v) > 0 else 0.0
        res['f0_std']  = float(np.std(f0v))  if len(f0v) > 0 else 0.0
    except:
        res['f0_mean'] = 0.0; res['f0_std'] = 0.0

    res['energy_rms'] = float(np.sqrt(np.mean(y_p ** 2)))

    try:
        dur = len(y_p) / SR
        env = librosa.onset.onset_strength(y=y_p, sr=SR)
        ons = librosa.onset.onset_detect(onset_envelope=env, sr=SR)
        res['speech_rate'] = float(len(ons) / dur) if dur > 0 else 0.0
    except:
        res['speech_rate'] = 0.0

    try:
        res['spectral_centroid'] = float(np.mean(
            librosa.feature.spectral_centroid(y=y_p, sr=SR)))
    except:
        res['spectral_centroid'] = 0.0

    try:
        frames = librosa.util.frame(y_p, frame_length=512, hop_length=256)
        rms_f  = np.sqrt(np.mean(frames ** 2, axis=0)) + 1e-9
        res['shimmer'] = float(np.mean(np.abs(np.diff(rms_f)) / rms_f[:-1]))
        res['jitter']  = float(np.mean(
            librosa.feature.zero_crossing_rate(y_p))) * 0.05
    except:
        res['shimmer'] = 0.0; res['jitter'] = 0.0

    return res


def send_telegram(emotion, confidence, ac):
    if TELEGRAM_TOKEN == 'ISI_TOKEN_BOT_ANDA':
        return False
    try:
        ts  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = (f"🔔 *Emosi Negatif Terdeteksi*\n"
               f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
               f"😤 *Emosi*      : `{emotion.upper()}`\n"
               f"📊 *Confidence* : `{confidence*100:.1f}%`\n"
               f"🕐 *Waktu*      : `{ts}`\n"
               f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
               f"🔊 *Fitur Akustik:*\n"
               f"  F0 Mean  : `{ac.get('f0_mean',0):.1f} Hz`\n"
               f"  Energy   : `{ac.get('energy_rms',0):.4f}`\n"
               f"  Spectral : `{ac.get('spectral_centroid',0):.0f} Hz`")
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg,
                  'parse_mode': 'Markdown'}, timeout=10)
        return r.status_code == 200
    except:
        return False


class AudioCapture:
    def __init__(self, out_q):
        self.q = out_q; self.running = False
        # Menggunakan MIC_SR untuk ukuran buffer perekaman
        self.buf = np.zeros(MIC_SR * DURATION, dtype=np.float32)
        self.pa  = pyaudio.PyAudio(); self.dev_idx = None
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0 and \
               any(k in info['name'].lower() for k in ['usb','audio','mic']):
                self.dev_idx = i
                logging.info(f"USB audio: [{i}] {info['name']}")
                break

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self): self.running = False

    def _run(self):
        # Merekam menggunakan rate MIC_SR (48000)
        stream = self.pa.open(format=FORMAT, channels=CHANNELS, rate=MIC_SR,
                               input=True, input_device_index=self.dev_idx,
                               frames_per_buffer=CHUNK)
        slide = int(MIC_SR * SLIDE_SEC); sb = []
        while self.running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                sb.extend((np.frombuffer(data, np.int16).astype(np.float32)
                            / 32768.0).tolist())
                if len(sb) >= slide:
                    new = np.array(sb[:slide], dtype=np.float32); sb = sb[slide:]
                    self.buf = np.roll(self.buf, -len(new))
                    self.buf[-len(new):] = new
                    if not self.q.full(): self.q.put(self.buf.copy())
            except Exception as e:
                logging.error(f"Audio: {e}"); time.sleep(0.1)
        stream.stop_stream(); stream.close()

class InferenceEngine:
    def __init__(self, aq, rq):
        self.aq = aq; self.rq = rq; self.running = False
        self.interp, self.inp, self.out = load_tflite(MODEL_PATH)

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self): self.running = False

    def _run(self):
        while self.running:
            try:
                raw_audio = self.aq.get(timeout=2.0)
                
                # --- PROSES RESAMPLE DARI 48000 KE 16000 ---
                audio = librosa.resample(y=raw_audio, orig_sr=MIC_SR, target_sr=SR)
                
                ac    = extract_all(audio)
                self.interp.set_tensor(self.inp[0]['index'], ac['feat'])
                self.interp.invoke()
                probs = self.interp.get_tensor(self.out[0]['index'])[0]
                idx   = int(np.argmax(probs))
                if not self.rq.full():
                    self.rq.put({
                        'emotion': CLASSES[idx],
                        'confidence': float(probs[idx]),
                        'all_probs': dict(zip(CLASSES, probs.tolist())),
                        'acoustic': ac,
                        'ts': datetime.now().strftime('%H:%M:%S'),
                    })
            except queue.Empty: continue
            except Exception as e: logging.error(f"Inference: {e}")


class SERApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Real-Time Speech Emotion Recognition (SER)")
        self.geometry("480x320")
        self.resizable(False, False)
        self.configure(bg='#F0F0F0')
        self.aq = queue.Queue(maxsize=5)
        self.rq = queue.Queue(maxsize=3)
        self.last_notif = 0.0
        self.n_sent = 0
        self._build()
        self._start()
        self.after(200, self._poll)

    def _build(self):
        # Judul
        tk.Label(self,
                 text="Sistem Pendeteksi Emosi Real-Time (1D-CNN)",
                 font=('Helvetica', 10, 'bold'),
                 bg='#F0F0F0', fg='#2C3E50').pack(pady=(4, 2))

        # Main container: 2 kolom
        main = tk.Frame(self, bg='#F0F0F0')
        main.pack(fill='both', expand=True, padx=4)

        # Kiri: kotak emosi + nama + conf
        left = tk.Frame(main, bg='#F0F0F0', width=200)
        left.pack(side='left', fill='y', padx=(0, 4))
        left.pack_propagate(False)

        # Kotak ekspresi
        self.box = tk.Frame(left, bg=EM_COLORS['Netral'],
                            width=196, height=120, relief='flat')
        self.box.pack(pady=(2, 4))
        self.box.pack_propagate(False)
        self.expr_lbl = tk.Label(self.box, text=EM_EXPR['Netral'],
                                  font=('Courier', 28, 'bold'),
                                  bg=EM_COLORS['Netral'], fg='white')
        self.expr_lbl.pack(expand=True)

        # Live indicator
        self.live_lbl = tk.Label(left, text="● MENDENGARKAN...",
                                  font=('Helvetica', 7, 'bold'),
                                  bg='#F0F0F0', fg='#27AE60')
        self.live_lbl.pack()

        # Emosi terdeteksi
        tk.Label(left, text="Emosi Terdeteksi:",
                 font=('Helvetica', 8),
                 bg='#F0F0F0', fg='#555').pack(pady=(4, 0))
        self.em_lbl = tk.Label(left, text="NETRAL",
                                font=('Helvetica', 18, 'bold'),
                                bg='#F0F0F0', fg=EM_COLORS['Netral'])
        self.em_lbl.pack()

        # Conf Score
        tk.Label(left, text="Conf Score:",
                 font=('Helvetica', 8),
                 bg='#F0F0F0', fg='#555').pack(pady=(2, 0))
        self.conf_lbl = tk.Label(left, text="0.0000",
                                  font=('Helvetica', 14, 'bold'),
                                  bg='#F0F0F0', fg='#2C3E50')
        self.conf_lbl.pack()

        # Confidence bar
        bar_bg = tk.Frame(left, bg='#D5D8DC', width=180, height=8)
        bar_bg.pack(pady=(1, 0))
        bar_bg.pack_propagate(False)
        self.conf_bar = tk.Frame(bar_bg, bg=EM_COLORS['Netral'], height=8)
        self.conf_bar.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        # Kanan: Acoustic Score
        right = tk.Frame(main, bg='#F0F0F0')
        right.pack(side='right', fill='both', expand=True)

        tk.Label(right, text="Acoustic Score:",
                 font=('Helvetica', 9, 'bold'),
                 bg='#F0F0F0', fg='#2C3E50').pack(anchor='w', pady=(2, 2))

        tbl = tk.Frame(right, bg='#F0F0F0')
        tbl.pack(fill='x')

        for col, (h, w) in enumerate(
                [('Fitur', 12), ('Nilai', 10), ('Threshold', 10), ('Sts', 4)]):
            tk.Label(tbl, text=h, font=('Helvetica', 7, 'bold'),
                     bg='#D5D8DC', fg='#2C3E50', width=w,
                     padx=2, pady=2).grid(row=0, column=col,
                                          padx=1, pady=1, sticky='ew')

        self.ac_cells = []
        for r in range(3):
            row_cells = []
            for c, w in enumerate([12, 10, 10, 4]):
                lbl = tk.Label(tbl, text="—", font=('Helvetica', 7),
                               bg='white', fg='#555', width=w, padx=2, pady=2)
                lbl.grid(row=r+1, column=c, padx=1, pady=1, sticky='ew')
                row_cells.append(lbl)
            self.ac_cells.append(row_cells)

        # Timestamp notif (di bawah acoustic)
        self.ts_lbl = tk.Label(right, text="",
                                font=('Helvetica', 7),
                                bg='#F0F0F0', fg='#AAA',
                                wraplength=270, justify='left')
        self.ts_lbl.pack(anchor='w', pady=(4, 0))

        # Status bar
        self.status = tk.Label(self, text="Menginisialisasi...",
                                font=('Helvetica', 7),
                                bg='#D5D8DC', fg='#555',
                                anchor='w', padx=4)
        self.status.pack(fill='x', side='bottom', ipady=2)

    def _start(self):
        try:
            self.cap = AudioCapture(self.aq)
            self.eng = InferenceEngine(self.aq, self.rq)
            self.cap.start(); self.eng.start()
            self.status.configure(text="✅ Sistem aktif — mendengarkan audio...")
        except Exception as e:
            self.status.configure(text=f"❌ Error: {e}")

    def _poll(self):
        try:
            while not self.rq.empty():
                self._update(self.rq.get_nowait())
        except queue.Empty:
            pass
        self.after(200, self._poll)

    def _update(self, res):
        em   = res['emotion']
        conf = res['confidence']
        ac   = res['acoustic']
        ts   = res['ts']
        col  = EM_COLORS[em]

        self.box.configure(bg=col)
        self.expr_lbl.configure(text=EM_EXPR[em], bg=col)
        self.em_lbl.configure(text=em.upper(), fg=col)
        self.conf_lbl.configure(text=f"{conf:.4f}")
        self.conf_bar.configure(bg=col)
        self.conf_bar.place_configure(relwidth=conf)

        feat_map = {
            'f0_mean': ac.get('f0_mean',0), 'f0_std': ac.get('f0_std',0),
            'energy_rms': ac.get('energy_rms',0),
            'speech_rate': ac.get('speech_rate',0),
            'spectral_centroid': ac.get('spectral_centroid',0),
            'jitter': ac.get('jitter',0), 'shimmer': ac.get('shimmer',0),
        }
        params = EMOTION_PARAMS.get(em, [])
        for i, row in enumerate(self.ac_cells):
            if i < len(params):
                fname, thr_str, check = params[i]
                val = feat_map.get(fname, 0)
                ok  = check(val)
                if fname in ('f0_mean','f0_std','spectral_centroid'):
                    val_str = f"{val:.1f}Hz"
                elif fname == 'speech_rate':
                    val_str = f"{val:.2f}"
                else:
                    val_str = f"{val:.4f}"
                row[0].configure(text=fname,    fg='#2C3E50', bg='white')
                row[1].configure(text=val_str,
                                  fg='#27AE60' if ok else '#E74C3C', bg='white')
                row[2].configure(text=thr_str,  fg='#555',   bg='white')
                row[3].configure(text='✅' if ok else '❌',
                                  fg='#27AE60' if ok else '#E74C3C', bg='white')
            else:
                for cell in row:
                    cell.configure(text='—', fg='#AAA', bg='#FAFAFA')

        self.status.configure(
            text=f"[{ts}] {em} ({conf*100:.1f}%) | "
                 f"F0={ac.get('f0_mean',0):.1f}Hz | "
                 f"E={ac.get('energy_rms',0):.4f}")

        elapsed = time.time() - self.last_notif
        if (em in NOTIFY_EMOTIONS and conf >= CONF_THRESHOLD
                and elapsed >= COOLDOWN_SEC):
            self.last_notif = time.time()
            def _send():
                ok = send_telegram(em, conf, ac)
                if ok:
                    self.n_sent += 1
                    self.after(0, lambda: self.ts_lbl.configure(
                        text=f"📨 {self.n_sent} notif | terakhir: {ts}"))
            threading.Thread(target=_send, daemon=True).start()

    def on_close(self):
        if hasattr(self, 'cap'): self.cap.stop()
        if hasattr(self, 'eng'): self.eng.stop()
        self.destroy()


if __name__ == '__main__':
    if not os.path.isfile(MODEL_PATH):
        print(f"❌ Model tidak ditemukan: {MODEL_PATH}")
        print("   Letakkan cnn1d_ser_int8.tflite di folder model/")
        sys.exit(1)
    app = SERApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

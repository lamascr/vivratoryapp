from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import uvicorn
import tempfile
import librosa
import numpy as np
import scipy.signal as sig
import pywt
import os
import json
import base64
import io
import urllib.request
from pyoctaveband import octavefilter
import waveform_analysis
from scipy.stats import kurtosis

app = FastAPI(title="AcoustiVis - Advanced Diagnostic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def lee_audio(ruta_fichero, sr=None):
    """
    Lee cualquier archivo de audio usando librosa con fallback a pydub.
    """
    try:
        # Intentar con librosa (maneja la mayoría de formatos si ffmpeg está instalado)
        senyal, fs = librosa.load(ruta_fichero, sr=sr, mono=True)
        return fs, senyal
    except Exception as e:
        print(f"Librosa falló, intentando pydub: {e}")
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(ruta_fichero)
            fs = audio.frame_rate
            audio = audio.set_channels(1)
            senyal = np.array(audio.get_array_of_samples(), dtype=np.float32)
            # Normalizar según profundidad de bits
            if audio.sample_width == 2:
                senyal /= 32768.0
            elif audio.sample_width == 4:
                senyal /= 2147483648.0
            return fs, senyal
        except Exception as e2:
            print(f"Pydub también falló: {e2}")
            return None, None

def downsample(data, max_points=3000):
    if len(data) > max_points:
        factor = len(data) // max_points
        return data[::factor]
    return data

def get_spectrogram_data(senyal, fs):
    f_spec, t_spec, Sxx = sig.spectrogram(senyal, fs, window=sig.get_window('hann', min(len(senyal), 1024)), noverlap=max(1, min(len(senyal)//2, 512)))
    max_t_bins = 600
    if len(t_spec) > max_t_bins:
        factor = len(t_spec) // max_t_bins
        t_spec = t_spec[::factor]
        Sxx = Sxx[:, ::factor]
    return f_spec.tolist(), t_spec.tolist(), np.log10(Sxx + 1e-12).tolist()

def get_cepstrogram_data(senyal, fs):
    # Usar una ventana de ~50ms
    n_win = int(0.05 * fs)
    if n_win < 128: n_win = 128
    hop = n_win // 2
    win = sig.get_window('hann', n_win)
    
    L = 1 + int((len(senyal) - n_win) / hop)
    if L <= 1: return [], [], []
    
    # Solo los primeros 50ms de quefrencia
    n_q = int(0.05 * fs)
    C = np.zeros((n_q, L-1))
    
    for l in range(L-1):
        xw = senyal[l*hop : n_win+l*hop] * win
        # Real cepstrum
        spectrum = np.fft.fft(xw)
        c = np.real(np.fft.ifft(np.log(np.abs(spectrum) + 1e-12)))
        C[:, l] = c[:n_q]
        
    t = np.arange(n_win/2, n_win/2 + (L-1)*hop, hop) / fs
    q = np.arange(n_q) / fs
    
    # Downsample si es muy grande
    max_t_bins = 600
    if len(t) > max_t_bins:
        factor = len(t) // max_t_bins
        t = t[::factor]
        C = C[:, ::factor]
        
    return q.tolist(), t.tolist(), C.tolist()

def get_wavelet_data(senyal, fs):
    s_wave = senyal[:int(fs*0.25)] if len(senyal) > fs * 0.25 else senyal
    N = len(s_wave)
    if N == 0: return [], [], []
    t = np.linspace(0, N/fs, N)
    escalas = np.linspace(3, 100, 30)
    cwt_trans, frecs = pywt.cwt(s_wave, escalas, 'morl', 1/fs)
    Z = np.abs(cwt_trans)
    max_t_bins = 600
    if len(t) > max_t_bins:
        factor = len(t) // max_t_bins
        t = t[::factor]
        Z = Z[:, ::factor]
    return frecs.tolist(), t.tolist(), Z.tolist()

def process_audio_sync(tmp_path, filename, machine_params=None, skip_2d=False):
    fs, senyal = lee_audio(tmp_path)
    if senyal is None:
        return {"error": "Error al procesar el audio"}
        
    max_duration = 30
    if len(senyal) > max_duration * fs:
        senyal = senyal[:int(max_duration * fs)]
    
    Npuntos = len(senyal)
    tmax = Npuntos / fs
    tiempo = np.linspace(0, tmax, Npuntos)
    
    # FFT
    tf_senyal = np.fft.fft(senyal)/Npuntos*2
    fr_senyal = np.fft.fftfreq(Npuntos, 1/fs)
    Nmitad = Npuntos//2 if Npuntos%2==0 else Npuntos//2+1
    frecuencias = fr_senyal[0:Nmitad]
    amplitudes = np.abs(tf_senyal[0:Nmitad])

    # Indicadores
    rms = np.sqrt(np.mean(senyal**2))
    peak = np.max(np.abs(senyal))
    crest_factor = peak / rms if rms != 0 else 0
    kurt_val = kurtosis(senyal, fisher=False)

    # Envolvente
    senyal_hilb = sig.hilbert(senyal)
    envolvente = np.abs(senyal_hilb)
    envolvente_ac = envolvente - np.mean(envolvente)
    tf_env = np.fft.fft(envolvente_ac)/Npuntos*2
    amp_env = np.abs(tf_env[0:Nmitad])

    # Cepstro
    spectrum_log = np.log(np.abs(np.fft.fft(senyal)) + 1e-12)
    ceps = np.abs(np.fft.ifft(spectrum_log).real)
    quefrency = np.arange(len(ceps)) / fs
    # Recortar cepstro para visualización (primeros 100ms es lo típico para ecos/periodicidad)
    ceps_viz = ceps[:int(0.1*fs)]
    q_viz = quefrency[:int(0.1*fs)]

    # Autocorrelación
    s_ac = senyal[:min(len(senyal), 20000)]
    autocorr = sig.correlate(s_ac, s_ac, 'full')
    autocorr = autocorr[len(s_ac)-1:]
    t_ac = np.arange(len(autocorr)) / fs

    # Bandas de Octava
    try:
        spl_oct, freq_oct = octavefilter(downsample(senyal, 40000), fs=fs, fraction=1, order=6, limits=[12, 20000], show=0)
        oct_x = freq_oct[:-1].tolist()
        oct_y = spl_oct[:-1].tolist()
    except:
        oct_x, oct_y = [], []

    # dB y dB(A)
    p0 = 0.00002
    db_val = 20 * np.log10(rms/p0) if rms > 0 else 0
    try:
        y_A = waveform_analysis.A_weight(senyal, fs)
        rms_A = np.sqrt(np.mean(y_A**2))
        db_A_val = 20 * np.log10(rms_A/p0) if rms_A > 0 else 0
    except:
        db_A_val = 0

    # Descriptores
    zcr_arr = librosa.feature.zero_crossing_rate(senyal)[0]
    cent_arr = librosa.feature.spectral_centroid(y=senyal, sr=fs)[0]
    rolloff_arr = librosa.feature.spectral_rolloff(y=senyal, sr=fs)[0]
    bw_arr = librosa.feature.spectral_bandwidth(y=senyal, sr=fs)[0]
    frames_t = librosa.frames_to_time(np.arange(len(zcr_arr)), sr=fs)

    # 2D Plots
    plots_2d = {}
    if not skip_2d:
        spec_y, spec_x, spec_z = get_spectrogram_data(senyal, fs)
        wav_y, wav_x, wav_z = get_wavelet_data(senyal, fs)
        ceps_q, ceps_t, ceps_z = get_cepstrogram_data(senyal, fs)
        plots_2d = {
            "spectrogram": {"x": spec_x, "y": spec_y, "z": spec_z},
            "wavelet": {"x": wav_x, "y": wav_y, "z": wav_z},
            "cepstrogram": {"x": ceps_t, "y": ceps_q, "z": ceps_z}
        }

    # Bearing Fault Frequencies
    fault_freqs = {}
    if machine_params:
        try:
            rpm = float(machine_params.get("rpm", 0))
            n = float(machine_params.get("n", 0))
            d = float(machine_params.get("d", 0))
            D = float(machine_params.get("D", 1))
            alpha = float(machine_params.get("alpha", 0)) * np.pi / 180
            
            f_giro = rpm / 60
            fault_freqs["f_giro"] = f_giro
            if n > 0 and D > 0:
                fault_freqs["BPFO"] = (n/2) * (1 - (d/D) * np.cos(alpha)) * f_giro
                fault_freqs["BPFI"] = (n/2) * (1 + (d/D) * np.cos(alpha)) * f_giro
                fault_freqs["BSF"] = (D/(2*d)) * (1 - (d/D)**2 * np.cos(alpha)**2) * f_giro if d > 0 else 0
                fault_freqs["FTF"] = (1/2) * (1 - (d/D) * np.cos(alpha)) * f_giro
        except: pass

    # Automated Diagnosis
    grid_freq = float(machine_params.get("grid_freq", 50)) if machine_params else 50
    machine_type = machine_params.get("machine_type", "general") if machine_params else "general"
    
    indicators = {
        "rms": float(rms),
        "peak": float(peak),
        "crest_factor": float(crest_factor),
        "kurtosis": float(kurt_val),
        "db": float(db_val),
        "db_a": float(db_A_val)
    }
    diagnosis = get_automated_diagnosis(frecuencias, amplitudes, frecuencias, amp_env, fault_freqs, grid_freq, indicators, machine_type, machine_params)

    return {
        "success": True,
        "filename": filename,
        "duration": tmax,
        "fs": fs,
        "indicators": {
            "rms": float(rms),
            "peak": float(peak),
            "crest_factor": float(crest_factor),
            "kurtosis": float(kurt_val),
            "db": float(db_val),
            "db_a": float(db_A_val)
        },
        "fault_freqs": fault_freqs,
        "plots": {
            "time": {"x": downsample(tiempo).tolist(), "y": downsample(senyal).tolist()},
            "spectrum": {"x": downsample(frecuencias).tolist(), "y": downsample(amplitudes).tolist()},
            "envelope_spectrum": {"x": downsample(frecuencias).tolist(), "y": downsample(amp_env).tolist()},
            "cepstrum": {"x": q_viz.tolist(), "y": ceps_viz.tolist()},
            "autocorr": {"x": downsample(t_ac).tolist(), "y": downsample(autocorr).tolist()},
            "octaves": {"x": oct_x, "y": oct_y},
            "descriptors": {
                "time": downsample(frames_t).tolist(),
                "zcr": downsample(zcr_arr).tolist(),
                "centroid": downsample(cent_arr).tolist(),
                "rolloff": downsample(rolloff_arr).tolist(),
                "bandwidth": downsample(bw_arr).tolist()
            }
        },
        "plots_2d": plots_2d,
        "diagnosis": diagnosis
    }

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    rpm: str = Form("0"),
    n: str = Form("0"),
    d: str = Form("0"),
    D: str = Form("1"),
    alpha: str = Form("0"),
    grid_freq: str = Form("50"),
    machine_type: str = Form("general"),
    pump_vanes: str = Form("0"),
    check_cavitation: str = Form("false"),
    teeth_z1: str = Form("0"),
    teeth_z2: str = Form("0"),
    ice_cylinders: str = Form("0"),
    ice_cycle: str = Form("4"),
    stator_slots: str = Form("0"),
    fan_blades: str = Form("0"),
    pulley_d1: str = Form("0"),
    pulley_d2: str = Form("0"),
    belt_length: str = Form("0")
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        params = {
            "rpm": rpm, "n": n, "d": d, "D": D, "alpha": alpha, 
            "grid_freq": grid_freq, "machine_type": machine_type,
            "pump_vanes": pump_vanes, "check_cavitation": check_cavitation,
            "teeth_z1": teeth_z1, "teeth_z2": teeth_z2,
            "ice_cylinders": ice_cylinders, "ice_cycle": ice_cycle,
            "stator_slots": stator_slots,
            "fan_blades": fan_blades,
            "pulley_d1": pulley_d1, "pulley_d2": pulley_d2,
            "belt_length": belt_length
        }
        data = await run_in_threadpool(process_audio_sync, tmp_path, file.filename, params)
        return JSONResponse(data)
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

@app.post("/analyze-url")
async def analyze_url(request: Request):
    try:
        body = await request.json()
        url = body.get("url")
        if not url: return JSONResponse({"error": "No URL provided"}, status_code=400)
        
        params = body.get("params", {})

        suffix = "." + url.split(".")[-1] if "." in url else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    tmp.write(response.read())
                tmp_path = tmp.name
            except Exception as e_url:
                return JSONResponse({"error": f"No se pudo descargar el archivo: {str(e_url)}"}, status_code=400)

        try:
            data = await run_in_threadpool(process_audio_sync, tmp_path, "remote_file" + suffix, params)
            return JSONResponse(data)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

def detect_peaks(f, a, n_peaks=10):
    # Find local maxima
    # Resolution is fs/N. We want at least a few Hz distance.
    # Use a small fixed number of bins or a small percentage.
    dist = max(5, len(f) // 1000) 
    peaks, _ = sig.find_peaks(a, height=np.max(a)*0.02, distance=dist)
    # Sort by amplitude and take top N
    top_indices = peaks[np.argsort(a[peaks])][-n_peaks:]
    return f[top_indices], a[top_indices]

def get_automated_diagnosis(f, a, f_env, a_env, fault_freqs, grid_freq, indicators, machine_type="general", machine_params=None):
    findings = []
    if len(f) == 0: return findings
    
    # Peak detection settings
    p_f, p_a = detect_peaks(f, a, n_peaks=20)
    pe_f, pe_a = detect_peaks(f_env, a_env, n_peaks=30)
    
    max_amp = np.max(a)
    max_env_amp = np.max(a_env) if len(a_env) > 0 else 1
    f_rot = fault_freqs.get("f_giro", 0)
    kurt = indicators.get("kurtosis", 3)
    cf = indicators.get("crest_factor", 1)
    
    # --- 1. SEVERIDAD ACÚSTICA (ISO 1999 / OSHA) ---
    db_a_val = indicators.get("db_a", 0)
    if db_a_val > 0:
        if db_a_val > 115:
            findings.append({
                "issue": "Ruido Extremadamente Peligroso (ISO 1999 / OSHA)",
                "freq": "Nivel Global",
                "confidence": "Alta",
                "description": f"El nivel de ruido ponderado A es de {db_a_val:.1f} dBA. Supera los 115 dBA de forma inmediata, lo cual es extremadamente peligroso y puede causar daño auditivo permanente instantáneo sin protección."
            })
        elif db_a_val > 90:
            findings.append({
                "issue": "Exposición Crítica de Ruido (Límite OSHA)",
                "freq": "Nivel Global",
                "confidence": "Alta",
                "description": f"El nivel de ruido medido es de {db_a_val:.1f} dBA. Supera el Límite de Exposición Permisible (PEL) de OSHA de 90 dBA para una jornada laboral estándar. Requiere el uso mandatorio de protección auditiva."
            })
        elif db_a_val > 85:
            findings.append({
                "issue": "Nivel de Ruido Elevado (Límite de Acción OSHA / NIOSH)",
                "freq": "Nivel Global",
                "confidence": "Alta",
                "description": f"El nivel de ruido es de {db_a_val:.1f} dBA, superando el nivel de acción preventivo de 85 dBA. Se recomienda implementar medidas de conservación y proveer protección."
            })

    # --- 2. SEVERIDAD DE VIBRACIÓN ESTIMADA (Proxy ISO 10816 / ISO 20816) ---
    rms_val = indicators.get("rms", 0)
    if rms_val > 0:
        if rms_val > 0.04:
            findings.append({
                "issue": "Severidad Operativa Crítica (Zona D - ISO 10816)",
                "freq": "Banda Ancha",
                "confidence": "Alta",
                "description": f"La energía de vibración global estimada (RMS: {rms_val:.4f}) sitúa a la máquina en la Zona D (Inaceptable). Riesgo alto de fallo mecánico inminente. Se aconseja detener la máquina."
            })
        elif rms_val > 0.015:
            findings.append({
                "issue": "Alerta de Severidad Mecánica (Zona C - ISO 10816)",
                "freq": "Banda Ancha",
                "confidence": "Media",
                "description": f"La energía global estimada (RMS: {rms_val:.4f}) sitúa el equipo en la Zona C. El funcionamiento continuo causará fatiga del material y fallos mecánicos. Se aconseja mantenimiento correctivo."
            })
        elif rms_val > 0.005:
            findings.append({
                "issue": "Operación Aceptable (Zona B - ISO 10816)",
                "freq": "Banda Ancha",
                "confidence": "Media",
                "description": f"La máquina opera en la Zona B. Es apta para servicio continuo ilimitado con un ligero incremento en la energía vibratoria de referencia."
            })

    # --- 3. ANÁLISIS ESTADÍSTICO DE IMPACTOS (Crest Factor & Kurtosis) ---
    if kurt > 4.5 and cf > 6.0:
        findings.append({
            "issue": "Detección de Impactos Recurrentes (Daño Mecánico Incipiente)",
            "freq": "Impulsivo",
            "confidence": "Alta",
            "description": f"Con un Factor de Cresta elevado de {cf:.2f} y una Kurtosis de {kurt:.2f}, se evidencia una señal sumamente impulsiva con impactos transitorios cíclicos. Esto es síntoma de daños mecánicos tempranos (pitting/desconchado en rodamientos, o dientes de engranaje fisurados)."
        })
    elif kurt > 3.8:
        findings.append({
            "issue": "Señal Impulsiva Anómala",
            "freq": "Impulsivo",
            "confidence": "Media",
            "description": f"La curtosis de {kurt:.2f} (superior a la normal de 3.0) indica la presencia de impulsos transitorios breves o rozamientos intermitentes en la máquina."
        })

    # --- 4. HOLGURA MECÁNICA (Tren de Armónicos y Subarmónicos) ---
    if f_rot > 0:
        harmonics_looseness = []
        for mult in range(1, 9):
            f_target = f_rot * mult
            matches = [i for i, freq in enumerate(p_f) if abs(freq - f_target) < 0.02 * f_target]
            if matches:
                harmonics_looseness.append(mult)
        
        subharmonics_found = []
        for mult in [0.5, 1.5, 2.5]:
            f_target = f_rot * mult
            matches = [i for i, freq in enumerate(p_f) if abs(freq - f_target) < 0.02 * f_target]
            if matches:
                subharmonics_found.append(mult)
                
        if len(harmonics_looseness) >= 4 or len(subharmonics_found) >= 1:
            confidence = "Alta" if (len(harmonics_looseness) >= 5 or len(subharmonics_found) >= 2) else "Media"
            desc = f"Se detecta un patrón de armónicos recurrentes {harmonics_looseness}"
            if subharmonics_found:
                desc += f" y subarmónicos de giro {subharmonics_found}"
            desc += f" de la frecuencia de rotación ({f_rot:.2f} Hz). Síntoma directo de holgura mecánica estructural (soltura de pernos de anclaje, grietas de bancada o desajuste en los alojamientos de rodamientos)."
            findings.append({
                "issue": "Holgura Mecánica (Mechanical Looseness)",
                "freq": f"{f_rot:.2f} Hz",
                "confidence": confidence,
                "description": desc
            })

    # --- 5. DETECCIONES COMUNES (Unbalance & Misalignment) ---
    
    # Desequilibrio (1x RPM)
    if f_rot > 0:
        matches = [i for i, freq in enumerate(p_f) if abs(freq - f_rot) < 0.02 * f_rot]
        if matches:
            amp = p_a[matches[0]]
            confidence = "Alta" if amp > 0.4 * max_amp else "Media"
            findings.append({
                "issue": "Posible Desequilibrio de Masa (1x RPM)",
                "freq": f"{f_rot:.2f} Hz",
                "confidence": confidence,
                "description": f"Pico dominante detectado exactamente en la frecuencia de giro ({f_rot:.2f} Hz), correspondiente a las especificaciones de la norma ISO 1940-1 para desequilibrio dinámico."
            })

    # Desalineación (2x, 3x RPM)
    if f_rot > 0:
        for mult in [2, 3]:
            f_target = f_rot * mult
            matches = [i for i, freq in enumerate(p_f) if abs(freq - f_target) < 0.02 * f_target]
            if matches:
                findings.append({
                    "issue": f"Posible Desalineación de Ejes ({mult}x RPM)",
                    "freq": f"{f_target:.2f} Hz",
                    "confidence": "Media",
                    "description": f"Componente armónico detectado a {mult}x la frecuencia de giro ({f_target:.2f} Hz). Común en desalineaciones de tipo angular o paralela en acoplamientos."
                })

    # --- 6. ANÁLISIS DE RODAMIENTOS EN ENVOLVENTE CON BANDAS LATERALES ---
    for fault, freq in fault_freqs.items():
        if fault == "f_giro" or freq <= 0: continue
        harmonics_found = []
        for h in range(1, 5):
            f_target = freq * h
            matches = [i for i, p_freq in enumerate(pe_f) if abs(p_freq - f_target) < 0.03 * f_target]
            if matches: harmonics_found.append(h)
        
        if len(harmonics_found) >= 1:
            sidebands_found = []
            if f_rot > 0:
                for h_idx in harmonics_found:
                    f_center = freq * h_idx
                    for sb in [-1, 1]:
                        f_sb = f_center + sb * f_rot
                        matches_sb = [i for i, p_freq in enumerate(pe_f) if abs(p_freq - f_sb) < 0.03 * f_sb]
                        if matches_sb:
                            sidebands_found.append(f"{h_idx}x{fault}{'+' if sb > 0 else ''}{sb}x RPM")
            
            confidence = "Baja"
            if len(harmonics_found) >= 2: confidence = "Media"
            if len(harmonics_found) >= 3 or (len(harmonics_found) >= 2 and (kurt > 4 or len(sidebands_found) >= 1)): confidence = "Alta"
            
            desc = f"Se detectan los armónicos {harmonics_found} de la frecuencia de fallo {fault} ({freq:.2f} Hz) en el espectro de la envolvente."
            if sidebands_found:
                desc += f" Asimismo, se registran bandas laterales de giro {sidebands_found}, confirmando un fallo avanzado con modulación de amplitud a la velocidad de giro."
            
            findings.append({
                "issue": f"Defecto en Rodamiento ({fault})",
                "freq": f"{freq:.2f} Hz",
                "confidence": confidence,
                "description": desc
            })

    # --- 7. DIAGNÓSTICO ESPECÍFICO POR MÁQUINA ---

    # Motores Eléctricos
    if machine_type == "motor" and grid_freq > 0:
        for mult in [1, 2]:
            f_target = grid_freq * mult
            matches = [i for i, freq in enumerate(p_f) if abs(freq - f_target) < 0.02 * f_target]
            if matches:
                findings.append({
                    "issue": f"Origen Electromagnético ({f_target} Hz)" if mult == 1 else "Asimetría / Excentricidad Estatórica (2x Frec. Red)",
                    "freq": f"{f_target:.2f} Hz",
                    "confidence": "Alta",
                    "description": f"Pico detectado a {mult}x la frecuencia de red ({grid_freq} Hz). Frecuentemente originado por asimetrías de flujo magnético, entrehierro variable o problemas de devanados."
                })
        
        # Broken rotor bars slip sidebands around grid frequency
        if f_rot > 0:
            rotor_sb = []
            for delta in [-3.0, -1.5, 1.5, 3.0]:
                f_target = grid_freq + delta
                matches_sb = [i for i, freq in enumerate(p_f) if abs(freq - f_target) < 0.02 * f_target]
                if matches_sb:
                    rotor_sb.append(f"{f_target:.2f} Hz")
            if rotor_sb:
                findings.append({
                    "issue": "Modulación de Deslizamiento (Posible Rotura de Barras del Rotor)",
                    "freq": f"{grid_freq:.2f} Hz",
                    "confidence": "Media",
                    "description": f"Se detectan bandas laterales del deslizamiento ({rotor_sb}) rodeando la frecuencia de red ({grid_freq} Hz). Síntoma típico de barras rotas o agrietadas en el rotor."
                })

        # Stator Slots
        slots = float(machine_params.get("stator_slots", 0) or 0)
        if slots > 0 and f_rot > 0:
            f_pass = slots * f_rot
            matches = [i for i, freq in enumerate(p_f) if abs(freq - f_pass) < 0.02 * f_pass]
            if matches:
                findings.append({
                    "issue": "Frecuencia de Paso de Ranuras del Estator",
                    "freq": f"{f_pass:.2f} Hz",
                    "confidence": "Media",
                    "description": f"Pico detectado en la frecuencia de paso de ranuras estatóricas ({slots}x RPM). Indica interacción ranura-diente anómala debido a excentricidad."
                })

    # Bombas Centrífugas
    elif machine_type == "bomba":
        vanes = float(machine_params.get("pump_vanes", 0) or 0)
        if vanes > 0 and f_rot > 0:
            f_vpf = vanes * f_rot
            matches = [i for i, freq in enumerate(p_f) if abs(freq - f_vpf) < 0.02 * f_vpf]
            if matches:
                findings.append({
                    "issue": "Frecuencia de Paso de Álabes (VPF)",
                    "freq": f"{f_vpf:.2f} Hz",
                    "confidence": "Media",
                    "description": f"Componente dominante detectada a la frecuencia VPF ({vanes}x RPM). Niveles excesivos indican problemas de desgaste en el rodete, turbulencias en la voluta o restricciones en conductos."
                })
        
        # Cavitación
        if machine_params.get("check_cavitation") == "true" or machine_params.get("check_cavitation") is True:
            mask_high = (f > 5000) & (f < 10000)
            mask_low = (f > 100) & (f < 1000)
            if np.any(mask_high) and np.any(mask_low):
                energy_high = np.mean(a[mask_high])
                energy_low = np.mean(a[mask_low])
                if energy_high > 0.3 * energy_low:
                    findings.append({
                        "issue": "Posible Cavitación en Bomba",
                        "freq": "5000 - 10000 Hz",
                        "confidence": "Alta" if energy_high > 0.5 * energy_low else "Media",
                        "description": "Se detecta un incremento drástico del ruido de banda ancha en alta frecuencia. Esto corresponde al colapso de burbujas de vapor de agua por cavitación, lo cual daña gravemente las caras de los álabes."
                    })
        
        # Recirculación
        if f_rot > 0:
            mask_recirc = (f > 2) & (f < 0.8 * f_rot)
            if np.any(mask_recirc):
                energy_recirc = np.mean(a[mask_recirc])
                if energy_recirc > 0.25 * max_amp:
                    findings.append({
                        "issue": "Recirculación de Flujo (Sub-sincrónica)",
                        "freq": "Sub-sincrónica (< 1x RPM)",
                        "confidence": "Media",
                        "description": "Se registra ruido turbulento a frecuencias muy bajas (por debajo de 1x RPM). Es indicativo de inestabilidades hidráulicas y flujo inverso por operar fuera del punto óptimo de la bomba."
                    })

    # Ventiladores
    elif machine_type == "ventilador":
        blades = float(machine_params.get("fan_blades", 0) or 0)
        if blades > 0 and f_rot > 0:
            f_bpf = blades * f_rot
            matches = [i for i, freq in enumerate(p_f) if abs(freq - f_bpf) < 0.02 * f_bpf]
            if matches:
                findings.append({
                    "issue": "Frecuencia de Paso de Álabes (BPF)",
                    "freq": f"{f_bpf:.2f} Hz",
                    "confidence": "Media",
                    "description": f"Pico registrado en BPF ({blades}x RPM). Si su amplitud aumenta progresivamente, denota problemas de turbulencia en la descarga, desgaste de álabes o mala orientación del conducto de entrada."
                })

    # Reductoras / Engranajes
    elif machine_type == "engranaje":
        z1 = float(machine_params.get("teeth_z1", 0) or 0)
        z2 = float(machine_params.get("teeth_z2", 0) or 0)
        if z1 > 0 and f_rot > 0:
            f_gmf = z1 * f_rot
            gmf_found = False
            gmf_freq = 0
            for mult in [1, 2]:
                f_target = f_gmf * mult
                matches = [i for i, freq in enumerate(p_f) if abs(freq - f_target) < 0.02 * f_target]
                if matches:
                    gmf_found = True
                    gmf_freq = p_f[matches[0]]
                    break
            
            if gmf_found:
                sidebands = []
                for sb_mult in [-2, -1, 1, 2]:
                    f_sb = gmf_freq + sb_mult * f_rot
                    matches_sb = [i for i, freq in enumerate(p_f) if abs(freq - f_sb) < 0.02 * f_sb]
                    if matches_sb:
                        sidebands.append(f"{sb_mult}x RPM")
                
                if sidebands:
                    findings.append({
                        "issue": "Desgaste de Dientes / Carga Asimétrica (GMF con Bandas Laterales)",
                        "freq": f"{gmf_freq:.2f} Hz",
                        "confidence": "Alta",
                        "description": f"Pico en la Frecuencia de Engrane ({gmf_freq:.2f} Hz) acompañado de bandas de modulación laterales en {sidebands}. Es una confirmación de modulación de amplitud y frecuencia, causada por desgaste de perfiles de dientes, desalineación o excentricidad de ejes."
                    })
                else:
                    findings.append({
                        "issue": "Frecuencia de Engrane Activa (GMF)",
                        "freq": f"{gmf_freq:.2f} Hz",
                        "confidence": "Media",
                        "description": f"Pico en GMF ({gmf_freq:.2f} Hz). Indica acoplamiento de carga entre engranajes. Si no hay bandas laterales, el desgaste de dientes es actualmente bajo."
                    })

    # Correas y Poleas
    elif machine_type == "correa":
        d1 = float(machine_params.get("pulley_d1", 0) or 0)
        d2 = float(machine_params.get("pulley_d2", 0) or 0)
        L = float(machine_params.get("belt_length", 0) or 0)
        if d1 > 0 and d2 > 0 and f_rot > 0:
            f_p2 = f_rot * d1 / d2
            matches = [i for i, freq in enumerate(p_f) if abs(freq - f_p2) < 0.02 * f_p2]
            if matches:
                findings.append({
                    "issue": "Velocidad de Polea Conducida",
                    "freq": f"{f_p2:.2f} Hz",
                    "confidence": "Media",
                    "description": f"Pico detectado en la velocidad rotacional de la polea secundaria ({f_p2:.2f} Hz)."
                })
        if d1 > 0 and L > 0 and f_rot > 0:
            f_belt = (np.pi * d1 * f_rot) / L
            matches = [i for i, freq in enumerate(p_f) if abs(freq - f_belt) < 0.02 * f_belt]
            if matches:
                findings.append({
                    "issue": "Frecuencia de Paso de Correa Activa",
                    "freq": f"{f_belt:.2f} Hz",
                    "confidence": "Media",
                    "description": f"Pico a la frecuencia de correa ({f_belt:.2f} Hz). Componentes elevadas o armónicos denotan grietas en la correa, holguras o desalineación de las poleas."
                })

    # Motores de Combustión
    elif machine_type == "combustion":
        cyl = float(machine_params.get("ice_cylinders", 0) or 0)
        cycle = float(machine_params.get("ice_cycle", 4) or 4)
        if cyl > 0 and f_rot > 0:
            f_fire = (f_rot * cyl) / (2 if cycle == 4 else 1)
            matches = [i for i, freq in enumerate(p_f) if abs(freq - f_fire) < 0.02 * f_fire]
            if matches:
                f_misfire = f_fire * 0.5
                matches_misfire = [i for i, freq in enumerate(p_f) if abs(freq - f_misfire) < 0.02 * f_misfire]
                if matches_misfire:
                    findings.append({
                        "issue": "Fallo de Encendido / Combustión Inestable (Misfire)",
                        "freq": f"{f_misfire:.2f} Hz",
                        "confidence": "Alta",
                        "description": f"Fuerte pico detectado a la mitad de la frecuencia de encendido ({f_misfire:.2f} Hz). Característico de fallos de encendido (misfires) en cilindros específicos o problemas de válvulas."
                    })
                else:
                    findings.append({
                        "issue": "Frecuencia de Combustión Dominante",
                        "freq": f"{f_fire:.2f} Hz",
                        "confidence": "Alta",
                        "description": f"Pico dominante detectado a la frecuencia de encendido de la combustión ({f_fire:.2f} Hz). Denota que el régimen de detonaciones es uniforme y estable."
                    })

    # 8. Unidentified periodic impacts (fallback)
    if not any(f["confidence"] == "Alta" for f in findings):
        top_env_idx = np.argmax(pe_a) if len(pe_a) > 0 else -1
        if top_env_idx != -1:
            f_top = pe_f[top_env_idx]
            amp_top = pe_a[top_env_idx]
            if f_top > 5 and amp_top > 0.2 * max_env_amp:
                h_matches = [i for i, p_f in enumerate(pe_f) if abs(p_f - 2*f_top) < 0.03 * (2*f_top)]
                if h_matches or kurt > 5:
                    findings.append({
                        "issue": "Impactos Periódicos no Identificados en Envolvente",
                        "freq": f"{f_top:.2f} Hz",
                        "confidence": "Media",
                        "description": f"Se registra un pico dominante a {f_top:.2f} Hz en el espectro de envolvente sin correspondencia con los parámetros cargados de la máquina. Indica la presencia de roces o impactos cíclicos en el sistema."
                    })

    return findings

@app.post("/analyze_chunk")
async def analyze_chunk(
    file: UploadFile = File(...),
    rpm: str = Form("0"),
    n: str = Form("0"),
    d: str = Form("0"),
    D: str = Form("1"),
    alpha: str = Form("0"),
    grid_freq: str = Form("50"),
    machine_type: str = Form("general"),
    pump_vanes: str = Form("0"),
    check_cavitation: str = Form("false"),
    teeth_z1: str = Form("0"),
    teeth_z2: str = Form("0"),
    ice_cylinders: str = Form("0"),
    ice_cycle: str = Form("4"),
    stator_slots: str = Form("0"),
    fan_blades: str = Form("0"),
    pulley_d1: str = Form("0"),
    pulley_d2: str = Form("0"),
    belt_length: str = Form("0")
):
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        params = {
            "rpm": rpm, "n": n, "d": d, "D": D, "alpha": alpha, 
            "grid_freq": grid_freq, "machine_type": machine_type,
            "pump_vanes": pump_vanes, "check_cavitation": check_cavitation,
            "teeth_z1": teeth_z1, "teeth_z2": teeth_z2,
            "ice_cylinders": ice_cylinders, "ice_cycle": ice_cycle,
            "stator_slots": stator_slots,
            "fan_blades": fan_blades,
            "pulley_d1": pulley_d1, "pulley_d2": pulley_d2,
            "belt_length": belt_length
        }
        data = await run_in_threadpool(process_audio_sync, tmp_path, file.filename, params, True)
        return JSONResponse(data)
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

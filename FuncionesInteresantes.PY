#@slideshow skip
#@title Módulos (o librerías, o paquetes) que hay que cargar previamente
# Si no ejecutas esta celda no vas a poder interactuar con las celdas que vendrán a continuación
import subprocess
import sys
import glob
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# Instalar librería pydub: sirve para leer multitud de ficheros de audio
install('pydub')
# Instalar librería yt-dlp: sirve para descargar audios de youtube
#install('yt-dlp[default]')
#import yt_dlp
# Instalar librería PyWavelets: sirve para usar la transformada de Wavelet
install('PyWavelets')
from IPython.display import IFrame
from IPython.display import Image
from IPython.display import Audio
from IPython.display import YouTubeVideo
from IPython.core.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import animation
import matplotlib.mlab as ml
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Instalando pyOctaveBand
install('PyOctaveBand')
from ipywidgets import interactive
import scipy.io.wavfile as wavfile # sirve para leer ficheros WAV
import librosa # sirve para leer ficheros MP4
import scipy.signal as sig # hilbert, espectrograma, chirp
import pywt # sirve para la transformada de wavelet: pip install PyWavelets
import pandas as pd
import urllib.request
import warnings
warnings.filterwarnings('ignore')
# Instalar librería ipympl: sirve para interactuar con las figuras
from bokeh.plotting import figure, show, output_notebook
# install('ipympl')
from pyoctaveband import octavefilter # para espectros en octavas y tercios de octava
# Instalando waveform-analysis - para ponderaciones A, B y C
install('git+https://github.com/endolith/waveform-analysis.git@master')
import waveform_analysis

def displayhoriz(*args):
    html_str=''
    for df in args:
        html_str+=df._repr_html_() # Using _repr_html_ directly instead of eval
    display(HTML(data=html_str))
class displayhoriz2(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">{1}</div>"""
    def __init__(self, *args):
        self.args = args
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_()) for a in self.args)
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a)) for a in self.args)

import pydub
from google.colab import output
output.enable_custom_widget_manager()

# Funciones que facilitan el análisis

def lee_audio_pydub(file_name):
  asg = pydub.AudioSegment.from_file(file_name)
  dtype = getattr(np, "int{:d}".format(asg.sample_width * 8))  # Or could create a mapping: {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}
  arr = np.ndarray((int(asg.frame_count()), asg.channels), buffer=asg.raw_data, dtype=dtype)
  # print("\n", asg.frame_rate, arr.shape, arr.dtype, arr.size, len(asg.raw_data), len(asg.get_array_of_samples()))  # @TODO: Comment this line!!!
  if len(np.shape(arr)) > 1:
    arr = arr[:, 0] # grabación en estéreo: nos quedamos con solo un canal
  if arr.dtype == 'int16':
    arr = arr.astype(np.float32, order='C') / 32768.0
  return asg.frame_rate, arr.flatten()

def lee_audio_local(fich):
  fs, senyal = wavfile.read(fich)
  if len(np.shape(senyal)) > 1:
    senyal = senyal[:, 0] # grabación en estéreo: nos quedamos con solo un canal
  if senyal.dtype == 'int16':
    senyal = senyal.astype(np.float32, order='C') / 32768.0
  return fs, senyal

def lee_audio_mp4(fich):
  # Leemos el fichero mp4 usando la libraría librosa
  senyal, fs = librosa.load(fich)
  if len(np.shape(senyal)) > 1:
    senyal = senyal[:, 0] # grabación en estéreo: nos quedamos con solo un canal
  return fs, senyal

def lee_audio_wav(fich):
  fs, senyal = wavfile.read(fich)
  if len(np.shape(senyal)) > 1:
    senyal = senyal[:, 0] # grabación en estéreo: nos quedamos con solo un canal
  if senyal.dtype == 'int16':
    senyal = senyal.astype(np.float32, order='C') / 32768.0
  return fs, senyal

def lee_audio(ruta_fichero, sr=None):
    """
    Lee cualquier archivo de audio (wav, mp4, m4a, ogg, opus, etc.)
    y lo devuelve normalizado en float32 y en mono.
    
    Args:
        ruta_fichero: Path al archivo.
        sr: Frecuencia de muestreo deseada. Si es None, usa la original.
    """
    try:
        # librosa.load hace casi todo el trabajo sucio por ti:
        # - 'sr=None' mantiene la frecuencia de muestreo original
        # - 'mono=True' mezcla los canales automáticamente si es estéreo
        # - Convierte automáticamente a float32 y normaliza entre [-1.0, 1.0]
        senyal, fs = librosa.load(ruta_fichero, sr=sr, mono=True)
        
        return fs, senyal
        
    except Exception as e:
        print(f"Error al leer el archivo {ruta_fichero}: {e}")
        return None, None

def lee_audio_servidor(url):
  direccion = url
  fichero = direccion.split('/')[-1]
  # Descarga el fichero
  fichin = urllib.request.urlopen(direccion)
  with open(fichero,'wb') as fichout:
    fichout.write(fichin.read())
  # Lectura de un fichero de audio (en formato WAV)
  fs, senyal = wavfile.read(fichero)
  if len(np.shape(senyal)) > 1:
    senyal = senyal[:, 0] # grabación en estéreo: nos quedamos con solo un canal
  if senyal.dtype == 'int16':
    senyal = senyal.astype(np.float32, order='C') / 32768.0
  Npuntos = len(senyal)
  # print('Se ha descargado el fichero {} \n de la dirección {}'.format(fichero, direccion))
  # print('La señal consta de {} puntos y la frecuencia de muestreo es de {} Hz'.format(Npuntos, fs))
  return fs, senyal

def grafica_tiempo(listasen, listafs, sizeX = 10, sizeY = 2, interactiva = False):
  if interactiva:
    #% settings for bokeh
    output_notebook()
    # create a new plot with a title and axis labels
    ax = figure(max_width=sizeX*100, height=sizeY*100)
    if type(listafs) is not list:
      listafs = [listafs] * len(listasen)
    for senyal, fs in zip(listasen, listafs):
      Npuntos = len(senyal)
      tmax = Npuntos /fs # tmax = Npuntos x t_muestreo = Npuntos / f_muestreo
      tiempo = np.linspace(0, tmax, Npuntos) # creamos el vector de tiempos
      ax.line(tiempo, senyal, alpha=0.7)
    ax.xaxis.axis_label = 'tiempo [s]'
    show(ax)
  else:
    #%matplotlib inline
    get_ipython().run_line_magic('matplotlib', 'inline')
    fig = plt.figure(figsize=(sizeX, sizeY))
    fig.canvas.header_visible = False # Hide the Figure name at the top
    fig.canvas.footer_visible = True # Show/Hide the footer
    fig.canvas.toolbar_position = 'right' # Change the toolbar position
    ax = fig.add_subplot(1, 1, 1)
    if type(listafs) is not list:
      listafs = [listafs] * len(listasen)
    for senyal, fs in zip(listasen, listafs):
      Npuntos = len(senyal)
      tmax = Npuntos /fs # tmax = Npuntos x t_muestreo = Npuntos / f_muestreo
      tiempo = np.linspace(0, tmax, Npuntos) # creamos el vector de tiempos
      ax.plot(tiempo, senyal, alpha=0.7)
    ax.set_xlabel('tiempo [s]')
    fig.tight_layout()
  return ax

def calcula_db(senyal):
  # Cálculo de los decibelios
  # Calculo del valor cuadrático medio:
  RMS_senyal = np.sqrt(np.mean(senyal**2))
  p0 = 0.00002 # Pa (nivel de referencia)
  # De la definición de nivel de presión sonora:
  return 20 * np.log10(RMS_senyal/p0)

def calcula_envolvente(senyal):
  # Cálculo de la transformada de Hilbert (son números complejos)
  senyal_hilb = sig.hilbert(senyal)
  # Cálculo de la envolvente (módulo de la transformada de Hilbert)
  envolvente = np.abs(senyal_hilb)
  return envolvente

def calcula_envolvente2(senyal, fs, tipo='arriba', windowsize=11):
  Npuntos = len(senyal)
  tmax = Npuntos/fs
  tiempo = np.linspace(0,tmax,Npuntos)
  df=pd.DataFrame(data={"y":senyal},index=tiempo)
  if tipo == 'arriba':
    envolvente=df["y"].rolling(window=windowsize).max().shift(int(-windowsize/2))
  else:
    envolvente=df["y"].rolling(window=windowsize).min().shift(int(-windowsize/2))
  return np.nan_to_num(envolvente.values)

def haz_espectro(senyal, fs):
  # La señal está guardada en el vector senyal
  # La frecuencia de muestreo en fs
  Npuntos = len(senyal)
  # Cálculo de la transformada de Fourier:
  tf_senyal = np.fft.fft(senyal)/Npuntos*2
  fr_senyal = np.fft.fftfreq(Npuntos, 1/fs)
  Nmitad = Npuntos//2 if Npuntos%2==0 else Npuntos//2+1
  return fr_senyal[0:Nmitad], np.abs(tf_senyal[0:Nmitad])

def grafica_espectro(listasen, listafs, sizeX = 10, sizeY = 4, interactiva = False):
  if type(listafs) is not list:
    listafs = [listafs] * len(listasen)
  if interactiva:
    #% settings for bokeh
    output_notebook()
    TOOLTIPS = [
      ("(frec,y)", "($x, $y)"),
    ]
    # create a new plot with a title and axis labels
    ax = figure(max_width=sizeX*100, height=sizeY*100, tooltips=TOOLTIPS)
    for senyal, fs in zip(listasen, listafs):
      Npuntos = len(senyal)
      # Cálculo de la transformada de Fourier:
      fr_, tf_abs_ = haz_espectro(senyal, fs)
      ax.line(fr_, tf_abs_, alpha=0.7)
    ax.xaxis.axis_label = 'frecuencia [Hz]'
    show(ax)
  else:
    #%matplotlib inline
    get_ipython().run_line_magic('matplotlib', 'inline')
    fig = plt.figure(figsize=(sizeX, sizeY))
    fig.canvas.header_visible = False # Hide the Figure name at the top
    fig.canvas.footer_visible = True # Show/Hide the footer
    fig.canvas.toolbar_position = 'right' # Change the toolbar position
    ax = fig.add_subplot(1, 1, 1)
    for senyal, fs in zip(listasen, listafs):
      Npuntos = len(senyal)
      # Cálculo de la transformada de Fourier:
      fr_, tf_abs_ = haz_espectro(senyal, fs)
      ax.plot(fr_, tf_abs_, alpha=0.7)
    ax.set_xlabel('frecuencia [Hz]')
    fig.tight_layout()
  return ax

def grafica_espectrograma(listasen,fs,lista_tam,lista_sol,sizeX=8,sizeY=8,
                          log=True,interactiva=False,vlim=None):
  if interactiva:
    #%matplotlib widget
    get_ipython().run_line_magic('matplotlib', 'widget')
  else:
    #%matplotlib inline
    get_ipython().run_line_magic('matplotlib', 'inline')
  fig, axs = plt.subplots(1, len(listasen),figsize=(sizeX, sizeY), squeeze=False)
  fig.canvas.header_visible = False # Hide the Figure name at the top
  fig.canvas.footer_visible = True # Show/Hide the footer
  fig.canvas.toolbar_position = 'right' # Change the toolbar position
  ax = 0
  for senyal in listasen:
    tamanyo_ventana = int(lista_tam[ax])
    solape_ventana = int(lista_sol[ax])
    f1, t1, Sxx1 = sig.spectrogram(senyal, fs, window=sig.get_window('hann', tamanyo_ventana), noverlap=solape_ventana)
    if log:
      if vlim == None:
        qm = axs[0,ax].pcolormesh(t1, f1, np.log10(Sxx1))
      else:
        qm = axs[0,ax].pcolormesh(t1, f1, np.log10(Sxx1),vmin=vlim[0],vmax=vlim[1])
    else:
      if vlim == None:
        qm = axs[0,ax].pcolormesh(t1, f1, Sxx1)
      else:
        qm = axs[0,ax].pcolormesh(t1, f1, Sxx1,vmin=vlim[0],vmax=vlim[1])
    axs[0,ax].set_xlabel('tiempo [s]')
    axs[0,ax].set_ylabel('frecuencia [Hz]')
    plt.colorbar(qm)
    ax += 1
  fig.tight_layout()
  return axs

def grafica_tiempo_espectro_espectrograma(senyal, fs, tdeseado, fmax, tamanyo_ventana, solape_ventana, sizeX = 18, sizeY = 5):
    '''
    Función que realiza un gráfico con la señal temporal, su espectro y su espectrograma
    Toma varios inputs:
    - fs: frecuencia de muestreo de la señal
    - senyal: la señal leída a partir de un fichero WAV (puede estar grabada en estéreo)
    - tdeseado: duración de la señal que queremos procesar (por si la señal es muy larga)
    - fmax: frecuencia máxima que se mostrará en espectro y espectrograma
    - tamanyo_ventana: tamaño de la ventana para el espectrograma
    - solape_ventana: tamaño del solape en el espectrograma
    '''
    if len(np.shape(senyal)) > 1:
        senyal = senyal[:, 0] # grabación en estéreo: nos quedamos con solo un canal
    N = len(senyal)
    tmax = N / fs
    if tmax > tdeseado:
        # Recortamos la señal para no colapsar el ordenador
        N = int(N * tdeseado/tmax)
        tmax = N / fs
        senyal = senyal[0:N]
    print(f"Duración: {tmax:.2f} segundos. Frec. muestreo: {fs:.2f} Hz. Puntos: {N}")
    t = np.linspace(0, tmax, N)
    # Cálculo de la transformada de Fourier y el vector de frecuencias
    tf_senyal = np.fft.fft(senyal) /N
    frec = np.fft.fftfreq(N, 1/fs)
    Nmitad = N//2 if N%2==0 else N//2+1
    # Gráficas
    fig = plt.figure(figsize=(sizeX,sizeY))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(t, senyal)
    ax1.set_ylabel('Señal', fontsize=14)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(frec[0:Nmitad], np.abs(tf_senyal[0:Nmitad]))
    ax2.set_ylabel('Amplitud de la señal', fontsize=14)
    # Cálculo del espectrograma
    ax3 = fig.add_subplot(1, 3, 3)
    f3, t3, Sxx3 = sig.spectrogram(senyal, fs, window=sig.get_window('hann', tamanyo_ventana), noverlap=solape_ventana)
    ax3.pcolormesh(t3, f3, np.log10(Sxx3))
    # Modifica las etiquetas para el eje X y el título en cada gráfica:
    ax1.set_xlabel('Tiempo [s]', fontsize=14)
    ax2.set_xlabel('Frecuencia [Hz]', fontsize=14)
    ax3.set_xlabel('Tiempo [s]', fontsize=14)
    ax3.set_ylabel('Frecuencia [Hz]', fontsize=14)
    ax1.set_title('Señal temporal', fontsize=14)
    ax2.set_title('Espectro', fontsize=14)
    ax3.set_title('Espectrograma', fontsize=14)
    ax2.set_xlim([0, fmax])
    ax3.set_ylim([0, fmax])
    return ax1, ax2, ax3

def grafica_wavelet(senyal,fs,smin=3,smax=100,Nescalas=100,sizeX=8,sizeY=8,
                          tipo = 'morl',interactiva=False):
  if interactiva:
    #%matplotlib widget
    get_ipython().run_line_magic('matplotlib', 'widget')
  else:
    #%matplotlib inline
    get_ipython().run_line_magic('matplotlib', 'inline')
  fig, ax = plt.subplots(1, 1,figsize=(sizeX, sizeY))
  fig.canvas.header_visible = False # Hide the Figure name at the top
  fig.canvas.footer_visible = True # Show/Hide the footer
  fig.canvas.toolbar_position = 'right' # Change the toolbar position
  # Transformada de wavelet
  N = len(senyal)
  t = np.linspace(0, N/fs, N)
  escalas = np.linspace(smin,smax,100)
  aux_cwt_trans = pywt.cwt(senyal, escalas, tipo, 1/fs)
  cwt_trans = aux_cwt_trans[0]
  frecs = aux_cwt_trans[1]
  X, Y = np.meshgrid(t, frecs)
  im = ax.pcolormesh(X, Y, np.abs(cwt_trans))
  ax.set_xlabel('tiempo [s]')
  ax.set_ylabel('frecuencia [Hz]')
  ax.set_ylim([0, frecs.max()])
  fig.tight_layout()
  print(f'Frecuencias mínima ({frecs.min():.2f} Hz) y máxima ({frecs.max():.2f} Hz)')
  return ax

def grafica_autocorrelacion(senyal, fs, sizeX = 10, sizeY = 4, interactiva = False):
  if interactiva:
    #%matplotlib widget
    get_ipython().run_line_magic('matplotlib', 'widget')
  else:
    #%matplotlib inline
    get_ipython().run_line_magic('matplotlib', 'inline')
  # La señal está guardada en el vector senyal
  # La frecuencia de muestreo en fs
  Npuntos = len(senyal)
  tmax = Npuntos /fs # tmax = Npuntos x t_muestreo = Npuntos / f_muestreo
  tiempo = np.linspace(0, tmax, Npuntos) # creamos el vector de tiempos
  # Cálculo de la función de autocorrelación
  autocorr = sig.correlate(senyal, senyal, 'full')
  # Gráficas:
  fig = plt.figure(figsize=(sizeX, sizeY))
  fig.canvas.header_visible = False # Hide the Figure name at the top
  fig.canvas.footer_visible = True # Show/Hide the footer
  fig.canvas.toolbar_position = 'right' # Change the toolbar position
  ax1 = fig.add_subplot(1, 2, 1)
  ax1.plot(tiempo,senyal)
  ax1.set_xlabel('tiempo [s]'), ax1.set_title('Señal temporal')
  ax2 = fig.add_subplot(1, 2, 2)
  ax2.plot(tiempo,autocorr[Npuntos-1:])
  ax2.set_xlabel(r'$\tau$ [s]'), ax2.set_title('Autocorrelación de la señal');
  fig.tight_layout()
  return ax2

def grafica_envolvente(senyal, fs, sizeX = 10, sizeY = 2, interactiva = False):
  Npuntos = len(senyal)
  tmax = Npuntos /fs # tmax = Npuntos x t_muestreo = Npuntos / f_muestreo
  tiempo = np.linspace(0, tmax, Npuntos) # creamos el vector de tiempos
  if interactiva:
    #%matplotlib widget
    get_ipython().run_line_magic('matplotlib', 'widget')
  else:
    #%matplotlib inline
    get_ipython().run_line_magic('matplotlib', 'inline')
  # Cálculo de la transformada de Hilbert (son números complejos)
  senyal_hilb = sig.hilbert(senyal)
  # Cálculo de la envolvente (módulo de la transformada de Hilbert)
  envolvente = np.abs(senyal_hilb)
  fig = plt.figure(figsize=(sizeX, sizeY))
  fig.canvas.header_visible = False # Hide the Figure name at the top
  fig.canvas.footer_visible = True # Show/Hide the footer
  fig.canvas.toolbar_position = 'right' # Change the toolbar position
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(tiempo,senyal,label='señal')
  ax.plot(tiempo,envolvente,label='envolvente')
  ax.set_xlabel('tiempo [s]')
  fig.tight_layout()
  return ax

def grafica_espectro_octavas(y, fs, sizeX = 10, sizeY = 2):
  spl, freq = octavefilter(y, fs=fs, fraction=1, order=16, limits=[12, 20000], show=0)
  #%matplotlib inline
  get_ipython().run_line_magic('matplotlib', 'inline')
  fig = plt.figure(figsize=(sizeX, sizeY))
  fig.canvas.header_visible = False # Hide the Figure name at the top
  fig.canvas.footer_visible = True # Show/Hide the footer
  fig.canvas.toolbar_position = 'right' # Change the toolbar position
  ax = fig.add_subplot(1, 1, 1)
  ax.bar(freq[:-1],spl[:-1], width=np.diff(freq)*0.6, ec='white')
  ax.set_xscale("log")
  ax.set_xlabel('frecuencia [Hz]')
  fig.tight_layout()
  return ax

def grafica_espectro_tercios(y, fs, sizeX = 10, sizeY = 2):
  spl, freq = octavefilter(y, fs=fs, fraction=3, order=16, limits=[12, 20000], show=0)
  #%matplotlib inline
  get_ipython().run_line_magic('matplotlib', 'inline')
  fig = plt.figure(figsize=(sizeX, sizeY))
  fig.canvas.header_visible = False # Hide the Figure name at the top
  fig.canvas.footer_visible = True # Show/Hide the footer
  fig.canvas.toolbar_position = 'right' # Change the toolbar position
  ax = fig.add_subplot(1, 1, 1)
  ax.bar(freq[:-1],spl[:-1], width=np.diff(freq)*0.8, ec='white')
  ax.set_xscale("log")
  ax.set_xlabel('frecuencia [Hz]')
  fig.tight_layout()
  return ax

def aplica_ponderacion_A(y, fs):
  yp = waveform_analysis.A_weight(y, fs)
  return yp

def aplica_filtro(senyal, fs, tipo, fcorte, orden=2):
  if tipo == 'pasabaja':
    # Creación del filtro (pasa-baja en este caso):
    b, a = sig.butter(orden, [fcorte[0]/(fs/2)], 'lowpass')
  elif tipo == 'pasaalta':
    b, a = sig.butter(orden, [fcorte[0]/(fs/2)], 'highpass')
  elif tipo == 'pasabanda':
    b, a = sig.butter(orden, [fcorte[0]/(fs/2),fcorte[1]/(fs/2)], 'bandpass')
  else:
    print('ERROR en el tipo de filtro.\
    Parámetros válidos: pasabaja, pasaalta o pasabanda')
  # Y aplicamos el filtro a la señal temporal
  senyalfilt = sig.filtfilt(b, a, senyal)
  return senyalfilt

def recorta(senyal, fs, tini, tfin):
  coordini = int(fs * tini)
  coordfin = int(fs * tfin)
  senyalrec = senyal[coordini:coordfin]
  return senyalrec

def grafica_cepstro(listasen, listafs, sizeX = 10, sizeY = 4, interactiva = False):
  if type(listafs) is not list:
    listafs = [listafs] * len(listasen)
  if interactiva:
    #% settings for bokeh
    output_notebook()
    # create a new plot with a title and axis labels
    ax = figure(max_width=sizeX*100, height=sizeY*100)
    for senyal, fs in zip(listasen, listafs):
      Npuntos = len(senyal)
      tmax = Npuntos /fs # tmax = Npuntos x t_muestreo = Npuntos / f_muestreo
      tiempo = np.linspace(0, tmax, Npuntos) # creamos el vector de tiempos
      spectrum = np.fft.fft(senyal)
      ceps = np.fft.ifft(np.log(np.abs(spectrum))).real
      ax.line(tiempo*1000,np.abs(ceps),alpha=0.7)
    ax.xaxis.axis_label = 'cuefrencia [ms]'
    show(ax)
  else:
    #%matplotlib inline
    get_ipython().run_line_magic('matplotlib', 'inline')
    fig = plt.figure(figsize=(sizeX, sizeY))
    fig.canvas.header_visible = False # Hide the Figure name at the top
    fig.canvas.footer_visible = True # Show/Hide the footer
    fig.canvas.toolbar_position = 'right' # Change the toolbar position
    ax = fig.add_subplot(1, 1, 1)
    for senyal, fs in zip(listasen, listafs):
      Npuntos = len(senyal)
      tmax = Npuntos /fs # tmax = Npuntos x t_muestreo = Npuntos / f_muestreo
      tiempo = np.linspace(0, tmax, Npuntos) # creamos el vector de tiempos
      spectrum = np.fft.fft(senyal)
      ceps = np.fft.ifft(np.log(np.abs(spectrum))).real
      ax.plot(tiempo*1000,np.abs(ceps),alpha=0.7)
    ax.set_xlabel('cuefrencia [ms]')
    fig.tight_layout()
  return ax

# Función para calcular un cepstrograma
def cepstrogram(x, win, hop, fs):
  # function: C, q, t = cepstrogram(x, win, hop, fs)
  # Input:
  # x - signal in the time domain
  # win - analysis window function
  # hop - hop size
  # fs - sampling frequency, Hz
  # Output:
  # C - real cepstrum-matrix (only unique points,
  #     time across columns, quefrency across rows)
  # q - quefrency vector, s
  # t - time vector, s
  # representation of the signal as column-vector
  x = x[:]
  # determination of the signal length
  xlen = len(x)
  # determination of the window length
  wlen = len(win)
  # stft matrix size estimation and preallocation
  NUP = int(np.ceil((1+wlen)/2))      # calculate the number of unique fft points
  L = 1+int((xlen-wlen)/hop)          # calculate the number of signal frames
  C = np.zeros([NUP-1, L-1])          # preallocate the stft matrix
  # STFT (via time-localized FFT)
  for l in range(L-1):
    # windowing
    xw = x[l*hop : wlen+l*hop] * win
    # cepstrum calculation
    c = np.real(np.fft.ifft(np.log(np.abs(np.fft.fft(xw)))))
    # update of the cepstrum-matrix
    C[:, l] = c[0:NUP-1]
  # calculation of the time and quefrency vectors
  t = np.arange(wlen/2,wlen/2+(L-1)*hop,hop)/fs
  q = np.arange(0,NUP-1)/fs
  return C, q, t

# Funciones relacionadas con el cepstro
# Estas funciones nos servirán para calcular el cepstro y su inversa
# Fuente: http://python-acoustics.github.io/python-acoustics/_modules/acoustics/cepstrum.html

def complex_cepstrum(x, n=None):
  r"""Compute the complex cepstrum of a real sequence.
  """
  def _unwrap(phase):
    samples = phase.shape[-1]
    unwrapped = np.unwrap(phase)
    center = (samples + 1) // 2
    if samples == 1:
      center = 0
    ndelay = np.array(np.round(unwrapped[..., center] / np.pi))
    unwrapped -= np.pi * ndelay[..., None] * np.arange(samples) / center
    return unwrapped, ndelay
  spectrum = np.fft.fft(x, n=n)
  unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
  log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped_phase
  ceps = np.fft.ifft(log_spectrum).real
  return ceps, ndelay

def inverse_complex_cepstrum(ceps, ndelay):
  r"""Compute the inverse complex cepstrum of a real sequence.
  """
  def _wrap(phase, ndelay):
    ndelay = np.array(ndelay)
    samples = phase.shape[-1]
    center = (samples + 1) // 2
    wrapped = phase + np.pi * ndelay[..., None] * np.arange(samples) / center
    return wrapped
  log_spectrum = np.fft.fft(ceps)
  spectrum = np.exp(log_spectrum.real + 1j * _wrap(log_spectrum.imag, ndelay))
  x = np.fft.ifft(spectrum).real
  return x

def real_cepstrum(x, n=None):
  r"""Compute the real cepstrum of a real sequence.
  """
  spectrum = np.fft.fft(x, n=n)
  ceps = np.fft.ifft(np.log(np.abs(spectrum))).real
  return ceps

#def descarga_audio_youtube(link):
#  with yt_dlp.YoutubeDL({'extract_audio': True, 'format': 'bestaudio', 'outtmpl': '%(title)s.mp3'}) as video:
#    info_dict = video.extract_info(link, download = True)
#    video_title = info_dict['title']
#    print(video_title)
#    video.download(link)
#    print("Descarga completada - mira la carpeta local de Google Colab")
#@title Cómo graficar un vector muestreado a una frecuencia fs
# OJO: solo son necesarios: senyal y fs (leídos anteriormente; pueden tener otro nombre)
# OJO: los argumentos sizeX y sizeY son opcionales; cambian el tamaño de la figura
# OJO: el argumento interactiva es opcional; si lo pones a True podrás hacer zoom
ax1 = grafica_tiempo([senyal], fs, sizeX = 6, sizeY = 2, interactiva = False)
ax1.set_xlim([0, 12])

#@title Cómo graficar dos o más señales en la misma gráfica con la misma fs
urlbase = 'https://hcliment.webs.upv.es/docencia/dar'
url = urlbase + '/examen/2024/bomba-oct.wav'
fs, senyal01 = lee_audio_servidor(url)
url = urlbase + '/examen/2024/bomba-nov.wav'
fs, senyal02 = lee_audio_servidor(url)
ax1 = grafica_tiempo([senyal01, senyal02], fs, sizeX = 6, sizeY = 2)
ax1.legend(['Señal 1', 'Señal 2'])

#@title Cómo graficar dos o más señales en la misma gráfica con distinta fs
urlbase = 'https://hcliment.webs.upv.es/docencia/dar'
url = urlbase + '/examen/2024/navemic01.wav'
fs01, senyal01 = lee_audio_servidor(url)
url = urlbase + '/examen/2024/navemic02.wav'
fs02, senyal02 = lee_audio_servidor(url)
ax1 = grafica_tiempo([senyal01, senyal02], [fs01, fs02], sizeX = 6, sizeY = 2)
ax1.legend(['Señal 1', 'Señal 2'])

#@title Cómo recortar una señal
urlbase = 'https://hcliment.webs.upv.es/docencia/dar/examen/2024/'
fs, senyal = lee_audio_servidor(urlbase + 'navemic02.wav')
senyal_rec = recorta(senyal, fs, 5, 8) # se indican los tiempos de inicio y fin
ax1 = grafica_tiempo([senyal, senyal_rec], fs, sizeX = 6, sizeY = 2, interactiva = False)
ax1.legend(['Señal original', 'Señal recortada'])

#@title Cómo calcular el nivel de presión sonora de una señal
fs, tmax = 48000, 5
Kcalibracion = 1 # constante para pasar la señal a Pascales
t = np.linspace(0, tmax, int(fs*tmax)) # vector de tiempos
# Señal senoidal
A1, f1 = 2/np.sqrt(2), 100
senyal= Kcalibracion * A1 * np.sin(2*np.pi*f1*t)
RMS = np.sqrt(np.mean(senyal**2))
SPL = calcula_db(senyal)
print(f'El valor de la amplitud es: {Kcalibracion * A1:.4f} Pa')
print(f'El valor eficaz es: {RMS:.2f} Pa')
print(f'El nivel de presión sonora es: {SPL:.2f} dB')

#@title Cómo graficar el espectro de una señal
urlbase = 'https://hcliment.webs.upv.es/docencia/dar/sounds/tema06/'
fs, senyal = lee_audio_servidor(urlbase + 'rodamientoaroexteriorBPFO.wav')
ax1 = grafica_espectro([senyal],fs,sizeX=6,sizeY=3,interactiva=False)
ax1.set_xlim([0,4000])

#@title Cómo graficar el espectro en octavas de una señal
fs, tmax = 48000, 5
t = np.linspace(0, tmax, int(fs*tmax)) # vector de tiempos
# Señal con 6 frecuencias
f1, f2, f3, f4, f5, f6 = 20, 100, 500, 2000, 4000, 10000
senyal=100*(np.sin(2*np.pi*f1*t)+np.sin(2*np.pi*f2*t)+np.sin(2*np.pi*f3*t) \
            +np.sin(2*np.pi*f4*t)+np.sin(2*np.pi*f5*t)+np.sin(2*np.pi*f6*t))
ax1 = grafica_espectro_octavas(senyal,fs,sizeX=6,sizeY=3)

#@title Cómo graficar el espectro en tercios de octava de una señal
fs, tmax = 48000, 5
t = np.linspace(0, tmax, int(fs*tmax)) # vector de tiempos
# Señal con 6 frecuencias
f1, f2, f3, f4, f5, f6 = 20, 100, 500, 2000, 4000, 10000
senyal=100*(np.sin(2*np.pi*f1*t)+np.sin(2*np.pi*f2*t)+np.sin(2*np.pi*f3*t) \
            +np.sin(2*np.pi*f4*t)+np.sin(2*np.pi*f5*t)+np.sin(2*np.pi*f6*t))
ax1 = grafica_espectro_tercios(senyal,fs,sizeX=6,sizeY=3)

#@title Cómo graficar la función de autocorrelación de una señal
Npuntos = 1000
tmax = 0.08
tiempo = np.linspace(0, tmax, Npuntos) # creamos el vector de tiempos
senyal = np.sin(2*np.pi*50*tiempo) \
         + 0.4 * np.sin(2*np.pi*2*50*tiempo +1.5*np.pi) \
         + 0.4 * np.sin(2*np.pi*3*50*tiempo+1.5*np.pi)
fs = Npuntos / tmax
ax2 = grafica_autocorrelacion(senyal, fs, sizeX=6, sizeY=3, interactiva=False)

#@title Cómo graficar la envolvente de una señal con la Transformada de Hilbert
urlbase = 'https://hcliment.webs.upv.es/docencia/dar/sounds/tema06/'
fs, senyal = lee_audio_servidor(urlbase + 'rodamientoaroexteriorBPFO.wav')
senyalfilt = aplica_filtro(senyal, fs, 'pasabaja', [100], orden=2)
senyalfiltrec = recorta(senyalfilt, fs, 2, 4)
env01 = calcula_envolvente(senyalfiltrec)
eje = grafica_tiempo([senyalfiltrec, env01], fs, sizeX=6, sizeY=2,interactiva=False)
eje.legend(["Señal", "Envolvente"])
ax1.set_title('Aplicación de la Transformada de Hilbert')

#@title Cómo graficar la envolvente de una señal usando una ventana móvil
env02 = calcula_envolvente2(senyal, fs, tipo='arriba', windowsize=2000)
eje = grafica_tiempo([senyal, env02], fs, sizeX=6, sizeY=2,interactiva=False)
eje.legend(["Señal", "Envolv. 02"])

#@title Cómo filtrar señales
urlbase = 'https://hcliment.webs.upv.es/docencia/dar/sounds/tema06/'
fs, senyal = lee_audio_servidor(urlbase + 'rodamientoaroexteriorBPFO.wav')
senyalbaja = aplica_filtro(senyal, fs, 'pasabaja', [1000], orden=4)
senyalalta = aplica_filtro(senyal, fs, 'pasaalta', [1000], orden=4)
senyalbanda = aplica_filtro(senyal, fs, 'pasabanda', [1000, 2000], orden=4)
ax1 = grafica_espectro([senyal, senyalbaja, senyalalta, senyalbanda], fs,
                       sizeX = 6, sizeY = 3, interactiva=False)
ax1.set_xlim([0,4000])
ax1.legend(['Original', 'Pasa baja', 'Pasa alta', 'Pasa banda'])

#@title Cómo graficar un espectrograma
urlbase = 'https://hcliment.webs.upv.es/docencia/dar/sounds/tema06/'
fs,senyal=lee_audio_servidor(urlbase+'wav_i3_bmw_test4_better_mic_close_to_motor.wav')
tamanyo_ventana1 = 1024
solape1 = tamanyo_ventana1/2
eje=grafica_espectrograma([senyal],fs,[tamanyo_ventana1],[solape1],sizeX=10, sizeY=4,
                          log=True,interactiva=False,vlim=[-8,-14])

#@title Cómo graficar varios espectrogramas
urlbase = 'https://hcliment.webs.upv.es/docencia/dar'
url = urlbase + '/sounds/tema04/ferrari355-freesound43484.wav'
fs, senyal = lee_audio_servidor(url)
tamanyo_ventana1, tamanyo_ventana2 = fs, 4096
solape1, solape2 = fs/2, 3000
eje = grafica_espectrograma([senyal, senyal], fs, [tamanyo_ventana1, tamanyo_ventana2],
                            [solape1, solape2], sizeX = 10, sizeY = 3,
                            log = True, interactiva = False)
eje[0,0].set_ylim([0,1000])
eje[0,1].set_ylim([0,1000])

#@title Cómo graficar una transformada de Wavelet
tmax, N = 2, 10000
fs = N / tmax
t = np.linspace(0, tmax, N)
senyal = 6 * np.cos(2 * np.pi * 50 * t) + 3 * sig.gausspulse(t-1.525, fc=400)
ejew = grafica_wavelet(y_trans,fs,smin=3,smax=250,Nescalas=1000,
                       sizeX=8,sizeY=4,tipo='morl',interactiva=False)

#@title Gráfica triple: tiempo, frecuencia y espectrograma
urlbase = 'https://hcliment.webs.upv.es/docencia/dar/sounds/tema06'
fs, senyal = lee_audio_servidor(urlbase + '/hunday_ionic_test6_93km_mic_close_to_motor.wav')
ejet, ejef, ejee = grafica_tiempo_espectro_espectrograma(senyal, fs, tdeseado=12,
  fmax=20000, tamanyo_ventana=2048, solape_ventana=1024, sizeX=18, sizeY=3)
Audio(senyal, rate=fs)

#@title Cálculo del cepstro de una señal
url_base = 'https://hcliment.webs.upv.es/docencia/dar/sounds/tema06/'
fs_m, y_m = lee_audio_servidor(url_base + 'FAG-6204-C-2Z-mal01.wav')
ejec = grafica_cepstro([y_m], [fs_m], sizeX = 7, sizeY = 2, interactiva = False)
ejec.set_xlim([0,120])
ejec.set_ylim([0,0.1])

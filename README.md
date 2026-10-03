# Manual del Detector de Caras (Multi-Cam Face Tracker)

## Índice

1. [Qué hace el detector](#1-qué-hace-el-detector)
2. [Conceptos previos](#2-conceptos-previos)
3. [Mapa de archivos](#3-mapa-de-archivos)
4. [Estructuras de datos](#4-estructuras-de-datos)
5. [La clase FaceDetector](#5-la-clase-facedetector)
6. [Carga del modelo](#6-carga-del-modelo)
7. [Detección de caras](#7-detección-de-caras)
8. [Reconocimiento de caras](#8-reconocimiento-de-caras)
9. [Gestión de caras conocidas](#9-gestión-de-caras-conocidas)
10. [Edad y género](#10-edad-y-género)
11. [Dibujo sobre el video](#11-dibujo-sobre-el-video)
12. [Cómo se integra con el resto de la app](#12-cómo-se-integra-con-el-resto-de-la-app)
13. [Flujo completo de un frame](#13-flujo-completo-de-un-frame)
14. [Parámetros de configuración](#14-parámetros-de-configuración)
15. [Cómo leer tu log de arranque](#15-cómo-leer-tu-log-de-arranque)
16. [Limitaciones conocidas](#16-limitaciones-conocidas)
17. [Ideas para modificar y mejorar](#17-ideas-para-modificar-y-mejorar)
18. [Solución de problemas](#18-solución-de-problemas)
19. [Glosario](#19-glosario)

---

## 1. Qué hace el detector

El detector (`FaceDetector`) es el componente de inteligencia artificial de la
aplicación. Recibe una imagen (un frame de una cámara) y hace tres cosas:

1. **Detecta** dónde hay caras.
2. **Describe** cada cara con un vector numérico único (embedding) y, si está activado,
   estima su **edad y género**.
3. **Reconoce** a la persona comparando ese vector con los de las personas registradas.

---

## 2. Conceptos previos

### 2.1 InsightFace
Librería de código abierto para análisis facial. El proyecto la usa mediante su clase
`FaceAnalysis`, que agrupa varios modelos de IA.

### 2.2 Modelo `buffalo_l`
Paquete de modelos de InsightFace que se descarga automáticamente la primera vez
(unos 280 MB) en `./models/models/buffalo_l/`. Contiene, según el log de arranque:

| Archivo | Función | ¿Lo usa el proyecto? |
|---|---|---|
| `det_10g.onnx` | Detección de caras | Sí |
| `w600k_r50.onnx` | Reconocimiento (genera el embedding) | Sí |
| `genderage.onnx` | Estimación de edad y género | Sí |
| `1k3d68.onnx` | Puntos faciales 3D (68) | No (aparece como "model ignore") |
| `2d106det.onnx` | Puntos faciales 2D (106) | No (aparece como "model ignore") |

### 2.3 ONNX y ONNX Runtime
ONNX es un formato estándar de modelos de IA. `onnxruntime` es el motor que los
ejecuta. Puede usar la CPU (`CPUExecutionProvider`) o una GPU NVIDIA
(`CUDAExecutionProvider`).

### 2.4 Embedding
Un vector de **512 números** que representa una cara. Dos fotos de la misma persona
producen vectores muy parecidos; fotos de personas distintas producen vectores
diferentes. Todo el reconocimiento se reduce a comparar vectores.

### 2.5 Similitud coseno
Medida de qué tan parecidos son dos vectores. Va de -1 a 1 (en la práctica, para caras,
entre 0 y 1). Cuanto más cerca de 1, más parecidas. El proyecto la usa para decidir si
una cara detectada coincide con una registrada.

### 2.6 Umbral (threshold)
Valor mínimo que debe alcanzarse para aceptar un resultado. Hay dos:
- **Umbral de detección:** qué tan segura debe estar la IA de que hay una cara.
- **Umbral de reconocimiento:** qué tan parecida debe ser a una cara registrada.

---

## 3. Mapa de archivos

```
multi-cam-face-tracker/
├── main.py                    # Punto de entrada
├── config/
│   ├── config.yaml            # Ajustes generales y de reconocimiento
│   └── camera_config.yaml     # Definición de cámaras
├── core/
│   ├── face_detection.py      # ★ EL DETECTOR (este manual)
│   ├── utils.py               # Dibujo de cajas y etiquetas
│   ├── camera_manager.py      # Captura de video (un hilo por cámara)
│   ├── alert_system.py        # Sonido, screenshots, historial
│   ├── telegram_manager.py    # Envío a Telegram
│   └── database.py            # SQLite
├── ui/
│   └── main_window.py         # Ventana principal (usa el detector)
├── data/
│   ├── known_faces/           # Fotos de personas registradas
│   ├── screenshots/           # Capturas de las alertas
│   └── database.db            # Base de datos
├── models/                    # Modelos descargados de InsightFace
├── assets/                    # Sonido de alerta y logo
└── logs/                      # Registros de la aplicación
```

---

## 4. Estructuras de datos

El detector define dos clases de datos (`dataclass`) al inicio del archivo.

### 4.1 `Face` (L14-22)

Representa **una cara detectada** en un frame.

| Campo | Contenido |
|---|---|
| Caja (`bbox`) | Coordenadas `x1, y1, x2, y2` del rectángulo de la cara |
| Puntos clave (`kps`) | 5 puntos (ojos, nariz, comisuras de la boca) usados para alinear la cara |
| Confianza de detección | Qué tan segura está la IA de que es una cara |
| Embedding | Vector de 512 números que identifica la cara |
| Edad | Entero, opcional |
| Género | `'Male'` o `'Female'`, opcional |
| Imagen de la cara | Recorte del frame con esa cara |


### 4.2 `KnownFace` (L24-28)

Representa **una persona registrada**.

| Campo | Contenido |
|---|---|
| Nombre | Nombre de la persona (sale del nombre del archivo de la foto) |
| Embedding | Vector de 512 números calculado a partir de su foto |
| Ruta | Dónde está el archivo de la foto |

---

## 5. La clase FaceDetector

Definida aproximadamente entre L30 y L222. Es la interfaz que usa el resto de la app.

### Métodos principales

| Método | Líneas aprox. | Qué hace |
|---|---|---|
| `__init__` / `_load_model` | L31-58 | Lee la configuración y carga el modelo |
| `load_known_faces()` | L60-102 | Carga las caras registradas desde `data/known_faces/` |
| `detect_faces()` | L104-125 | Detecta caras en un frame |
| `recognize_faces()` | L127-160 | Compara caras detectadas con las conocidas |
| `add_known_face()` | L175-206 | Registra una cara nueva en tiempo de ejecución |
| `_get_age()`, `_get_gender()` | L209-222 | Extraen edad y género de la predicción |

### Atributos de configuración que guarda

- `recognition_threshold`: similitud mínima para reconocer.
- `detection_threshold`: confianza mínima para aceptar una detección.
- `max_batch_size`: máximo de caras procesadas por lote.
- `device`: `"cpu"` o `"cuda"`.
- `analysis_enabled`: si se estima edad y género.

El deslizador de la pestaña **Controls** modifica `recognition_threshold` en vivo.

---

## 6. Carga del modelo

Ocurre una sola vez al iniciar (L41-58).

### Pasos

1. Lee los parámetros de `config/config.yaml` (sección `recognition`).
2. Crea un objeto `FaceAnalysis` de InsightFace con el paquete `buffalo_l`,
   habilitando los módulos de **detección**, **reconocimiento** y **género/edad**.
3. Elige el proveedor de ejecución:
   - `device: "cuda"` intenta usar la GPU NVIDIA.
   - `device: "cpu"` usa la CPU.
4. Prepara el modelo con un **tamaño de detección de 640x640** píxeles.
5. Registra en el log: *"Face detection model loaded successfully"*.

### Sobre la advertencia de CUDA

En equipos sin GPU aparece:

```
UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```

---

## 7. Detección de caras

Método `detect_faces()` (L104-125).

### Entrada
Un frame de la cámara (imagen OpenCV, formato BGR).

### Proceso (conceptual)

```
frame ──► modelo de detección (det_10g) ──► lista de caras candidatas
                                              │
                          descarta las que estén por debajo de detection_threshold
                                              │
                    por cada cara: calcula embedding (w600k_r50)
                                   estima edad y género (genderage)
                                   recorta la imagen de la cara
                                              │
                                              ▼
                                       lista de objetos Face
```

### Salida
Una lista de objetos `Face` (vacía si no hay caras).

### Detalles importantes

- El tamaño de detección de 640x640 es un compromiso: más grande detecta caras más
  pequeñas/lejanas pero es más lento; más pequeño es más rápido pero pierde caras
  distantes.
- La alineación usa los 5 puntos clave. En tu log aparece un `FutureWarning` de
  `tform.estimate` en `insightface/utils/face_align.py`: es un aviso de
  `scikit-image` sobre una función que cambiará en el futuro. No afecta el
  funcionamiento actual.

---

## 8. Reconocimiento de caras

Método `recognize_faces()` (L127-160).

### Idea

Para cada cara detectada, se busca cuál de las caras registradas se le parece más.

### Proceso (conceptual)

```
Para cada Face detectada:
    1. Toma su embedding (512 números).
    2. Calcula la similitud coseno contra los embeddings de TODAS las KnownFace.
    3. Encuentra la más parecida.
    4. Si similitud >= recognition_threshold  → reconocida (nombre = el de esa KnownFace)
       Si no                                   → desconocida
```

### Eficiencia

La comparación está **vectorizada con numpy**: en vez de un bucle cara por cara, se
multiplican matrices y se obtienen todas las similitudes de una vez. Esto permite
comparar contra muchas personas registradas sin mucho costo.

### Ejemplo ilustrativo

```python
import numpy as np

def similitud_coseno(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# embedding_detectado: vector de 512 números de la cara en cámara
# conocidas: lista de (nombre, embedding)
mejor_nombre, mejor_sim = None, 0
for nombre, emb in conocidas:
    sim = similitud_coseno(embedding_detectado, emb)
    if sim > mejor_sim:
        mejor_nombre, mejor_sim = nombre, sim

if mejor_sim >= 0.6:   # recognition_threshold
    print(f"Reconocido: {mejor_nombre} ({mejor_sim:.2f})")
else:
    print("Desconocido")
```

### Efecto del umbral (`recognition_threshold`, por defecto 0.6)

| Umbral | Efecto |
|---|---|
| Más bajo (ej. 0.4) | Reconoce más fácil, pero aumentan los **falsos positivos** (confunde personas) |
| Más alto (ej. 0.75) | Más estricto, pero puede **no reconocer** a alguien registrado |

Ajusta desde el deslizador de la pestaña *Controls* y prueba con personas reales.

---

## 9. Gestión de caras conocidas

### 9.1 `load_known_faces()` (L60-102)

Se ejecuta al iniciar y cada vez que se cierra el Face Manager o el Alert Panel.

1. Recorre la carpeta `data/known_faces/`.
2. Por cada imagen, detecta caras y **toma la primera** que encuentre.
3. Calcula su embedding.
4. Usa el **nombre del archivo** como identidad (`juan.jpg` → "juan").
5. Guarda todo como objetos `KnownFace`.

### 9.2 `add_known_face()` (L175-206)

Permite añadir una persona sin reiniciar la app. Es lo que usa el **Face Manager**
(*Tools > Face Manager > Add Face*).

### 9.3 Buenas prácticas para registrar caras

- Una foto **de frente**, nítida y bien iluminada.
- Una sola persona en la imagen (solo se usa la primera cara detectada).
- Sin gafas oscuras, gorra que tape los ojos ni mascarilla.
- Nombre de archivo claro, sin caracteres raros.
- Si el reconocimiento falla, agrega una segunda foto con otro nombre
  (por ejemplo `juan_2.jpg`), aunque el sistema las tratará como identidades separadas.

---

## 10. Edad y género

Solo si `analysis_enabled: true` en la configuración.

| Método | Salida | Cómo se obtiene |
|---|---|---|
| `_get_age()` | Entero (años) | Predicción del modelo `genderage` |
| `_get_gender()` | `'Male'` o `'Female'` | Argmax de la predicción binaria del modelo |

**Advertencia:** son estimaciones. La edad puede errar por varios años y el género se
predice solo de forma binaria. No las uses para decisiones importantes.

---

## 11. Dibujo sobre el video

`core/utils.py` (L8-62) contiene funciones auxiliares. La principal para el usuario es
`draw_face_info()`, que dibuja sobre el frame:

- El rectángulo alrededor de la cara.
- El nombre (o "Unknown").
- La edad y el género, si están activos.
- Un color según el estado (reconocida o no).

El resultado es lo que ves en la pestaña **Monitor**.

---

## 12. Cómo se integra con el resto de la app

### 12.1 Con `camera_manager.py`
Cada cámara tiene su propio hilo que captura frames y los deja en una cola de un solo
espacio (`maxsize=1`). Si llega un frame nuevo y la cola está llena, se descarta el
viejo. Así el detector siempre recibe la imagen más reciente, sin retraso acumulado.

### 12.2 Con `main_window.py`
Un temporizador de Qt se dispara cada **30 ms**. En cada disparo se pide el último frame
de cada cámara, pero **no se procesan todos**: solo uno de cada N frames, según el
*intervalo de procesamiento* de la pestaña Controls. El método `process_frame()` es
quien llama al detector.

### 12.3 Con `alert_system.py`
Cuando `recognize_faces()` devuelve una persona **conocida**, se llama a
`trigger_alert()`, que:
- reproduce el sonido (`assets/alert.wav`) con `pygame`;
- guarda un screenshot en `data/screenshots/` con el formato
  `YYYYMMDD_HHMMSS_camN_NOMBRE.jpg`;
- envía el mensaje a Telegram si está habilitado (con límite de frecuencia);
- añade el evento al historial en memoria.

### 12.4 Con `database.py`
Cada detección se registra en SQLite (`data/database.db`) con: hora, cámara, nombre
(o "Unknown"), edad, género, confianza y ruta del screenshot. La pestaña **History**
lee estos datos.

---

## 13. Flujo completo de un frame

```
[Cámara]
   │  cv2.VideoCapture (hilo propio)
   ▼
[Cola de 1 frame]  ← se descarta el frame viejo si llega uno nuevo
   │
   ▼  (temporizador cada 30 ms; procesa 1 de cada N frames)
[MainWindow.process_frame]
   │
   ├─► FaceDetector.detect_faces(frame)
   │        └─ lista de Face (caja, embedding, edad, género)
   │
   ├─► FaceDetector.recognize_faces(caras)
   │        └─ similitud coseno vs. KnownFace
   │
   ├─► Si hay persona conocida:
   │        ├─ AlertSystem.trigger_alert()  → sonido + screenshot + Telegram
   │        └─ FaceDatabase.log_face_event() → SQLite
   │
   └─► utils.draw_face_info() → dibuja cajas y etiquetas
            │
            ▼
      [Pestaña Monitor]
```

---

## 14. Parámetros de configuración

### `config/config.yaml`

```yaml
app:
  threshold: 0.6
  screenshot_dir: "data/screenshots"
  known_faces_dir: "data/known_faces"
  database_path: "data/database.db"
  alert_sound: "assets/alert.wav"
  logo: "assets/logo.png"
  log_dir: "logs"

recognition:
  detection_threshold: 0.5
  recognition_threshold: 0.6
  max_batch_size: 8
  device: "cpu"            
  analysis_enabled: true
  age_estimation: true
  gender_detection: true

telegram:
  enabled: false
  bot_token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"
  rate_limit: 30          
```

### Guía rápida de ajuste

| Quiero... | Cambio |
|---|---|
| Que detecte menos "falsas caras" | Subir `detection_threshold` |
| Que detecte caras poco claras | Bajar `detection_threshold` |
| Que confunda menos a las personas | Subir `recognition_threshold` |
| Que reconozca con más facilidad | Bajar `recognition_threshold` |
| Más velocidad en CPU | Subir el intervalo de procesamiento, bajar resolución y FPS de la cámara |
| Usar la GPU | `device: "cuda"` (requiere PyTorch/ONNX con CUDA) |
| Quitar edad y género | `analysis_enabled: false` |

### `config/camera_config.yaml`

```yaml
cameras:
  - id: 0
    name: "Front Camera"
    source: 0              
    enabled: true
    resolution:
      width: 1280
      height: 720
    fps: 30
    rotate: 0        
```

---

## 15. Cómo leer tu log de arranque

Ejemplo real de una primera ejecución, línea por línea:

| Línea del log | Significado |
|---|---|
| `Downloading ... buffalo_l.zip` | Primera ejecución: descarga los modelos de IA |
| `Specified provider 'CUDAExecutionProvider' is not in available provider names` | No hay GPU; usará CPU (normal) |
| `Applied providers: ['CPUExecutionProvider']` | Confirmación de que corre en CPU |
| `model ignore: ... 1k3d68.onnx` / `2d106det.onnx` | Modelos de puntos faciales que no se usan |
| `find model: ... det_10g.onnx detection` | Cargó el modelo de detección |
| `find model: ... genderage.onnx genderage` | Cargó el de edad y género |
| `find model: ... w600k_r50.onnx recognition` | Cargó el de reconocimiento |
| `set det-size: (640, 640)` | Tamaño de detección configurado |
| `Face detection model loaded successfully` | Detector listo |
| `Loaded 1 camera configurations` | Leyó 1 cámara del YAML |
| `Database initialized successfully` | SQLite lista |
| `Loaded 0 known faces` | No hay personas registradas todavía |
| `Started camera ID 0` / `opened successfully` | La cámara está capturando |
| `capture thread exiting` / `Stopped camera ID 0` | Se detuvo la cámara (desde la interfaz) |

---

## 16. Limitaciones conocidas

- **Solo caras.** No detecta movimiento, personas de cuerpo entero ni objetos.
- **Sin grabación de video.** Solo screenshots al reconocer a alguien.
- **Alertas solo para conocidos.** No avisa por personas desconocidas.
- **Sin seguimiento entre cámaras.** Cada frame se analiza de forma independiente.
- **Rendimiento en CPU.** El detector corre dentro del bucle de la interfaz; con varias
  cámaras puede ralentizarse.
- **Sensible a las condiciones de imagen.** Mala luz, contraluz, ángulos extremos,
  mascarillas, gafas oscuras o caras muy lejanas reducen la precisión.
- **Una sola foto por identidad.** Una foto poco representativa empeora el
  reconocimiento.
- **Edad y género aproximados.**
- **SQLite y una sola máquina.** Adecuado para sistemas pequeños.

---

## 17. Ideas para modificar y mejorar

| Mejora | Idea general |
|---|---|
| **Alerta por desconocidos** | En `process_frame`, si una cara no supera el umbral, disparar una alerta distinta |
| **Varias fotos por persona** | Promediar embeddings de varias fotos de la misma identidad |
| **Detector en hilo aparte** | Mover la IA a un hilo o proceso separado para no bloquear la interfaz |
| **Grabación** | Grabar el stream con ffmpeg en paralelo y guardar el instante del evento |
| **Modo sin ventana** | Separar la lógica de la interfaz para correr como servicio |
| **Interfaz web** | Exponer el estado y los eventos vía una API (la carpeta `web/` puede ser un punto de partida) |
| **Reconexión automática** | Reintentar la apertura de cámaras RTSP caídas |
| **Notificaciones extra** | Añadir correo o WhatsApp además de Telegram |

---

## 18. Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ImportError ... cannot enable executable stack` (onnxruntime) | Versión antigua de onnxruntime incompatible con el sistema | `uv pip install onnxruntime==1.18.1` |
| `Could not load the Qt platform plugin "xcb"` | Falta una librería del sistema o conflicto de OpenCV | Instalar `libxcb-*` y dejar solo `opencv-python-headless` |
| `libxcb-keysyms.so.1` no encontrada | Falta el paquete | `sudo apt install libxcb-keysyms1` |
| No reconoce a nadie | `known_faces` vacía o umbral muy alto | Registrar caras y bajar el umbral |
| Confunde personas | Umbral muy bajo | Subir `recognition_threshold` |
| Va lento | CPU sobrecargada | Subir intervalo de procesamiento, bajar resolución y FPS |
| Cámara en negro o no abre | Índice incorrecto o sin permisos | Probar `source: 1`; verificar `groups` y `/dev/video*` |
| No llegan alertas de Telegram | `enabled: false` o credenciales inválidas | Configurar token y chat id reales |

---

## 19. Glosario

- **Frame:** una imagen individual de un video.
- **Embedding:** vector numérico que representa una cara.
- **Similitud coseno:** medida del parecido entre dos vectores.
- **Umbral (threshold):** valor mínimo para aceptar un resultado.
- **InsightFace:** librería de análisis facial usada por el proyecto.
- **ONNX / ONNX Runtime:** formato de modelos de IA y su motor de ejecución.
- **RTSP:** protocolo para transmitir video desde cámaras IP.
- **Hilo (thread):** flujo de ejecución paralelo dentro de un programa.
- **Cola (queue):** estructura para pasar datos entre hilos.
- **PyQt5:** librería usada para la interfaz gráfica.
- **SQLite:** base de datos ligera en un archivo.
- **Falso positivo:** el sistema reconoce a alguien que no es esa persona.
- **Falso negativo:** el sistema no reconoce a alguien que sí está registrado.

---

## Nota sobre privacidad

El reconocimiento facial trata **datos biométricos**, que en muchos países (por ejemplo
Colombia, Ley 1581) se consideran datos sensibles. Si lo usas fuera del ámbito
personal (empleados, clientes, espacios compartidos), infórmate sobre los avisos
obligatorios, el consentimiento y la protección de los datos almacenados.

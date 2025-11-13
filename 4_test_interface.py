#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INTERFAZ GRÁFICA PARA DETECCIÓN DE EMOCIONES EN AUDIO
Interfaz completa con grabación, reproducción y análisis en tiempo real
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import librosa
import librosa.display
from PIL import Image, ImageTk
import time
from pathlib import Path
import pickle
import json

# Verificar si TensorFlow está disponible
try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow no está disponible. Algunas funciones estarán limitadas.")


class EmotionDetectionGUI:
    """Interfaz gráfica completa para detección de emociones"""

    def __init__(self, root):
        self.root = root
        self.setup_gui()
        self.load_model()

        # Variables de estado
        self.recorded_audio = None
        self.current_file_path = None
        self.prediction_history = []

    def setup_gui(self):
        """Configura la interfaz gráfica"""
        self.root.title("🎭 Detector de Emociones en Audio - Subir Archivos")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')

        # Configurar estilo
        style = ttk.Style()
        style.theme_use('clam')

        # Marco principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configurar peso de las columnas y filas
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Título
        title_label = ttk.Label(
            main_frame,
            text="🎭 Detector de Emociones en Audio",
            font=("Arial", 20, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Marco izquierdo - Controles
        left_frame = ttk.LabelFrame(main_frame, text="🎮 Controles", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Marco derecho - Visualización
        right_frame = ttk.LabelFrame(main_frame, text="📊 Análisis", padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.setup_controls(left_frame)
        self.setup_visualization(right_frame)

        # Barra de estado
        self.status_var = tk.StringVar()
        self.status_var.set("🟢 Listo para usar")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

    def setup_controls(self, parent):
        """Configura los controles de la interfaz"""

        # Sección de archivo
        file_frame = ttk.LabelFrame(parent, text="📁 Archivo de Audio", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            file_frame,
            text="📂 Seleccionar Archivo",
            command=self.select_file,
            style='Accent.TButton'
        ).pack(fill=tk.X, pady=(0, 5))

        self.file_path_var = tk.StringVar()
        self.file_path_var.set("Ningún archivo seleccionado")
        file_path_label = ttk.Label(
            file_frame,
            textvariable=self.file_path_var,
            wraplength=200
        )
        file_path_label.pack()

        # Información del archivo
        self.file_info_var = tk.StringVar()
        self.file_info_var.set("")
        file_info_label = ttk.Label(
            file_frame,
            textvariable=self.file_info_var,
            font=("Arial", 9),
            foreground="gray"
        )
        file_info_label.pack()

        # Controles de reproducción
        play_frame = ttk.LabelFrame(parent, text="▶️ Reproducción", padding="10")
        play_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            play_frame,
            text="▶️ Reproducir",
            command=self.play_audio
        ).pack(fill=tk.X, pady=(0, 5))

        ttk.Button(
            play_frame,
            text="⏹️ Detener",
            command=self.stop_audio
        ).pack(fill=tk.X)

        # Análisis
        analysis_frame = ttk.LabelFrame(parent, text="🔍 Análisis", padding="10")
        analysis_frame.pack(fill=tk.X, pady=(0, 10))

        self.analyze_button = ttk.Button(
            analysis_frame,
            text="🎭 Detectar Emoción",
            command=self.analyze_emotion,
            style='Accent.TButton'
        )
        self.analyze_button.pack(fill=tk.X, pady=(0, 10))

        # Resultado
        result_frame = ttk.LabelFrame(parent, text="🎯 Resultado", padding="10")
        result_frame.pack(fill=tk.X, pady=(0, 10))

        self.emotion_var = tk.StringVar()
        self.emotion_var.set("Sin análisis")
        emotion_label = ttk.Label(
            result_frame,
            textvariable=self.emotion_var,
            font=("Arial", 14, "bold")
        )
        emotion_label.pack()

        self.confidence_var = tk.StringVar()
        self.confidence_var.set("")
        confidence_label = ttk.Label(result_frame, textvariable=self.confidence_var)
        confidence_label.pack()

        # Historial
        history_frame = ttk.LabelFrame(parent, text="📊 Historial", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True)

        # Lista de historial
        self.history_listbox = tk.Listbox(history_frame, height=6)
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_listbox.yview)
        self.history_listbox.configure(yscrollcommand=scrollbar.set)

        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Botón limpiar historial
        ttk.Button(
            history_frame,
            text="🗑️ Limpiar Historial",
            command=self.clear_history
        ).pack(fill=tk.X, pady=(5, 0))

    def setup_visualization(self, parent):
        """Configura el área de visualización"""

        # Crear figura para matplotlib
        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor='white')

        # Canvas para matplotlib
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar para interacción
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()

        # Inicializar con gráfica vacía
        self.init_empty_plot()

    def load_model(self):
        """Carga el modelo entrenado"""
        self.model = None
        self.label_encoder = None

        try:
            # Buscar modelos entrenados
            model_paths = [
                'models/balanced_emotion_model.h5',
                'models/emotion_model_balanced_final.h5',
                'models/emotion_model.h5'
            ]

            model_found = False
            for model_path in model_paths:
                if os.path.exists(model_path) and TENSORFLOW_AVAILABLE:
                    try:
                        self.model = tf.keras.models.load_model(model_path)
                        model_found = True
                        self.status_var.set(f"🤖 Modelo cargado: {os.path.basename(model_path)}")
                        break
                    except Exception as e:
                        print(f"Error cargando {model_path}: {e}")
                        continue

            # Cargar label encoder
            encoder_paths = [
                'models/label_encoder_balanced.pkl',
                'models/label_encoder.pkl'
            ]

            for encoder_path in encoder_paths:
                if os.path.exists(encoder_path):
                    try:
                        with open(encoder_path, 'rb') as f:
                            self.label_encoder = pickle.load(f)
                        break
                    except Exception as e:
                        print(f"Error cargando {encoder_path}: {e}")

            if not model_found:
                self.status_var.set("⚠️ Modelo no encontrado. Entrenar primero.")
                messagebox.showwarning(
                    "Modelo no encontrado",
                    "No se encontró un modelo entrenado.\nPor favor, ejecuta primero el entrenamiento."
                )

        except Exception as e:
            self.status_var.set(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Error cargando modelo: {str(e)}")

    def init_empty_plot(self):
        """Inicializa una gráfica vacía"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, '🎵 Selecciona un audio para ver el espectrograma',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Espectrograma de Audio')
        self.canvas.draw()

    def select_file(self):
        """Selecciona un archivo de audio"""
        file_types = [
            ("Archivos de Audio", "*.wav *.mp3 *.flac *.m4a"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("FLAC files", "*.flac"),
            ("Todos los archivos", "*.*")
        ]

        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo de audio",
            filetypes=file_types
        )

        if file_path:
            self.current_file_path = file_path
            filename = os.path.basename(file_path)
            self.file_path_var.set(filename)
            self.status_var.set("📁 Archivo cargado")

            # Cargar y visualizar audio
            try:
                audio, sr = librosa.load(file_path, sr=22050)
                self.recorded_audio = audio

                # Mostrar información del archivo
                duration = len(audio) / sr
                file_size = os.path.getsize(file_path) / 1024  # KB
                self.file_info_var.set(f"Duración: {duration:.1f}s | Tamaño: {file_size:.1f} KB")

                self.visualize_audio(audio, sr)
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando archivo: {str(e)}")
                self.file_info_var.set("Error al cargar archivo")

    def visualize_audio(self, audio, sr=22050):
        """Visualiza el espectrograma del audio"""
        self.figure.clear()

        # Crear subplots
        gs = self.figure.add_gridspec(2, 1, height_ratios=[1, 2])

        # Gráfica de onda
        ax1 = self.figure.add_subplot(gs[0])
        time_axis = np.linspace(0, len(audio) / sr, len(audio))
        ax1.plot(time_axis, audio, color='blue', linewidth=0.5)
        ax1.set_title('Forma de Onda')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Amplitud')
        ax1.grid(True, alpha=0.3)

        # Espectrograma
        ax2 = self.figure.add_subplot(gs[1])
        D = librosa.stft(audio)
        DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        img = librosa.display.specshow(
            DB, sr=sr, x_axis='time', y_axis='hz', ax=ax2, cmap='viridis'
        )
        ax2.set_title('Espectrograma')
        self.figure.colorbar(img, ax=ax2, format='%+2.0f dB')

        self.figure.tight_layout()
        self.canvas.draw()

    def play_audio(self):
        """Reproduce el audio actual"""
        if self.current_file_path and os.path.exists(self.current_file_path):
            try:
                # Usar pygame para reproducir audio
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(self.current_file_path)
                pygame.mixer.music.play()
                self.status_var.set("▶️ Reproduciendo...")
            except ImportError:
                messagebox.showinfo(
                    "Reproducción no disponible",
                    "Instala pygame para reproducción: pip install pygame"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Error reproduciendo audio: {str(e)}")
        else:
            messagebox.showwarning("Sin audio", "No hay audio cargado para reproducir")

    def stop_audio(self):
        """Detiene la reproducción de audio"""
        try:
            import pygame
            pygame.mixer.music.stop()
            self.status_var.set("⏹️ Reproducción detenida")
        except ImportError:
            pass
        except Exception as e:
            print(f"Error deteniendo audio: {e}")

    def analyze_emotion(self):
        """Analiza la emoción del audio actual - MEJORADO"""
        if not self.model or not self.label_encoder:
            messagebox.showwarning(
                "Modelo no disponible",
                "No hay modelo cargado. Entrena el modelo primero."
            )
            return

        if self.recorded_audio is None:
            messagebox.showwarning("Sin audio", "No hay audio para analizar")
            return

        try:
            self.status_var.set("🔍 Analizando emoción...")

            # Extraer espectrograma con preprocesamiento mejorado
            spectrogram = self.extract_mel_spectrogram(self.recorded_audio, 22050)

            if spectrogram is None:
                messagebox.showerror("Error", "No se pudo extraer el espectrograma")
                return

            # Preparar para predicción - el espectrograma ya es RGB (128,128,3)
            input_data = np.expand_dims(spectrogram, axis=0)  # Batch dimension (1,128,128,3)

            print(f"Input shape: {input_data.shape}")  # Debug
            print(f"Input range: {input_data.min():.3f} - {input_data.max():.3f}")  # Debug

            # Hacer predicción
            prediction = self.model.predict(input_data, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]

            # Debug: mostrar todas las probabilidades
            for i, prob in enumerate(prediction[0]):
                emotion_name = self.label_encoder.classes_[i]
                print(f"{emotion_name}: {prob:.3f}")

            # Obtener nombre de la emoción
            emotion_name = self.label_encoder.classes_[predicted_class]

            # Actualizar interfaz
            emotion_emoji = self.get_emotion_emoji(emotion_name)
            self.emotion_var.set(f"{emotion_emoji} {emotion_name.upper()}")
            self.confidence_var.set(f"Confianza: {confidence:.2%}")

            # Agregar al historial
            timestamp = time.strftime("%H:%M:%S")
            filename = os.path.basename(self.current_file_path) if self.current_file_path else "Sin archivo"
            history_entry = f"{timestamp} - {filename}: {emotion_name} ({confidence:.1%})"

            self.prediction_history.append({
                'time': timestamp,
                'file': filename,
                'emotion': emotion_name,
                'confidence': confidence
            })

            self.history_listbox.insert(0, history_entry)

            # Visualizar predicciones
            self.visualize_prediction(prediction[0])

            self.status_var.set(f"✅ Emoción detectada: {emotion_name}")

        except Exception as e:
            messagebox.showerror("Error de Análisis", f"Error analizando emoción: {str(e)}")
            self.status_var.set("❌ Error en análisis")
            import traceback
            traceback.print_exc()  # Para debug

    def extract_mel_spectrogram(self, audio, sr, target_shape=(128, 128)):
        """Extrae espectrograma MEL del audio - MEJORADO para coincidir con entrenamiento"""
        try:
            # Parámetros EXACTOS del entrenamiento
            n_mels = 128
            hop_length = 512
            n_fft = 2048

            # Normalizar audio primero (como en entrenamiento)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            # Extraer espectrograma MEL
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=n_mels,
                hop_length=hop_length, n_fft=n_fft
            )

            # Convertir a dB
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Redimensionar EXACTAMENTE como en entrenamiento
            if mel_spec_db.shape != target_shape:
                from skimage.transform import resize
                mel_spec_db = resize(mel_spec_db, target_shape, mode='constant')

            # Normalizar entre 0 y 1 (EXACTO como entrenamiento)
            mel_spec_min = mel_spec_db.min()
            mel_spec_max = mel_spec_db.max()

            if mel_spec_max > mel_spec_min:
                mel_spec_normalized = (mel_spec_db - mel_spec_min) / (mel_spec_max - mel_spec_min)
            else:
                mel_spec_normalized = np.zeros_like(mel_spec_db)

            # Convertir a RGB (como espectrogramas guardados)
            mel_spec_rgb = np.stack([mel_spec_normalized] * 3, axis=-1)

            return mel_spec_rgb

        except Exception as e:
            print(f"Error extrayendo espectrograma: {e}")
            return None

    def get_emotion_emoji(self, emotion):
        """Obtiene emoji correspondiente a la emoción"""
        emoji_map = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'neutral': '😐',
            'fear': '😨',
            'disgust': '🤢',
            'surprised': '😲',
            'calm': '😌'
        }
        return emoji_map.get(emotion.lower(), '🎭')

    def visualize_prediction(self, prediction):
        """Visualiza las probabilidades de predicción"""
        emotions = self.label_encoder.classes_

        # Limpiar y crear nueva visualización
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])

        # Mantener espectrograma en la parte superior izquierda
        if self.recorded_audio is not None:
            ax1 = self.figure.add_subplot(gs[0, :])
            D = librosa.stft(self.recorded_audio)
            DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(
                DB, sr=22050, x_axis='time', y_axis='hz',
                ax=ax1, cmap='viridis'
            )
            ax1.set_title('Espectrograma del Audio Analizado')

        # Gráfica de barras de probabilidades
        ax2 = self.figure.add_subplot(gs[1, 0])
        colors = plt.cm.Set3(range(len(emotions)))
        bars = ax2.bar(emotions, prediction, color=colors, alpha=0.7)
        ax2.set_title('Probabilidades por Emoción')
        ax2.set_ylabel('Probabilidad')
        ax2.set_ylim(0, 1)

        # Agregar valores en las barras
        for bar, prob in zip(bars, prediction):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{prob:.2%}', ha='center', va='bottom', fontsize=8)

        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Gráfica circular de la predicción principal
        ax3 = self.figure.add_subplot(gs[1, 1])
        max_idx = np.argmax(prediction)
        max_emotion = emotions[max_idx]
        max_prob = prediction[max_idx]

        ax3.pie([max_prob, 1 - max_prob],
                labels=[max_emotion, 'Otras'],
                autopct='%1.1f%%',
                colors=[colors[max_idx], 'lightgray'],
                startangle=90)
        ax3.set_title('Predicción Principal')

        self.figure.tight_layout()
        self.canvas.draw()

    def clear_history(self):
        """Limpia el historial de predicciones"""
        self.history_listbox.delete(0, tk.END)
        self.prediction_history.clear()

    def on_closing(self):
        """Función llamada al cerrar la aplicación"""
        self.root.destroy()


def main():
    """Función principal"""
    print("🎭 Iniciando Interfaz Gráfica de Detección de Emociones")
    print("=" * 60)

    # Verificar dependencias esenciales
    missing_deps = []

    try:
        import librosa
    except ImportError:
        missing_deps.append("librosa")

    try:
        from skimage.transform import resize
    except ImportError:
        missing_deps.append("scikit-image")

    if missing_deps:
        print("⚠️ Dependencias faltantes:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstala con: pip install " + " ".join(missing_deps))
        return

    # Crear y ejecutar interfaz
    root = tk.Tk()
    app = EmotionDetectionGUI(root)

    # Configurar cierre de aplicación
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    print("✅ Interfaz iniciada. ¡Sube archivos de audio para detectar emociones!")
    root.mainloop()


if __name__ == "__main__":
    main()
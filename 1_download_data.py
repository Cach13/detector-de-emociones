#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT 1: DESCARGAR Y ORGANIZAR DATOS
Este script descarga un dataset simple de emociones o te ayuda a crear uno propio
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf


def create_sample_audio_files():
    """
    Crea archivos de audio de muestra para probar el sistema
    (Para cuando no tienes acceso a datasets reales)
    """
    print("🎵 Creando archivos de audio de muestra...")

    # Crear directorio
    os.makedirs('data/raw_audio', exist_ok=True)

    # Definir emociones
    emotions = ['happy', 'sad', 'angry', 'neutral']

    # Generar archivos de audio sintéticos
    sample_rate = 22050
    duration = 3  # 3 segundos

    for emotion in emotions:
        emotion_dir = f'data/raw_audio/{emotion}'
        os.makedirs(emotion_dir, exist_ok=True)

        print(f"Creando muestras para: {emotion}")

        for i in range(10):  # 10 archivos por emoción
            # Generar audio sintético (solo para testing)
            t = np.linspace(0, duration, sample_rate * duration)

            if emotion == 'happy':
                # Frecuencia alta, vibrante
                frequency = 440 + np.random.randint(-50, 100)
                audio = 0.3 * np.sin(2 * np.pi * frequency * t) * np.exp(-t / 2)
            elif emotion == 'sad':
                # Frecuencia baja, descendente
                frequency = 220 + np.random.randint(-30, 50)
                audio = 0.2 * np.sin(2 * np.pi * frequency * t) * (1 - t / duration)
            elif emotion == 'angry':
                # Frecuencia irregular, rugosa
                frequency = 330 + np.random.randint(-100, 100)
                noise = np.random.normal(0, 0.1, len(t))
                audio = 0.4 * np.sin(2 * np.pi * frequency * t) + noise
            else:  # neutral
                # Frecuencia media, estable
                frequency = 300 + np.random.randint(-20, 20)
                audio = 0.25 * np.sin(2 * np.pi * frequency * t)

            # Guardar archivo
            filename = f'{emotion_dir}/{emotion}_{i:02d}.wav'
            sf.write(filename, audio, sample_rate)

        print(f"✅ {emotion}: 10 archivos creados")


def download_dataset_urls():
    """
    Muestra URLs de datasets reales que puedes descargar manualmente
    """
    print("\n📂 DATASETS RECOMENDADOS PARA DESCARGAR:")
    print("=" * 50)

    datasets = [
        {
            "name": "RAVDESS (Básico)",
            "url": "https://www.kaggle.com/uwrfkaggle/ravdess-emotional-speech-audio",
            "size": "~1GB",
            "emotions": "8 emociones",
            "description": "Dataset completo de actores"
        },
        {
            "name": "CREMA-D",
            "url": "https://www.kaggle.com/ejlok1/cremad",
            "size": "~800MB",
            "emotions": "6 emociones",
            "description": "91 actores con diferentes emociones"
        },
        {
            "name": "TESS",
            "url": "https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess",
            "size": "~400MB",
            "emotions": "7 emociones",
            "description": "Dataset de Toronto"
        }
    ]

    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset['name']}")
        print(f"   URL: {dataset['url']}")
        print(f"   Tamaño: {dataset['size']}")
        print(f"   Emociones: {dataset['emotions']}")
        print(f"   Descripción: {dataset['description']}")
        print()

    print("📋 INSTRUCCIONES:")
    print("1. Ve a Kaggle.com y crea una cuenta (gratis)")
    print("2. Descarga uno de los datasets")
    print("3. Extrae los archivos en: data/raw_audio/")
    print("4. Organiza por carpetas de emoción")


def organize_downloaded_data():
    """
    Ayuda a organizar datos descargados en la estructura correcta
    """
    print("\n📁 ORGANIZANDO DATOS DESCARGADOS...")

    raw_audio_path = Path('data/raw_audio')

    if not raw_audio_path.exists():
        print("❌ No se encuentra la carpeta data/raw_audio")
        return

    # Buscar archivos de audio
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(raw_audio_path.glob(f"**/{ext}"))

    if not audio_files:
        print("❌ No se encontraron archivos de audio")
        print("Asegúrate de haber copiado los archivos en data/raw_audio/")
        return

    print(f"✅ Encontrados {len(audio_files)} archivos de audio")

    # Mostrar estructura actual
    print("\n📂 Estructura actual:")
    for audio_file in audio_files[:10]:  # Mostrar solo los primeros 10
        print(f"   {audio_file}")

    if len(audio_files) > 10:
        print(f"   ... y {len(audio_files) - 10} archivos más")


def record_your_own_audio():
    """
    Instrucciones para grabar tu propio audio
    """
    print("\n🎤 CÓMO GRABAR TU PROPIO AUDIO:")
    print("=" * 40)

    emotions = ['happy', 'sad', 'angry', 'neutral']

    print("📋 INSTRUCCIONES:")
    print("1. Usa cualquier app de grabación (móvil, Audacity, etc.)")
    print("2. Configura a 44.1 kHz, formato WAV")
    print("3. Graba 10-15 frases por emoción")
    print("4. Duración: 3-5 segundos por archivo")
    print("5. Ambiente silencioso")

    print("\n🎭 FRASES SUGERIDAS POR EMOCIÓN:")

    phrases = {
        'happy': [
            "¡Qué día tan hermoso!",
            "Estoy muy feliz",
            "Me encanta este lugar",
            "¡Excelente noticia!"
        ],
        'sad': [
            "Me siento muy triste",
            "Qué pena me da esto",
            "No puedo más",
            "Todo está mal"
        ],
        'angry': [
            "¡Esto me molesta mucho!",
            "¡Ya no aguanto más!",
            "¡Qué injusticia!",
            "¡Estoy furioso!"
        ],
        'neutral': [
            "Hoy es lunes",
            "Son las tres de la tarde",
            "El cielo está nublado",
            "Voy a estudiar"
        ]
    }

    for emotion in emotions:
        print(f"\n🎭 {emotion.upper()}:")
        for phrase in phrases[emotion]:
            print(f"   - {phrase}")

    print("\n📁 ORGANIZACIÓN:")
    for emotion in emotions:
        emotion_dir = f'data/raw_audio/{emotion}'
        os.makedirs(emotion_dir, exist_ok=True)
        print(f"   Guardar archivos de {emotion} en: {emotion_dir}/")


def main():
    """Función principal"""
    print("🎵 OBTENCIÓN DE DATOS PARA DETECCIÓN DE EMOCIONES")
    print("=" * 60)

    print("\nElige una opción:")
    print("1. Crear archivos de muestra (para probar el sistema)")
    print("2. Mostrar URLs de datasets reales")
    print("3. Organizar datos ya descargados")
    print("4. Instrucciones para grabar tu propio audio")

    while True:
        choice = input("\nIngresa tu opción (1-4): ").strip()

        if choice == '1':
            create_sample_audio_files()
            break
        elif choice == '2':
            download_dataset_urls()
            break
        elif choice == '3':
            organize_downloaded_data()
            break
        elif choice == '4':
            record_your_own_audio()
            break
        else:
            print("❌ Opción inválida. Ingresa 1, 2, 3 o 4")

    print("\n✅ ¡Listo! Continúa con: python 2_preprocess_audio.py")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT 2: PROCESAMIENTO DE AUDIO Y CREACIÓN DE ESPECTROGRAMAS (VERSIÓN MEJORADA)
Este script detecta automáticamente el formato del dataset y procesa los archivos
"""

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import re
from collections import Counter


class SmartAudioPreprocessor:
    """Clase inteligente para procesar múltiples formatos de datasets de audio"""

    def __init__(self, target_sr=22050, n_mels=128, hop_length=512, n_fft=2048):
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft

        # Mapeos de emociones para diferentes datasets
        self.emotion_mappings = {
            # CREMA-D / TESS format
            'crema': {
                'ANG': 'angry',
                'DIS': 'disgust',
                'FEA': 'fear',
                'HAP': 'happy',
                'NEU': 'neutral',
                'SAD': 'sad'
            },
            # RAVDESS format
            'ravdess': {
                '01': 'neutral',
                '02': 'calm',
                '03': 'happy',
                '04': 'sad',
                '05': 'angry',
                '06': 'fear',
                '07': 'disgust',
                '08': 'surprised'
            },
            # Formato por carpetas
            'folders': {
                'angry': 'angry',
                'happy': 'happy',
                'sad': 'sad',
                'neutral': 'neutral',
                'fear': 'fear',
                'disgust': 'disgust',
                'surprised': 'surprised',
                'calm': 'calm'
            }
        }

        # Crear directorios
        os.makedirs('data/spectrograms/train', exist_ok=True)
        os.makedirs('data/spectrograms/test', exist_ok=True)
        os.makedirs('results', exist_ok=True)

    def detect_dataset_format(self, data_path):
        """
        Detecta automáticamente el formato del dataset
        """
        print("🔍 Detectando formato del dataset...")

        # Buscar archivos de audio
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac']:
            audio_files.extend(list(Path(data_path).glob(f'*{ext}')))
            audio_files.extend(list(Path(data_path).glob(f'*/*{ext}')))

        if not audio_files:
            print("❌ No se encontraron archivos de audio")
            return None, []

        print(f"✅ Encontrados {len(audio_files)} archivos de audio")

        # Verificar si están organizados en carpetas por emoción
        subfolder_names = [f.parent.name.lower() for f in audio_files if f.parent.name != data_path.split('/')[-1]]
        if subfolder_names:
            emotion_folders = set(subfolder_names)
            known_emotions = {'angry', 'happy', 'sad', 'neutral', 'fear', 'disgust', 'surprised', 'calm'}
            if len(emotion_folders & known_emotions) > 2:
                print(f"📁 Formato detectado: CARPETAS POR EMOCIÓN")
                print(f"📂 Emociones encontradas: {sorted(emotion_folders & known_emotions)}")
                return 'folders', audio_files

        # Analizar nombres de archivos para detectar formato
        sample_files = [f.name for f in audio_files[:20]]  # Analizar muestra

        # Detectar CREMA-D/TESS format (1001_DFA_ANG_XX.wav)
        crema_pattern = r'.*_(ANG|DIS|FEA|HAP|NEU|SAD)_'
        crema_matches = [re.search(crema_pattern, f) for f in sample_files]
        crema_matches = [m for m in crema_matches if m]

        if len(crema_matches) > len(sample_files) * 0.5:  # >50% coincidencias
            print("📁 Formato detectado: CREMA-D/TESS")
            emotions_found = [m.group(1) for m in crema_matches]
            print(f"😊 Emociones encontradas: {sorted(set(emotions_found))}")
            return 'crema', audio_files

        # Detectar RAVDESS format (03-01-03-01-01-01-01.wav)
        ravdess_pattern = r'03-01-(\d{2})-'
        ravdess_matches = [re.search(ravdess_pattern, f) for f in sample_files]
        ravdess_matches = [m for m in ravdess_matches if m]

        if len(ravdess_matches) > len(sample_files) * 0.5:  # >50% coincidencias
            print("📁 Formato detectado: RAVDESS")
            emotions_found = [m.group(1) for m in ravdess_matches]
            print(f"😊 Emociones encontradas: {sorted(set(emotions_found))}")
            return 'ravdess', audio_files

        # Si no se detecta un formato específico, asumir carpetas
        print("📁 Formato detectado: ARCHIVOS SUELTOS (se organizarán manualmente)")
        return 'unknown', audio_files

    def extract_emotion_from_filename(self, filename, dataset_format):
        """
        Extrae la emoción del nombre del archivo según el formato detectado
        """
        if dataset_format == 'crema':
            # CREMA-D format: 1001_DFA_ANG_XX.wav
            match = re.search(r'.*_(ANG|DIS|FEA|HAP|NEU|SAD)_', filename)
            if match:
                emotion_code = match.group(1)
                return self.emotion_mappings['crema'].get(emotion_code, 'unknown')

        elif dataset_format == 'ravdess':
            # RAVDESS format: 03-01-03-01-01-01-01.wav
            match = re.search(r'03-01-(\d{2})-', filename)
            if match:
                emotion_code = match.group(1)
                return self.emotion_mappings['ravdess'].get(emotion_code, 'unknown')

        elif dataset_format == 'folders':
            # Usar el nombre de la carpeta padre
            return filename  # Se procesará por separado

        return 'unknown'

    def organize_files_by_emotion(self, audio_files, dataset_format):
        """
        Organiza archivos por emoción según el formato detectado
        """
        organized_data = []

        for file_path in audio_files:
            if dataset_format == 'folders':
                # Si están en carpetas, usar el nombre de la carpeta
                emotion = file_path.parent.name.lower()
                if emotion not in self.emotion_mappings['folders']:
                    continue
            else:
                # Extraer emoción del nombre del archivo
                emotion = self.extract_emotion_from_filename(file_path.name, dataset_format)
                if emotion == 'unknown':
                    continue

            organized_data.append({
                'file_path': str(file_path),
                'emotion': emotion,
                'filename': file_path.name
            })

        return organized_data

    def extract_mel_spectrogram(self, audio_path, target_shape=(128, 128)):
        """
        Extrae espectrograma MEL de un archivo de audio
        """
        try:
            # Cargar audio
            y, sr = librosa.load(audio_path, sr=self.target_sr)

            # Normalizar audio
            y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

            # Extraer espectrograma MEL
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels,
                hop_length=self.hop_length, n_fft=self.n_fft
            )

            # Convertir a dB
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Redimensionar si es necesario
            if mel_spec_db.shape != target_shape:
                from skimage.transform import resize
                mel_spec_db = resize(mel_spec_db, target_shape, mode='constant')

            # Normalizar entre 0 y 1
            mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

            return mel_spec_normalized

        except Exception as e:
            print(f"❌ Error procesando {audio_path}: {str(e)}")
            return None

    def save_spectrogram_as_image(self, spectrogram, output_path):
        """
        Guarda espectrograma como imagen
        """
        plt.figure(figsize=(4, 4))
        plt.imshow(spectrogram, cmap='viridis', aspect='auto', origin='lower')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=32)
        plt.close()

    def create_sample_visualizations(self, organized_data):
        """
        Crea visualizaciones de muestra del dataset
        """
        print("📊 Creando visualizaciones de muestra...")

        # Contar emociones
        emotion_counts = Counter([item['emotion'] for item in organized_data])

        # Gráfica de distribución de emociones
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        colors = plt.cm.Set3(range(len(emotions)))

        bars = plt.bar(emotions, counts, color=colors)
        plt.title('Distribución de Emociones en el Dataset', fontsize=14, fontweight='bold')
        plt.xlabel('Emociones')
        plt.ylabel('Número de Archivos')
        plt.xticks(rotation=45)

        # Agregar números en las barras
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(counts),
                     str(count), ha='center', va='bottom', fontweight='bold')

        # Gráfica de porcentajes
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=emotions, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Distribución Porcentual', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('results/dataset_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Crear muestras de espectrogramas
        print("🎵 Creando muestras de espectrogramas...")

        # Seleccionar una muestra de cada emoción
        sample_files = {}
        for item in organized_data:
            emotion = item['emotion']
            if emotion not in sample_files and len(sample_files) < 6:
                sample_files[emotion] = item['file_path']

        # Crear figura con espectrogramas de muestra
        n_samples = len(sample_files)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (emotion, file_path) in enumerate(sample_files.items()):
            if idx >= 6:
                break

            # Extraer espectrograma
            spectrogram = self.extract_mel_spectrogram(file_path)
            if spectrogram is not None:
                axes[idx].imshow(spectrogram, cmap='viridis', aspect='auto', origin='lower')
                axes[idx].set_title(f'{emotion.capitalize()}', fontsize=12, fontweight='bold')
                axes[idx].axis('off')

        # Ocultar ejes sobrantes
        for idx in range(n_samples, 6):
            axes[idx].axis('off')

        plt.suptitle('Muestras de Espectrogramas por Emoción', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/sample_spectrograms.png', dpi=300, bbox_inches='tight')
        plt.close()

    def process_dataset(self):
        """
        Función principal para procesar todo el dataset
        """
        print("🎵 PROCESAMIENTO INTELIGENTE DE AUDIO")
        print("=" * 60)

        # Detectar formato del dataset
        data_path = 'data/raw_audio'
        dataset_format, audio_files = self.detect_dataset_format(data_path)

        if not audio_files:
            print("❌ No se encontraron archivos de audio para procesar")
            return

        # Organizar archivos por emoción
        print("\n📂 Organizando archivos por emoción...")
        organized_data = self.organize_files_by_emotion(audio_files, dataset_format)

        if not organized_data:
            print("❌ No se pudieron organizar los archivos por emoción")
            print("💡 Verifica que los nombres de archivos sigan un formato reconocible")
            return

        print(f"✅ Organizados {len(organized_data)} archivos")

        # Mostrar estadísticas
        emotion_counts = Counter([item['emotion'] for item in organized_data])
        print(f"\n📊 Distribución de emociones:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"   😊 {emotion}: {count} archivos")

        # Crear visualizaciones
        self.create_sample_visualizations(organized_data)

        # Dividir en entrenamiento y prueba
        print(f"\n🔀 Dividiendo dataset (80% entrenamiento, 20% prueba)...")
        train_data, test_data = train_test_split(
            organized_data, test_size=0.2,
            stratify=[item['emotion'] for item in organized_data],
            random_state=42
        )

        print(f"📚 Datos de entrenamiento: {len(train_data)} archivos")
        print(f"🧪 Datos de prueba: {len(test_data)} archivos")

        # Procesar archivos de entrenamiento
        print(f"\n🔄 Procesando archivos de entrenamiento...")
        train_processed = self.process_split(train_data, 'train')

        # Procesar archivos de prueba
        print(f"\n🔄 Procesando archivos de prueba...")
        test_processed = self.process_split(test_data, 'test')

        # Guardar información del dataset
        self.save_dataset_info(train_processed, test_processed, emotion_counts, dataset_format)

        print(f"\n🎉 ¡PROCESAMIENTO COMPLETADO!")
        print(f"📁 Espectrogramas guardados en: data/spectrograms/")
        print(f"📊 Visualizaciones guardadas en: results/")
        print(f"📋 Información del dataset: data/dataset_info.csv")
        print(f"\n➡️  Continúa con: python 3_train_model.py")

    def process_split(self, data_split, split_name):
        """
        Procesa una división del dataset (train o test)
        """
        processed_data = []

        for idx, item in enumerate(data_split):
            if idx % 50 == 0:
                print(f"   Procesando {idx + 1}/{len(data_split)} archivos...")

            # Extraer espectrograma
            spectrogram = self.extract_mel_spectrogram(item['file_path'])

            if spectrogram is not None:
                # Generar nombre de archivo de salida
                output_filename = f"{item['emotion']}_{idx:04d}.png"
                output_path = f"data/spectrograms/{split_name}/{output_filename}"

                # Guardar espectrograma como imagen
                self.save_spectrogram_as_image(spectrogram, output_path)

                processed_data.append({
                    'original_file': item['filename'],
                    'spectrogram_file': output_filename,
                    'emotion': item['emotion'],
                    'split': split_name
                })

        print(f"   ✅ Procesados {len(processed_data)} archivos para {split_name}")
        return processed_data

    def save_dataset_info(self, train_data, test_data, emotion_counts, dataset_format):
        """
        Guarda información del dataset procesado
        """
        # Información general del dataset
        all_data = train_data + test_data
        df_all = pd.DataFrame(all_data)

        # Guardar CSV con toda la información
        df_all.to_csv('data/dataset_info.csv', index=False)

        # Guardar CSVs separados para entrenamiento y prueba
        df_train = pd.DataFrame(train_data)
        df_test = pd.DataFrame(test_data)

        df_train.to_csv('data/train_data.csv', index=False)
        df_test.to_csv('data/test_data.csv', index=False)

        # Crear resumen
        summary = {
            'formato_dataset': dataset_format,
            'total_archivos': len(all_data),
            'archivos_entrenamiento': len(train_data),
            'archivos_prueba': len(test_data),
            'emociones': list(emotion_counts.keys()),
            'distribucion_emociones': dict(emotion_counts)
        }

        import json
        with open('results/dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    """
    Función principal
    """
    try:
        processor = SmartAudioPreprocessor()
        processor.process_dataset()

    except Exception as e:
        print(f"❌ Error durante el procesamiento: {str(e)}")
        print("💡 Verifica que los archivos de audio estén en 'data/raw_audio/'")


if __name__ == "__main__":
    main()
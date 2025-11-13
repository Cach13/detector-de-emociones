#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT DE DIAGNÓSTICO - ¿Por qué solo detecta tristeza?
Identifica problemas en el modelo y datos
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from pathlib import Path

try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow no disponible")


class ModelDiagnostic:
    """Clase para diagnosticar problemas del modelo"""

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.load_model_and_encoder()

    def load_model_and_encoder(self):
        """Cargar modelo y encoder"""
        print("🔍 DIAGNÓSTICO DEL MODELO")
        print("=" * 50)

        # Cargar modelo
        model_paths = [
            'models/balanced_emotion_model.h5',
            'models/emotion_model_balanced_final.h5',
            'models/emotion_model.h5'
        ]

        for model_path in model_paths:
            if os.path.exists(model_path) and TENSORFLOW_AVAILABLE:
                try:
                    self.model = tf.keras.models.load_model(model_path)
                    print(f"✅ Modelo cargado: {model_path}")
                    break
                except Exception as e:
                    print(f"❌ Error cargando {model_path}: {e}")

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
                    print(f"✅ Label encoder cargado: {encoder_path}")
                    break
                except Exception as e:
                    print(f"❌ Error cargando {encoder_path}: {e}")

    def check_data_distribution(self):
        """Verificar distribución de datos"""
        print("\n📊 1. DISTRIBUCIÓN DE DATOS")
        print("-" * 30)

        # Verificar archivos CSV
        csv_files = ['data/train_data.csv', 'data/test_data.csv', 'data/dataset_info.csv']

        for csv_file in csv_files:
            if os.path.exists(csv_file):
                print(f"\n📁 {csv_file}:")
                df = pd.read_csv(csv_file)

                if 'emotion' in df.columns:
                    emotion_counts = df['emotion'].value_counts()
                    print(emotion_counts)

                    # Verificar si hay desbalance extremo
                    total = len(df)
                    percentages = (emotion_counts / total * 100).round(1)

                    print("\nPorcentajes:")
                    for emotion, pct in percentages.items():
                        if pct > 80:
                            print(f"⚠️  {emotion}: {pct}% (DESBALANCEADO)")
                        elif pct < 5:
                            print(f"⚠️  {emotion}: {pct}% (MUY POCO)")
                        else:
                            print(f"✅ {emotion}: {pct}%")
            else:
                print(f"❌ {csv_file} no encontrado")

    def check_label_encoder(self):
        """Verificar label encoder"""
        print("\n🏷️  2. VERIFICACIÓN DEL LABEL ENCODER")
        print("-" * 40)

        if self.label_encoder is None:
            print("❌ Label encoder no disponible")
            return

        classes = self.label_encoder.classes_
        print(f"📋 Clases detectadas: {len(classes)}")

        for i, emotion in enumerate(classes):
            print(f"   {i}: {emotion}")

        # Verificar si 'sad' domina los índices
        if 'sad' in classes:
            sad_index = list(classes).index('sad')
            print(f"\n🎭 Índice de 'sad': {sad_index}")
            if sad_index == 0:
                print("⚠️ 'sad' está en índice 0 - puede estar siendo predicho por defecto")

    def test_model_predictions(self):
        """Probar predicciones del modelo"""
        print("\n🧠 3. PRUEBA DE PREDICCIONES")
        print("-" * 35)

        if self.model is None:
            print("❌ Modelo no disponible")
            return

        if self.label_encoder is None:
            print("❌ Label encoder no disponible")
            return

        # Crear datos de prueba sintéticos
        print("🔬 Creando datos de prueba sintéticos...")

        # Diferentes patrones de entrada
        test_patterns = {
            'zeros': np.zeros((128, 128, 3)),
            'ones': np.ones((128, 128, 3)),
            'random_low': np.random.random((128, 128, 3)) * 0.1,
            'random_high': np.random.random((128, 128, 3)) * 0.9 + 0.1,
            'gradient': np.tile(np.linspace(0, 1, 128), (128, 3, 1)).transpose(0, 2, 1)
        }

        predictions_summary = {}

        for pattern_name, pattern_data in test_patterns.items():
            # Expandir dimensiones para batch
            input_data = np.expand_dims(pattern_data, axis=0)

            try:
                prediction = self.model.predict(input_data, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                emotion = self.label_encoder.classes_[predicted_class]

                predictions_summary[pattern_name] = {
                    'emotion': emotion,
                    'confidence': confidence,
                    'all_probs': prediction[0]
                }

                print(f"📊 {pattern_name}: {emotion} ({confidence:.2%})")

            except Exception as e:
                print(f"❌ Error con {pattern_name}: {e}")

        # Verificar si todas las predicciones son iguales
        emotions_predicted = [pred['emotion'] for pred in predictions_summary.values()]
        unique_emotions = set(emotions_predicted)

        if len(unique_emotions) == 1:
            print(f"\n⚠️ PROBLEMA: Todas las predicciones son '{emotions_predicted[0]}'")
            print("   Esto indica un problema en el modelo.")
        else:
            print(f"\n✅ El modelo predice {len(unique_emotions)} emociones diferentes")

        return predictions_summary

    def check_model_architecture(self):
        """Verificar arquitectura del modelo"""
        print("\n🏗️  4. ARQUITECTURA DEL MODELO")
        print("-" * 35)

        if self.model is None:
            print("❌ Modelo no disponible")
            return

        print("📋 Resumen del modelo:")
        self.model.summary()

        # Verificar capas finales
        print(f"\n🎯 Capa final: {self.model.layers[-1].name}")
        print(f"📊 Unidades de salida: {self.model.layers[-1].units}")
        print(f"🔄 Activación: {self.model.layers[-1].activation}")

        # Verificar si el número de clases coincide
        if self.label_encoder:
            expected_classes = len(self.label_encoder.classes_)
            actual_output = self.model.layers[-1].units

            if expected_classes != actual_output:
                print(f"⚠️ PROBLEMA: Clases esperadas ({expected_classes}) != Salidas del modelo ({actual_output})")
            else:
                print(f"✅ Número de clases coincide: {expected_classes}")

    def analyze_real_spectrograms(self):
        """Analizar espectrogramas reales si están disponibles"""
        print("\n📸 5. ANÁLISIS DE ESPECTROGRAMAS REALES")
        print("-" * 45)

        spectrogram_dirs = ['data/spectrograms/train', 'data/spectrograms/test']

        for spec_dir in spectrogram_dirs:
            if os.path.exists(spec_dir):
                files = list(Path(spec_dir).glob('*.png'))[:5]  # Solo 5 archivos

                print(f"\n📁 Analizando {len(files)} archivos de {spec_dir}")

                for file_path in files:
                    try:
                        from PIL import Image
                        img = Image.open(file_path).convert('RGB')
                        img_array = np.array(img.resize((128, 128))) / 255.0

                        # Expandir dimensiones
                        input_data = np.expand_dims(img_array, axis=0)

                        prediction = self.model.predict(input_data, verbose=0)
                        predicted_class = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class]
                        emotion = self.label_encoder.classes_[predicted_class]

                        filename = file_path.name
                        print(f"   📊 {filename}: {emotion} ({confidence:.2%})")

                    except Exception as e:
                        print(f"   ❌ Error con {file_path.name}: {e}")

    def generate_recommendations(self):
        """Generar recomendaciones basadas en el diagnóstico"""
        print("\n💡 6. RECOMENDACIONES")
        print("-" * 25)

        recommendations = []

        # Verificar si solo predice tristeza
        if hasattr(self, 'test_results'):
            # Lógica basada en resultados
            pass

        print("🔧 POSIBLES SOLUCIONES:")
        print("1. 📊 Verificar balance de datos:")
        print("   python 2_preprocess_audio.py  # Recrear dataset balanceado")

        print("\n2. 🔄 Re-entrenar con más epochs:")
        print("   # Editar 3_train_balanced_model.py, epochs=50")

        print("\n3. 🎯 Usar data augmentation:")
        print("   # Agregar transformaciones a los espectrogramas")

        print("\n4. 🧠 Verificar función de pérdida:")
        print("   # Usar categorical_crossentropy con class_weight")

        print("\n5. 🔀 Mezclar mejor los datos:")
        print("   # Verificar shuffle=True en el entrenamiento")

    def create_diagnostic_plots(self):
        """Crear gráficas de diagnóstico"""
        print("\n📈 7. CREANDO GRÁFICAS DIAGNÓSTICAS")
        print("-" * 40)

        plt.figure(figsize=(15, 10))

        # Gráfica 1: Distribución de datos
        plt.subplot(2, 3, 1)
        try:
            if os.path.exists('data/train_data.csv'):
                df = pd.read_csv('data/train_data.csv')
                emotion_counts = df['emotion'].value_counts()
                plt.bar(emotion_counts.index, emotion_counts.values, alpha=0.7)
                plt.title('Distribución de Emociones - Entrenamiento')
                plt.xticks(rotation=45)
        except Exception as e:
            plt.text(0.5, 0.5, f'Error: {e}', ha='center', va='center')

        # Gráfica 2: Predicciones de prueba
        if hasattr(self, 'test_results') and self.test_results:
            plt.subplot(2, 3, 2)
            emotions = [pred['emotion'] for pred in self.test_results.values()]
            emotion_counts = pd.Series(emotions).value_counts()
            plt.bar(emotion_counts.index, emotion_counts.values, alpha=0.7, color='orange')
            plt.title('Predicciones en Datos Sintéticos')
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('results/diagnostic_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Gráficas guardadas: results/diagnostic_analysis.png")
        plt.close()

    def run_full_diagnostic(self):
        """Ejecutar diagnóstico completo"""
        self.check_data_distribution()
        self.check_label_encoder()
        self.test_results = self.test_model_predictions()
        self.check_model_architecture()
        self.analyze_real_spectrograms()
        self.create_diagnostic_plots()
        self.generate_recommendations()

        print("\n🎉 DIAGNÓSTICO COMPLETADO")
        print("=" * 50)
        print("📊 Revisa 'results/diagnostic_analysis.png' para más detalles")


def main():
    """Función principal"""
    diagnostic = ModelDiagnostic()
    diagnostic.run_full_diagnostic()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT 3: ENTRENAMIENTO BALANCEADO PARA 80% ACCURACY
Versión que mantiene lo bueno del primer modelo con mejoras moderadas
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import json


class BalancedEmotionCNN:
    """CNN balanceada - mantiene lo que funcionó, mejora gradualmente"""

    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.model = None
        self.label_encoder = None
        self.history = None

        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)

    def load_data(self):
        """Carga datos igual que el modelo que funcionó"""
        print("📂 Cargando datos con configuración balanceada...")

        try:
            train_df = pd.read_csv('data/train_data.csv')
            test_df = pd.read_csv('data/test_data.csv')
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            return None, None, None, None

        print(f"📊 Datos cargados:")
        print(f"   Entrenamiento: {len(train_df)} muestras")
        print(f"   Prueba: {len(test_df)} muestras")

        # Cargar espectrogramas
        X_train, y_train = self._load_spectrograms_and_labels(train_df, 'train')
        X_test, y_test = self._load_spectrograms_and_labels(test_df, 'test')

        if X_train is None or X_test is None:
            return None, None, None, None

        # Label encoder
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        self.num_classes = len(self.label_encoder.classes_)
        print(f"🎭 Emociones: {list(self.label_encoder.classes_)}")

        # Convertir a categorical
        y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded, self.num_classes)
        y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded, self.num_classes)

        # Normalizar
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        print(f"✅ Datos preparados:")
        print(f"   Shape: {X_train.shape}")
        print(f"   Clases: {self.num_classes}")

        return X_train, X_test, y_train_categorical, y_test_categorical

    def _load_spectrograms_and_labels(self, df, split_name):
        """Carga espectrogramas con la configuración que funcionó"""
        spectrograms = []
        labels = []

        for idx, row in df.iterrows():
            try:
                img_filename = row['spectrogram_file']
                img_path = f"data/spectrograms/{split_name}/{img_filename}"

                if not os.path.exists(img_path):
                    continue

                img = Image.open(img_path).convert('RGB')
                img = img.resize((128, 128))
                img_array = np.array(img)

                spectrograms.append(img_array)
                labels.append(row['emotion'])

                if (idx + 1) % 1000 == 0:
                    print(f"   Cargados {idx + 1}/{len(df)} archivos...")

            except Exception:
                continue

        print(f"✅ Cargados {len(spectrograms)} espectrogramas para {split_name}")
        return np.array(spectrograms), np.array(labels)

    def create_balanced_cnn(self):
        """
        CNN balanceada - similar al primer modelo pero con mejoras moderadas
        """
        print("🏗️ Creando CNN balanceada (basada en el modelo que funcionó)...")

        model = models.Sequential([
            # Primer bloque - similar al original pero con BatchNorm
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Segundo bloque - mantenemos la estructura que funcionó
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Tercer bloque - igual que el modelo original
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Clasificador - mantenemos estructura similar pero mejorada
            layers.GlobalAveragePooling2D(),  # Cambio moderado del Flatten original
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Optimizer balanceado - learning rate moderado
        optimizer = Adam(
            learning_rate=0.0005,  # Más conservador que 0.002, mejor que 0.001
            beta_1=0.9,
            beta_2=0.999
        )

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("✅ Modelo balanceado creado")
        print(f"📊 Parámetros: {model.count_params():,}")

        self.model = model
        return model

    def train_balanced_model(self, X_train, y_train, X_test, y_test, epochs=35):
        """
        Entrenamiento balanceado - mejoras moderadas del primer modelo
        """
        print(f"🚀 Entrenamiento balanceado (mezcla lo mejor de ambos enfoques)")
        print(f"📊 Configuración balanceada:")
        print(f"   Epochs: {epochs} (intermedio entre 25 y 50)")
        print(f"   Batch size: 32 (como el primer modelo)")
        print(f"   Learning rate: 0.0005 (balanceado)")
        print(f"   Arquitectura: Primer modelo + BatchNorm")

        # Callbacks balanceados
        callbacks = [
            EarlyStopping(
                patience=12,  # Más paciencia que el agresivo, menos que el original
                restore_best_weights=True,
                monitor='val_accuracy',
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                factor=0.5,
                patience=6,  # Intermedio
                min_lr=0.00001,
                monitor='val_loss',
                verbose=1
            ),
            ModelCheckpoint(
                'models/balanced_emotion_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
        ]

        print(f"\n🎯 ¡Entrenamiento balanceado iniciado!")
        print(f"⏰ Inicio: {pd.Timestamp.now()}")

        # Entrenar SIN data augmentation (volver a lo que funcionó)
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,  # Volver al batch size que funcionó
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        print(f"⏰ Fin: {pd.Timestamp.now()}")

        # Cargar mejor modelo
        self.model.load_weights('models/balanced_emotion_model.h5')

        # Guardar modelo final
        self.model.save('models/emotion_model_balanced_final.h5')

        # Guardar label encoder
        with open('models/label_encoder_balanced.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print("✅ Entrenamiento balanceado completado")
        return self.history

    def evaluate_balanced_model(self, X_test, y_test):
        """Evaluación del modelo balanceado"""
        print("📊 Evaluando modelo balanceado...")

        # Predicciones
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        # Accuracy
        accuracy = np.mean(y_pred_classes == y_test_classes)
        print(f"🎯 Accuracy balanceada: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Comparar con versiones anteriores
        print(f"📊 Comparación de resultados:")
        print(f"   Modelo original: 59.70%")
        print(f"   Modelo agresivo: 30.49%")
        print(f"   Modelo balanceado: {accuracy * 100:.2f}%")

        # Mostrar progreso hacia el objetivo
        if accuracy >= 0.80:
            print(f"🎉 ¡OBJETIVO ALCANZADO! Superaste el 80%")
        elif accuracy >= 0.70:
            print(f"🔥 ¡EXCELENTE! Solo {(0.8 - accuracy) * 100:.1f}% más para el 80%")
        elif accuracy >= 0.60:
            print(f"📈 BUEN PROGRESO. Necesitas {(0.8 - accuracy) * 100:.1f}% más")
        else:
            print(f"⚠️ Necesita más trabajo. Diferencia: {(0.8 - accuracy) * 100:.1f}%")

        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_test_classes, y_pred_classes,
            target_names=class_names,
            output_dict=True
        )

        print("\n📋 Reporte por emoción:")
        for emotion in class_names:
            metrics = report[emotion]
            print(f"   😊 {emotion}: Precisión={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")

        # Gráficas
        self._plot_balanced_results(accuracy)
        self._plot_comparison_confusion_matrix(y_test_classes, y_pred_classes, class_names)

        # Guardar métricas
        metrics = {
            'balanced_accuracy': float(accuracy),
            'comparison': {
                'original_model': 0.597,
                'aggressive_model': 0.305,
                'balanced_model': float(accuracy)
            },
            'target_reached': bool(accuracy >= 0.80),
            'classification_report': report,
            'recommendations': self._get_recommendations(accuracy)
        }

        with open('results/balanced_model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        return accuracy, report

    def _plot_balanced_results(self, accuracy):
        """Gráficas del modelo balanceado"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Accuracy con línea objetivo
        axes[0, 0].plot(self.history.history['accuracy'], 'b-', label='Entrenamiento', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], 'r-', label='Validación', linewidth=2)
        axes[0, 0].axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label='Objetivo 80%')
        axes[0, 0].axhline(y=0.597, color='orange', linestyle=':', alpha=0.7, label='Modelo Original')
        axes[0, 0].set_title('Accuracy - Modelo Balanceado', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        axes[0, 1].plot(self.history.history['loss'], 'b-', label='Entrenamiento', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], 'r-', label='Validación', linewidth=2)
        axes[0, 1].set_title('Loss - Modelo Balanceado', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Comparación de modelos
        models = ['Original\n(59.70%)', 'Agresivo\n(30.49%)', f'Balanceado\n({accuracy * 100:.2f}%)',
                  'Objetivo\n(80.00%)']
        accuracies = [0.597, 0.305, accuracy, 0.8]
        colors = ['skyblue', 'red', 'lightgreen', 'gold']

        bars = axes[1, 0].bar(models, accuracies, color=colors, alpha=0.8)
        axes[1, 0].set_title('Comparación de Modelos', fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)

        # Agregar valores en las barras
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # Progreso hacia objetivo
        progress = accuracy / 0.8 * 100
        axes[1, 1].pie([progress, 100 - progress],
                       labels=[f'Logrado\n{accuracy * 100:.1f}%', f'Falta\n{(0.8 - accuracy) * 100:.1f}%'],
                       colors=['lightgreen', 'lightcoral'],
                       autopct='%1.1f%%',
                       startangle=90)
        axes[1, 1].set_title('Progreso hacia 80% de Accuracy', fontweight='bold')

        plt.tight_layout()
        plt.savefig('results/balanced_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("📈 Análisis balanceado: results/balanced_model_analysis.png")

    def _plot_comparison_confusion_matrix(self, y_true, y_pred, class_names):
        """Matriz de confusión mejorada"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Matriz de Confusión - Modelo Balanceado', fontsize=16, fontweight='bold')
        plt.xlabel('Predicción', fontsize=14)
        plt.ylabel('Etiqueta Real', fontsize=14)
        plt.tight_layout()
        plt.savefig('results/balanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("📊 Matriz de confusión: results/balanced_confusion_matrix.png")

    def _get_recommendations(self, accuracy):
        """Genera recomendaciones basadas en el resultado"""
        recommendations = []

        if accuracy < 0.65:
            recommendations.extend([
                "Aumentar epochs a 45-50",
                "Verificar calidad de los datos",
                "Considerar más data augmentation moderado"
            ])
        elif accuracy < 0.75:
            recommendations.extend([
                "Ajustar learning rate a 0.0003",
                "Usar early stopping con más paciencia",
                "Probar arquitectura ligeramente más profunda"
            ])
        elif accuracy < 0.80:
            recommendations.extend([
                "¡Muy cerca! Ejecutar con epochs=45",
                "Fine-tuning con learning rate 0.0001",
                "Verificar y balancear mejor las clases"
            ])
        else:
            recommendations.append("¡Objetivo alcanzado! Modelo listo para producción")

        return recommendations


def main():
    """Función principal balanceada"""
    print("⚖️  ENTRENAMIENTO BALANCEADO - LO MEJOR DE AMBOS MUNDOS")
    print("=" * 70)
    print("🎯 Objetivo: Combinar lo que funcionó (59.70%) con mejoras moderadas")

    try:
        # Crear modelo balanceado
        emotion_cnn = BalancedEmotionCNN()

        # Cargar datos
        X_train, X_test, y_train, y_test = emotion_cnn.load_data()
        if X_train is None:
            return

        # Crear modelo balanceado
        model = emotion_cnn.create_balanced_cnn()

        # Entrenamiento balanceado
        history = emotion_cnn.train_balanced_model(
            X_train, y_train, X_test, y_test,
            epochs=35  # Intermedio entre 25 y 50
        )

        # Evaluación
        accuracy, report = emotion_cnn.evaluate_balanced_model(X_test, y_test)

        # Resumen final
        print("\n🎉 ¡ENTRENAMIENTO BALANCEADO COMPLETADO!")
        print("=" * 70)
        print(f"📊 RESULTADOS COMPARATIVOS:")
        print(f"   🥉 Modelo original: 59.70%")
        print(f"   ❌ Modelo agresivo: 30.49%")
        print(f"   🏆 Modelo balanceado: {accuracy * 100:.2f}%")

        if accuracy > 0.597:
            print(f"✅ ¡MEJORA EXITOSA! +{(accuracy - 0.597) * 100:.2f}%")

        print(f"\n🧠 Mejor modelo: models/balanced_emotion_model.h5")
        print(f"📊 Análisis completo: results/balanced_*")
        print(f"➡️  Prueba la interfaz: python 4_test_interface.py")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
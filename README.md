#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT 0: VERIFICAR INSTALACIÓN Y DEPENDENCIAS
Ejecuta este script PRIMERO para asegurarte que todo está bien instalado
"""

import sys
import subprocess
import pkg_resources

def install_package(package):
    """Instala un paquete usando pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install_requirements():
    """Verifica e instala todas las dependencias necesarias"""
    
    # Lista de paquetes requeridos
    required_packages = [
        'tensorflow>=2.8.0',
        'librosa>=0.9.0',
        'matplotlib>=3.5.0',
        'numpy>=1.21.0',
        'pandas>=1.4.0',
        'scikit-learn>=1.0.0',
        'seaborn>=0.11.0',
        'soundfile>=0.10.0',
        'pillow>=8.0.0'
    ]
    
    print("🔍 Verificando dependencias...")
    
    for package in required_packages:
        package_name = package.split('>=')[0]
        try:
            pkg_resources.get_distribution(package_name)
            print(f"✅ {package_name} - INSTALADO")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package_name} - NO ENCONTRADO")
            print(f"📦 Instalando {package_name}...")
            try:
                install_package(package_name)
                print(f"✅ {package_name} - INSTALADO EXITOSAMENTE")
            except Exception as e:
                print(f"❌ Error instalando {package_name}: {e}")

def test_imports():
    """Prueba que se puedan importar todas las librerías"""
    print("\n🧪 Probando importaciones...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"❌ Error importando TensorFlow: {e}")
        
    try:
        import librosa
        print(f"✅ Librosa {librosa.__version__}")
    except ImportError as e:
        print(f"❌ Error importando Librosa: {e}")
        
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib")
    except ImportError as e:
        print(f"❌ Error importando Matplotlib: {e}")
        
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ Error importando NumPy: {e}")
        
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Error importando Pandas: {e}")

def create_project_structure():
    """Crea la estructura de carpetas del proyecto"""
    import os
    
    print("\n📁 Creando estructura del proyecto...")
    
    directories = [
        'data',
        'data/raw_audio',
        'data/spectrograms',
        'data/spectrograms/train',
        'data/spectrograms/test',
        'models',
        'results',
        'scripts'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Carpeta creada: {directory}")

def main():
    """Función principal"""
    print("🚀 CONFIGURACIÓN DEL PROYECTO DE DETECCIÓN DE EMOCIONES")
    print("=" * 60)
    
    # Verificar versión de Python
    print(f"🐍 Python version: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("❌ ERROR: Necesitas Python 3.8 o superior")
        return
    
    # Verificar e instalar dependencias
    check_and_install_requirements()
    
    # Probar importaciones
    test_imports()
    
    # Crear estructura del proyecto
    create_project_structure()
    
    print("\n🎉 ¡CONFIGURACIÓN COMPLETADA!")
    print("Ahora puedes ejecutar los siguientes scripts en orden:")
    print("1. python 1_download_data.py")
    print("2. python 2_preprocess_audio.py")
    print("3. python 3_train_model.py")
    print("4. python 4_test_interface.py")

if __name__ == "__main__":
    main()

# 🤖 Chaski UMSS – Asistente Académico Inteligente

Este proyecto implementa un chatbot académico entrenado con una red neuronal recurrente (RNN), diseñado para responder preguntas frecuentes sobre la UMSS y la carrera de Ingeniería de Sistemas. Utiliza TensorFlow/Keras para el modelo, Pickle para el manejo de datos, y Gradio para desplegar una interfaz web amigable.

---

## 📌 Requisitos del sistema

- Python 3.8 o superior  
- pip (gestor de paquetes de Python)  
- RAM mínima: 4 GB (recomendado 8 GB)  
- Sistema operativo: Windows, Linux o macOS

---

## ⚙️ Instalación y ejecución del proyecto

### 1. Crear un entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
```
### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```
### 3. Ejecutar cada script del proyecto
```bash
python3 preprocessing.py
python3 training.py
python3 chatting.py
```

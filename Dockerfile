# Usamos la base oficial de RunPod con CUDA 12.1 (Compatible con SDXL y Wan)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Evitar preguntas durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalamos dependencias del sistema básico
RUN apt-get update && apt-get install -y \
    git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiamos los requerimientos e instalamos
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copiamos el código del worker
COPY handler.py /handler.py

# Comando de arranque
CMD ["python", "-u", "/handler.py"]
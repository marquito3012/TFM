FROM rocm/pytorch:latest

# 1. Configuración de zona horaria y sistema
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python-is-python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Definimos el directorio de trabajo dentro del contenedor
WORKDIR /compartida

# 3. Preparación del entorno Python
RUN pip install --no-cache-dir --upgrade pip

# 4. Instalación de dependencias (Estrategia de Cache)
COPY requirements.txt .
# Instalamos las librerías
RUN pip install --no-cache-dir -r requirements.txt

# 5. Exponemos el puerto de Jupyter Lab
EXPOSE 8888

# 6. Configuración de volumen para compartir datos entre el host y el contenedor
VOLUME ["/compartida"]

# Comando para iniciar el entorno de experimentación
CMD ["/bin/bash"]
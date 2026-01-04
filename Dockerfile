# Usamos una imagen que ya tiene Python y Rust instalados
FROM python:3.10-slim-bullseye

# Instalar dependencias del sistema necesarias para compilar vtracer
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

# Configurar directorio de trabajo
WORKDIR /app

# Copiar archivos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Comando para iniciar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

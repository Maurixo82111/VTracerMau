from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import vtracer
import os
import shutil
import uuid

app = FastAPI()

# Configuración de CORS total para evitar bloqueos en Shopify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/vectorize")
async def vectorize_image(file: UploadFile = File(...)):
    # Generar nombres únicos para evitar conflictos entre usuarios
    job_id = str(uuid.uuid4())
    input_path = f"temp_{job_id}_{file.filename}"
    output_path = f"{input_path}.svg"
    
    try:
        # 1. Guardar archivo subido
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Vectorización Pro (Ajustes de alta fidelidad)
        vtracer.convert_image_to_svg(
            input_path, 
            output_path,
            mode='spline',       # Curvas suaves (estilo vectorizer.ai)
            iteration_count=20,  # Más iteraciones = más precisión (Máximo recomendado: 50)
            cutoff_size=1,       # Detecta hasta el detalle más mínimo
            hierarchical='cut',  # Crea capas limpias
            filter_speckle=2,    # Limpia ruido visual
            corner_threshold=60  # Mejora las esquinas
        )

        # 3. Leer resultado
        with open(output_path, "r") as f:
            svg_data = f.read()
        
        return {"svg": svg_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 4. Limpieza garantizada de archivos temporales
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

@app.get("/")
def health_check():
    return {"status": "servidor activo y listo"}

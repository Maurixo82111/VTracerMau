from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import vtracer
import os
import shutil
import uuid

app = FastAPI()

# Configuración de CORS Robusta
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/vectorize")
async def vectorize_image(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    input_path = f"in_{job_id}_{file.filename}"
    output_path = f"out_{job_id}.svg"
    
    try:
        # Guardar imagen temporal
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parámetros de Alta Fidelidad (Estilo Premium)
        vtracer.convert_image_to_svg(
            input_path, 
            output_path,
            mode='spline',       # Curvas suaves
            iteration_count=35,  # Precisión alta
            cutoff_size=1,       # No ignora detalles pequeños
            hierarchical='cut',  # Capas de color limpias
            filter_speckle=2,    # Elimina puntos de ruido
            corner_threshold=60  # Mantiene bordes definidos
        )

        with open(output_path, "r") as f:
            svg_data = f.read()
        
        return {"svg": svg_data}

    except Exception as e:
        print(f"Error interno: {e}")
        raise HTTPException(status_code=500, detail="Error al procesar la imagen")
    
    finally:
        # Limpieza de archivos
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)

@app.get("/")
def health_check():
    return {"status": "online"}

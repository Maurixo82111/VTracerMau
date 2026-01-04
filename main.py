from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
# Importación directa del motor
try:
    from vtracer import convert_image_to_svg as v_convert
except ImportError:
    import vtracer
    v_convert = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    import vtracer
    return {
        "status": "ready",
        "methods": dir(vtracer),
        "module_path": str(vtracer.__file__)
    }

@app.post("/vectorize")
async def vectorize_image(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    input_path = f"in_{job_id}_{file.filename}"
    output_path = f"out_{job_id}.svg"
    
    try:
        # 1. Guardar imagen
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Ejecutar vectorización usando el detector de método
        import vtracer
        
        # Intentamos detectar cuál de estos nombres es el que funciona en tu instancia
        method = None
        for m in ['convert_image_to_svg', 'convert', 'vtracer']:
            if hasattr(vtracer, m):
                method = getattr(vtracer, m)
                break
        
        if method:
            method(
                input_path, 
                output_path,
                mode='spline',
                iteration_count=30,
                cutoff_size=1,
                hierarchical='cut'
            )
        else:
            raise Exception("No se pudo inicializar el motor de vtracer")

        # 3. Leer y responder
        if not os.path.exists(output_path):
            raise Exception("El motor no generó el archivo de salida")

        with open(output_path, "r") as f:
            svg_data = f.read()
        
        return {"svg": svg_data}

    except Exception as e:
        print(f"Error Crítico: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Limpieza absoluta
        for path in [input_path, output_path]:
            if os.path.exists(path):
                os.remove(path)

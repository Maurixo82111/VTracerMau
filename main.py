from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import vtracer
import io
import os
import uuid

app = FastAPI()

# Permitir conexiones desde Shopify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ready", "engine": "vtracer-rust-native"}

@app.post("/vectorize")
async def vectorize(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    input_path = f"input_{job_id}_{file.filename}"
    output_path = f"output_{job_id}.svg"

    try:
        # Guardar imagen temporal
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # EJECUCIÓN DEL MOTOR RUST (vtracer original)
        vtracer.convert_image_to_svg(
            input_path,
            output_path,
            mode='spline',        # Curvas suaves estilo Bézier
            iteration_count=10,   # Precisión de trazado
            cutoff_size=4,        # Eliminar ruido
            hierarchical='cut'    # Manejo de capas de color
        )

        with open(output_path, "rb") as f:
            svg_content = f.read()

        return StreamingResponse(
            io.BytesIO(svg_content),
            media_type="image/svg+xml"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Limpieza de archivos temporales
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import vtracer
import os
import shutil
import uuid

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
    return {"status": "ready"}

@app.post("/vectorize")
async def vectorize_image(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    input_path = f"in_{job_id}_{file.filename}"
    output_path = f"out_{job_id}.svg"
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # CORRECCIÓN AQUÍ: Usamos vtracer.convert
        vtracer.convert(
            input_path, 
            output_path,
            mode='spline',
            iteration_count=30,
            cutoff_size=1,
            hierarchical='cut'
        )

        with open(output_path, "r") as f:
            svg_data = f.read()
        
        return {"svg": svg_data}
    except Exception as e:
        print(f"Error detectado: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)

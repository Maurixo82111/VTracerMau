from fastapi import FastAPI, UploadFile, File
import vtracer
import os

app = FastAPI()

@app.post("/vectorize")
async def vectorize_image(file: UploadFile = File(...)):
    # Guardar temporalmente la imagen subida
    input_path = f"temp_{file.filename}"
    output_path = "output.svg"
    
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Ejecutar la vectorización
    # mode: 'spline' para curvas suaves, 'polygon' para formas rectas
    vtracer.convert_image_to_svg(
        input_path, 
        output_path,
        mode='spline', 
        iteration_count=10,
        cutoff_size=4
    )

    # Leer el resultado y limpiar archivos
    with open(output_path, "r") as f:
        svg_data = f.read()
    
    os.remove(input_path)
    os.remove(output_path)

    return {"svg": svg_data}

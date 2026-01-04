from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uuid

app = FastAPI()

# Permitir conexiones desde tu Shopify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_svg_path(points):
    """Convierte una lista de puntos en un path de SVG (formato Polígono)."""
    if not points:
        return ""
    d = f"M {points[0][0]},{points[0][1]} "
    for i in range(1, len(points)):
        d += f"L {points[i][0]},{points[i][1]} "
    d += "Z"
    return d

def custom_vectorize(image_bytes):
    """
    Lógica de vectorización manual:
    1. Reduce colores.
    2. Encuentra contornos.
    3. Genera SVG.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Redimensionar si es muy grande para no colapsar el server gratuito
    img.thumbnail((400, 400))
    width, height = img.size
    
    # Reducir paleta a 16 colores para simplificar (Cuantización)
    img_small = img.quantize(colors=16).convert("RGB")
    pixels = img_small.load()
    
    # Diccionario para agrupar puntos por color
    color_paths = {}

    # Escaneo básico: Agrupamos píxeles por color
    # Nota: Este es un algoritmo de 'puntos a polígonos' simplificado
    for y in range(0, height, 2): # Saltamos de 2 en 2 para velocidad
        for x in range(0, width, 2):
            r, g, b = pixels[x, y]
            color_hex = '#{:02x}{:02x}{:02x}'.format(r, g, b)
            
            if color_hex not in color_paths:
                color_paths[color_hex] = []
            
            # Guardamos el rectángulo del píxel
            color_paths[color_hex].append((x, y))

    # Construcción del SVG
    svg_header = f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
    svg_body = ""
    
    for color, points in color_paths.items():
        # Para cada color, creamos rectángulos pequeños (esto es vectorización básica)
        for p in points:
            svg_body += f'<rect x="{p[0]}" y="{p[1]}" width="2.1" height="2.1" fill="{color}" stroke="{color}" />'
    
    svg_footer = "</svg>"
    return svg_header + svg_body + svg_footer

@app.get("/health")
def health():
    return {"status": "ready", "engine": "custom_python_v1"}

@app.post("/vectorize")
async def vectorize(file: UploadFile = File(...)):
    try:
        content = await file.read()
        svg_output = custom_vectorize(content)
        return {"svg": svg_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

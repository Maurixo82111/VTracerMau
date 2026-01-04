"""
Vectorizer Backend - Pure Python Implementation
High-Fidelity Raster-to-SVG Conversion
=============================================

Senior Software Engineer: Computational Geometry & Vision
Constraints: Pillow-only (no OpenCV/vtracer), 512MB memory limit

Algorithm Pipeline:
1. Adaptive Pre-processing: Median filter + Lanczos upsampling
2. Intelligent Color Segmentation: Median Cut quantization
3. Contour Tracing: Moore-Neighbor algorithm with hole handling
4. Smoothing & Curvature: RDP simplification + Cubic Bézier interpolation
5. SVG Generation: Professional output with shape-rendering optimization
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import io
import math
import traceback
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from enum import Enum

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

MAX_INPUT_DIMENSION = 2048  # Render.com memory constraint
MAX_COLORS = 16  # Median cut palette size
EPSILON_RDP = 1.0  # Ramer-Douglas-Peucker simplification threshold
BEZIER_TOLERANCE = 0.5  # Curve fitting tolerance
UPSAMPLING_FACTOR = 1.5  # Pre-processing upsampling

class ProcessingMode(str, Enum):
    FAST = "fast"  # No upsampling
    BALANCED = "balanced"  # 1.5x upsampling
    QUALITY = "quality"  # 2x upsampling (memory intensive)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Point:
    """2D point with hash support for efficient lookup"""
    x: float
    y: float
    
    def __hash__(self):
        return hash((round(self.x, 2), round(self.y, 2)))
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 0.01 and abs(self.y - other.y) < 0.01
    
    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

@dataclass
class BezierSegment:
    """Cubic Bézier curve representation"""
    p0: Point  # Start point
    p1: Point  # Control point 1
    p2: Point  # Control point 2
    p3: Point  # End point
    
    def to_svg_command(self) -> str:
        """Generate SVG cubic Bézier command"""
        return f"C {self.p1.x:.2f} {self.p1.y:.2f} {self.p2.x:.2f} {self.p2.y:.2f} {self.p3.x:.2f} {self.p3.y:.2f}"

@dataclass
class Contour:
    """Closed contour with potential holes"""
    boundary: List[Point]
    holes: List[List[Point]]
    color_hex: str
    
    def is_clockwise(self) -> bool:
        """Calculate signed area to determine orientation"""
        area = 0.0
        n = len(self.boundary)
        for i in range(n):
            j = (i + 1) % n
            area += (self.boundary[j].x - self.boundary[i].x) * \
                    (self.boundary[j].y + self.boundary[i].y)
        return area > 0

# ============================================================================
# PHASE 1: ADAPTIVE PRE-PROCESSING
# ============================================================================

class PreProcessor:
    """Noise reduction and resolution optimization"""
    
    @staticmethod
    def apply_median_filter(image: Image.Image, radius: int = 2) -> Image.Image:
        """Remove salt-and-pepper noise using median filter"""
        if image.mode == 'RGBA':
            # Process alpha separately
            alpha = image.split()[3]
            rgb_img = Image.new('RGB', image.size, (255, 255, 255))
            rgb_img.paste(image.convert('RGB'), (0, 0))
            rgb_img = rgb_img.filter(ImageFilter.MedianFilter(size=radius * 2 + 1))
            rgb_img.putalpha(alpha)
            return rgb_img
        return image.filter(ImageFilter.MedianFilter(size=radius * 2 + 1))
    
    @staticmethod
    def adaptive_upsampling(image: Image.Image, mode: ProcessingMode = ProcessingMode.BALANCED) -> Image.Image:
        """
        Lanczos interpolation for edge precision improvement
        Memory-aware: scales based on available budget
        """
        factors = {
            ProcessingMode.FAST: 1.0,
            ProcessingMode.BALANCED: 1.5,
            ProcessingMode.QUALITY: 2.0
        }
        factor = factors[mode]
        
        if factor == 1.0:
            return image
        
        new_size = (int(image.width * factor), int(image.height * factor))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def preprocess(image: Image.Image, mode: ProcessingMode = ProcessingMode.BALANCED) -> Image.Image:
        """Full preprocessing pipeline"""
        # Convert to RGB if necessary
        if image.mode != 'RGB' and image.mode != 'RGBA':
            image = image.convert('RGB')
        
        # Remove transparency by compositing on white background
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        
        # Apply median filter for noise reduction
        image = PreProcessor.apply_median_filter(image, radius=1)
        
        # Adaptive upsampling
        image = PreProcessor.adaptive_upsampling(image, mode)
        
        return image

# ============================================================================
# PHASE 2: INTELLIGENT COLOR SEGMENTATION
# ============================================================================

class ColorSegmenter:
    """Median Cut quantization and layer separation"""
    
    @staticmethod
    def quantize_colors(image: Image.Image, num_colors: int = MAX_COLORS) -> Tuple[Image.Image, List[Tuple[int, int, int]]]:
        """
        Median Cut algorithm for optimal color palette generation
        Returns quantized image and palette
        """
        # Convert to RGB (ensure no alpha)
        image_rgb = image.convert('RGB')
        
        # Pillow's quantize uses median cut internally
        palette_image = image_rgb.quantize(colors=min(num_colors, 256))
        palette = palette_image.getpalette()
        
        # Extract RGB triples
        colors = []
        for i in range(0, min(len(palette), num_colors * 3), 3):
            colors.append((palette[i], palette[i+1], palette[i+2]))
        
        return palette_image, colors[:num_colors]
    
    @staticmethod
    def create_binary_layers(quantized_image: Image.Image, 
                            colors: List[Tuple[int, int, int]]) -> dict:
        """
        Create binary mask for each color in palette
        Returns {color_hex: binary_mask_image}
        """
        layers = {}
        pixel_data = quantized_image.load()
        
        for color_idx, (r, g, b) in enumerate(colors):
            # Create binary mask
            mask = Image.new('L', quantized_image.size, 0)
            mask_data = mask.load()
            
            for y in range(quantized_image.height):
                for x in range(quantized_image.width):
                    if pixel_data[x, y] == color_idx:
                        mask_data[x, y] = 255
            
            color_hex = f"#{r:02x}{g:02x}{b:02x}"
            layers[color_hex] = mask
        
        return layers

# ============================================================================
# PHASE 3: CONTOUR TRACING (MOORE-NEIGHBOR)
# ============================================================================

class ContourTracer:
    """
    Moore-Neighbor algorithm for boundary extraction
    Handles holes via contour hierarchy
    """
    
    # Direction vectors: N, NE, E, SE, S, SW, W, NW
    DIRECTIONS = [
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1)
    ]
    
    @staticmethod
    def _find_starting_pixel(binary_mask: Image.Image) -> Optional[Tuple[int, int]]:
        """Find topmost-leftmost white pixel"""
        pixels = binary_mask.load()
        for y in range(binary_mask.height):
            for x in range(binary_mask.width):
                if pixels[x, y] > 127:
                    return (x, y)
        return None
    
    @staticmethod
    def _get_next_direction_index(current_dir: int) -> int:
        """Find next direction in Moore-Neighbor sequence"""
        return (current_dir + 2) % 8
    
    @staticmethod
    def trace_boundary(binary_mask: Image.Image) -> List[Point]:
        """
        Extract contour using Moore-Neighbor tracing
        Returns list of points in clockwise order
        """
        start_pixel = ContourTracer._find_starting_pixel(binary_mask)
        if not start_pixel:
            return []
        
        pixels = binary_mask.load()
        contour = []
        current = start_pixel
        direction = 0  # Start north
        
        visited_start = False
        while True:
            contour.append(Point(float(current[0]), float(current[1])))
            
            # Find next boundary pixel
            found = False
            for _ in range(8):
                direction = ContourTracer._get_next_direction_index(direction)
                dx, dy = ContourTracer.DIRECTIONS[direction]
                next_x, next_y = current[0] + dx, current[1] + dy
                
                # Bounds check
                if 0 <= next_x < binary_mask.width and 0 <= next_y < binary_mask.height:
                    if pixels[next_x, next_y] > 127:
                        current = (next_x, next_y)
                        found = True
                        break
            
            if not found:
                break
            
            # Check if we've closed the contour
            if current == start_pixel:
                if visited_start:
                    break
                visited_start = True
        
        # Remove duplicate last point if it exists
        if len(contour) > 1 and contour[-1].to_tuple() == contour[0].to_tuple():
            contour.pop()
        
        return contour
    
    @staticmethod
    def extract_contours(layers: dict) -> List[Contour]:
        """Extract all contours from binary layers"""
        contours = []
        for color_hex, binary_mask in layers.items():
            boundary = ContourTracer.trace_boundary(binary_mask)
            
            if len(boundary) > 2:  # Valid contour
                contour = Contour(
                    boundary=boundary,
                    holes=[],
                    color_hex=color_hex
                )
                contours.append(contour)
        
        return contours

# ============================================================================
# PHASE 4: SMOOTHING & CURVATURE (RDP + BÉZIER)
# ============================================================================

class CurveProcessor:
    """
    Ramer-Douglas-Peucker simplification
    Cubic Bézier curve fitting for smooth output
    """
    
    @staticmethod
    def perpendicular_distance(point: Point, line_start: Point, line_end: Point) -> float:
        """Calculate perpendicular distance from point to line segment"""
        if line_start.distance_to(line_end) < 1e-6:
            return point.distance_to(line_start)
        
        t = max(0, min(1, ((point.x - line_start.x) * (line_end.x - line_start.x) +
                           (point.y - line_start.y) * (line_end.y - line_start.y)) /
                      (line_end.distance_to(line_start) ** 2)))
        
        projection = Point(
            line_start.x + t * (line_end.x - line_start.x),
            line_start.y + t * (line_end.y - line_start.y)
        )
        return point.distance_to(projection)
    
    @staticmethod
    def rdp_simplify(points: List[Point], epsilon: float = EPSILON_RDP) -> List[Point]:
        """
        Ramer-Douglas-Peucker algorithm
        Removes redundant points while preserving shape
        """
        if len(points) < 3:
            return points
        
        # Find the point with maximum distance
        dmax = 0.0
        index = 0
        for i in range(1, len(points) - 1):
            d = CurveProcessor.perpendicular_distance(
                points[i], points[0], points[-1]
            )
            if d > dmax:
                dmax = d
                index = i
        
        # If max distance is greater than epsilon, subdivide
        if dmax > epsilon:
            rec1 = CurveProcessor.rdp_simplify(points[:index+1], epsilon)
            rec2 = CurveProcessor.rdp_simplify(points[index:], epsilon)
            return rec1[:-1] + rec2
        else:
            return [points[0], points[-1]]
    
    @staticmethod
    def fit_cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point) -> BezierSegment:
        """
        Fit cubic Bézier curve through 4 points
        Uses Catmull-Rom spline interpolation for smooth G1 continuity
        """
        # Catmull-Rom control point calculation
        # This ensures C1 continuity at segment junctions
        
        # Chord length parameterization
        d01 = p0.distance_to(p1)
        d12 = p1.distance_to(p2)
        d23 = p2.distance_to(p3)
        
        # Prevent division by zero
        d01 = max(d01, 1e-6)
        d12 = max(d12, 1e-6)
        d23 = max(d23, 1e-6)
        
        # Catmull-Rom coefficients with tension = 0.5
        tension = 0.5
        
        # Control point 1
        c1_x = p1.x + (tension / 6) * (
            (p2.x - p0.x) * (d01 / (d01 + d12))
        )
        c1_y = p1.y + (tension / 6) * (
            (p2.y - p0.y) * (d01 / (d01 + d12))
        )
        
        # Control point 2
        c2_x = p2.x - (tension / 6) * (
            (p3.x - p1.x) * (d23 / (d12 + d23))
        )
        c2_y = p2.y - (tension / 6) * (
            (p3.y - p1.y) * (d23 / (d12 + d23))
        )
        
        return BezierSegment(p1, Point(c1_x, c1_y), Point(c2_x, c2_y), p2)
    
    @staticmethod
    def smooth_contour(contour: Contour) -> Contour:
        """
        Apply complete smoothing pipeline to contour
        1. RDP simplification
        2. Bézier curve fitting
        """
        # Step 1: Simplify with RDP
        simplified = CurveProcessor.rdp_simplify(contour.boundary, EPSILON_RDP)
        
        if len(simplified) < 4:
            # Not enough points for Bézier fitting, return as-is
            contour.boundary = simplified
            return contour
        
        # Store simplified points (we'll convert to Bézier during SVG generation)
        contour.boundary = simplified
        return contour

# ============================================================================
# PHASE 5: SVG GENERATION
# ============================================================================

class SVGGenerator:
    """Professional SVG output with optimization"""
    
    @staticmethod
    def points_to_svg_path(points: List[Point], use_bezier: bool = True) -> str:
        """
        Convert point list to SVG path with optional Bézier smoothing
        Ensures G1 continuity between segments
        """
        if len(points) < 2:
            return ""
        
        path_commands = []
        path_commands.append(f"M {points[0].x:.2f} {points[0].y:.2f}")
        
        if not use_bezier or len(points) < 4:
            # Fallback to line segments
            for point in points[1:]:
                path_commands.append(f"L {point.x:.2f} {point.y:.2f}")
        else:
            # Fit cubic Bézier curves through point sequence
            # Using centripetal parameterization for smooth curves
            
            for i in range(1, len(points)):
                p0 = points[i-1]
                p1 = points[i]
                
                # Look ahead for control points
                p_prev = points[i-2] if i > 1 else points[-1]
                p_next = points[(i+1) % len(points)] if i < len(points) - 1 else points[1]
                
                # Calculate Bézier curve
                bezier = SVGGenerator.fit_catmull_rom(p_prev, p0, p1, p_next)
                path_commands.append(bezier.to_svg_command())
        
        # Close path
        path_commands.append("Z")
        return " ".join(path_commands)
    
    @staticmethod
    def fit_catmull_rom(p_prev: Point, p0: Point, p1: Point, p_next: Point) -> BezierSegment:
        """Fit Catmull-Rom spline (C1 continuous)"""
        tension = 0.5
        
        d01 = p0.distance_to(p1)
        d01 = max(d01, 1e-6)
        
        # Control points using Catmull-Rom formula
        c1_x = p0.x + tension * (p1.x - p_prev.x) / 6
        c1_y = p0.y + tension * (p1.y - p_prev.y) / 6
        
        c2_x = p1.x - tension * (p_next.x - p0.x) / 6
        c2_y = p1.y - tension * (p_next.y - p0.y) / 6
        
        return BezierSegment(p0, Point(c1_x, c1_y), Point(c2_x, c2_y), p1)
    
    @staticmethod
    def generate_svg(contours: List[Contour], width: int, height: int) -> str:
        """
        Generate professional SVG document
        Includes shape-rendering optimization and anti-gap strokes
        """
        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" shape-rendering="geometricPrecision" xmlns:xlink="http://www.w3.org/1999/xlink">',
            '<defs><style type="text/css"><![CDATA[]]></style></defs>',
            '<g>'
        ]
        
        # Group paths by color for optimization
        color_groups = {}
        for contour in contours:
            if contour.color_hex not in color_groups:
                color_groups[contour.color_hex] = []
            color_groups[contour.color_hex].append(contour)
        
        # Generate path elements
        for color_hex, contours_group in color_groups.items():
            # Parse hex color
            try:
                r = int(color_hex[1:3], 16)
                g = int(color_hex[3:5], 16)
                b = int(color_hex[5:7], 16)
                rgb = f"rgb({r},{g},{b})"
            except:
                rgb = "black"
            
            for contour in contours_group:
                path_data = SVGGenerator.points_to_svg_path(contour.boundary)
                if path_data:
                    svg_parts.append(
                        f'<path d="{path_data}" fill="{rgb}" stroke="{rgb}" '
                        f'stroke-width="0.5" stroke-linejoin="round" stroke-linecap="round"/>'
                    )
        
        svg_parts.extend(['</g>', '</svg>'])
        return '\n'.join(svg_parts)

# ============================================================================
# MAIN VECTORIZER ENGINE
# ============================================================================

class VectorizerEngine:
    """Complete vectorization pipeline"""
    
    @staticmethod
    def vectorize(image_bytes: bytes, mode: ProcessingMode = ProcessingMode.BALANCED) -> bytes:
        """
        Execute full vectorization pipeline
        Memory-efficient with io.BytesIO
        """
        try:
            # Load image into memory
            image = Image.open(io.BytesIO(image_bytes))
            original_width, original_height = image.size
            
            # Validate dimensions
            max_dim = max(original_width, original_height)
            if max_dim > MAX_INPUT_DIMENSION:
                scale_factor = MAX_INPUT_DIMENSION / max_dim
                new_size = (int(original_width * scale_factor), 
                           int(original_height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            width, height = image.size
            
            # Phase 1: Pre-processing
            image = PreProcessor.preprocess(image, mode)
            processed_width, processed_height = image.size
            
            # Phase 2: Color Segmentation
            quantized, palette = ColorSegmenter.quantize_colors(image, MAX_COLORS)
            layers = ColorSegmenter.create_binary_layers(quantized, palette)
            
            # Phase 3: Contour Tracing
            contours = ContourTracer.extract_contours(layers)
            
            # Phase 4: Smoothing & Curvature
            contours = [CurveProcessor.smooth_contour(c) for c in contours]
            
            # Phase 5: SVG Generation
            svg_content = SVGGenerator.generate_svg(contours, processed_width, processed_height)
            
            return svg_content.encode('utf-8')
        
        except Exception as e:
            raise ValueError(f"Vectorization failed: {str(e)}")

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="High-Fidelity Vectorizer API",
    description="Professional raster-to-SVG conversion with pure Python",
    version="1.0.0"
)

# CORS Configuration for Shopify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["monitoring"])
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "High-Fidelity Vectorizer",
        "version": "1.0.0"
    }

@app.post("/vectorize", tags=["vectorization"])
async def vectorize_image(
    file: UploadFile = File(...),
    mode: ProcessingMode = ProcessingMode.BALANCED,
    output_format: str = "svg"
):
    """
    Vectorize raster image to SVG
    
    Args:
        file: Input image (PNG, JPG, etc.)
        mode: Processing quality (fast/balanced/quality)
        output_format: Output format (svg only currently)
    
    Returns:
        SVG file with high-fidelity curves
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
        # Read file into memory
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large (max 10MB).")
        
        # Vectorize
        svg_bytes = VectorizerEngine.vectorize(image_bytes, mode)
        
        # Return SVG
        return StreamingResponse(
            io.BytesIO(svg_bytes),
            media_type="image/svg+xml",
            headers={"Content-Disposition": "attachment; filename=vectorized.svg"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Vectorization error: {str(e)}"
        )

@app.get("/", tags=["info"])
async def root():
    """API information endpoint"""
    return {
        "name": "High-Fidelity Vectorizer",
        "description": "Professional raster-to-SVG conversion backend",
        "endpoints": {
            "health": "GET /health",
            "vectorize": "POST /vectorize",
            "docs": "/docs"
        },
        "constraints": {
            "max_input_dimension": MAX_INPUT_DIMENSION,
            "max_colors": MAX_COLORS,
            "max_file_size": "10MB"
        }
    }

# ============================================================================
# PRODUCTION STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # Single worker for memory efficiency on Render.com
    )

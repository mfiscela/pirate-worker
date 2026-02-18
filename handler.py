import runpod
import torch
import os
from diffusers import StableDiffusionXLPipeline
from diffusers import (
    EulerAncestralDiscreteScheduler, 
    DPMSolverMultistepScheduler, 
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler
)

# --- 1. CONFIGURACIÓN GLOBAL ---
# Aquí guardamos el modelo cargado para no recargarlo si no cambia
current_pipeline = None
current_model_path = None

# Mapeo de nombres de samplers (Interfaz) -> Clases de Diffusers
SCHEDULER_MAP = {
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "euler": EulerDiscreteScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
    "heun": HeunDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
}

def load_pipeline(model_name):
    global current_pipeline, current_model_path
    
    # Ruta al Network Volume
    full_path = f"/network-volume/models/checkpoints/{model_name}"
    
    # Si el modelo ya está cargado, lo devolvemos directo (0 segundos)
    if current_pipeline is not None and current_model_path == full_path:
        print(f"⚡ Usando modelo en caché: {model_name}")
        return current_pipeline

    print(f"📦 Cargando NUEVO modelo: {model_name}...")
    
    # Limpiar memoria anterior
    if current_pipeline is not None:
        del current_pipeline
        torch.cuda.empty_cache()

    # Cargar pipeline de SDXL
    pipe = StableDiffusionXLPipeline.from_single_file(
        full_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # Optimización clave para 1536x1536
    pipe.enable_xformers_memory_efficient_attention()
    
    current_pipeline = pipe
    current_model_path = full_path
    return pipe

def handler(job):
    job_input = job['input']
    
    try:
        # --- PARÁMETROS OBLIGATORIOS ---
        model_name = job_input.get("model_name") # Ej: CyberRealisticPony.safetensors
        if not model_name:
            return {"status": "error", "message": "Falta 'model_name'"}

        # 1. Cargar Modelo
        pipe = load_pipeline(model_name)

        # 2. Configurar Sampler
        sampler_name = job_input.get("sampler", "euler_ancestral")
        scheduler_class = SCHEDULER_MAP.get(sampler_name, EulerAncestralDiscreteScheduler)
        pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)

        # --- PREPARACIÓN DE INFERENCIA ---
        prompt = job_input.get("positive_prompt", "")
        neg_prompt = job_input.get("negative_prompt", "")
        seed = job_input.get("seed", 42)
        steps = job_input.get("quality_steps", 30)
        cfg = job_input.get("cfg_scale", 7.0)
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        batch_size = job_input.get("batch_size", 1)
        # clip_skip = job_input.get("clip_skip", 2) # Implementar lógica de capas si es necesario

        generator = torch.Generator("cuda").manual_seed(seed)

        print(f"🎨 Generando con: {model_name} | {width}x{height} | Steps: {steps}")

        # 3. GENERAR IMAGEN
        images = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            num_images_per_prompt=batch_size,
            generator=generator
        ).images

        # 4. RESULTADO (Por ahora simulamos la URL)
        # Aquí añadiremos luego el código para subir a S3
        return {
            "status": "success", 
            "message": "Generación completada",
            "meta": {
                "model": model_name,
                "count": len(images)
            }
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
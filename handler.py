import runpod
import torch
import os
import requests
import io
from diffusers import (
    StableDiffusionXLPipeline, 
    EulerAncestralDiscreteScheduler, 
    DPMSolverMultistepScheduler, 
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler
)

# --- 1. CONFIGURACIÓN DEL ENTORNO ---
# Estas variables deben configurarse en RunPod > Endpoint > Environment Variables
# Si no están, usará los valores por defecto (útil para pruebas locales)
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "https://tu-dominio.com/webhook/receive-image")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "cambia_esto_por_tu_clave_segura")

# Ruta donde montaste el Network Volume en RunPod
MODELS_DIR = "/network-volume/models/checkpoints"

# --- 2. GESTIÓN DE ESTADO GLOBAL (CACHÉ) ---
current_pipeline = None
current_model_path = None

# Mapeo de nombres de tu Interfaz -> Clases de Diffusers
SCHEDULER_MAP = {
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "euler": EulerDiscreteScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
    "heun": HeunDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "dpm_2_ancestral": KDPM2AncestralDiscreteScheduler,
}

def load_pipeline(model_name):
    """
    Carga el modelo solo si es diferente al que ya está en memoria.
    """
    global current_pipeline, current_model_path
    
    full_path = f"{MODELS_DIR}/{model_name}"
    
    # Si el modelo ya está cargado, retornamos la instancia en caché (0 segundos)
    if current_pipeline is not None and current_model_path == full_path:
        print(f"♻️ Usando modelo en caché: {model_name}")
        return current_pipeline

    print(f"🔄 Cargando NUEVO modelo: {model_name}...")
    
    # Limpiamos la VRAM del modelo anterior
    if current_pipeline is not None:
        del current_pipeline
        torch.cuda.empty_cache()

    # Carga optimizada para SDXL
    try:
        pipe = StableDiffusionXLPipeline.from_single_file(
            full_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")

        # Optimizaciones de memoria y velocidad
        pipe.enable_xformers_memory_efficient_attention()
        
        current_pipeline = pipe
        current_model_path = full_path
        print("✅ Modelo cargado correctamente.")
        return pipe
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo {model_name}: {e}")

def handler(job):
    """
    Función principal que ejecuta RunPod.
    """
    job_input = job['input']
    job_id = job['id'] # ID único del trabajo

    try:
        # --- A. VALIDACIÓN DE PARÁMETROS ---
        model_name = job_input.get("model_name")
        if not model_name:
            return {"status": "error", "message": "Falta el parámetro 'model_name'"}

        # --- B. PREPARACIÓN DEL PIPELINE ---
        pipe = load_pipeline(model_name)

        # Configurar Scheduler
        sampler_name = job_input.get("sampler", "euler_ancestral")
        scheduler_class = SCHEDULER_MAP.get(sampler_name, EulerAncestralDiscreteScheduler)
        
        # A veces es necesario cargar la config del scheduler actual
        if hasattr(pipe.scheduler, 'config'):
            pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)

        # --- C. EXTRACCIÓN DE DATOS DE GENERACIÓN ---
        prompt = job_input.get("positive_prompt", "")
        neg_prompt = job_input.get("negative_prompt", "")
        seed = job_input.get("seed", 42)
        steps = job_input.get("quality_steps", 30)
        cfg = job_input.get("cfg_scale", 7.0)
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        batch_size = job_input.get("batch_size", 1)
        # clip_skip = job_input.get("clip_skip", 2) # SDXL suele manejar esto internamente o via layers

        generator = torch.Generator("cuda").manual_seed(seed)

        print(f"🎨 Generando Job {job_id} | Modelo: {model_name} | {width}x{height}")

        # --- D. INFERENCIA (LA MAGIA) ---
        with torch.inference_mode():
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

        # --- E. ENVÍO AL VPS ---
        generated_image = images[0] # Tomamos la primera imagen (si batch > 1, habría que iterar)
        
        # Convertir imagen a bytes en memoria
        img_byte_arr = io.BytesIO()
        generated_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        print(f"📡 Enviando resultado al VPS: {WEBHOOK_URL}")

        files = {'file': (f'{job_id}.png', img_byte_arr, 'image/png')}
        data = {'job_id': job_id}
        headers = {'x-api-key': WEBHOOK_SECRET}

        response = requests.post(WEBHOOK_URL, files=files, data=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            print("✅ Imagen entregada con éxito.")
            return {"status": "success", "job_id": job_id, "delivery": "delivered"}
        else:
            print(f"⚠️ Error del VPS: {response.status_code} - {response.text}")
            return {"status": "warning", "message": f"Imagen generada pero falló envío: {response.status_code}"}

    except Exception as e:
        print(f"❌ Error crítico: {str(e)}")
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
import qi
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.fft
import time
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter

# ===========================
# Configurações do Benchmark
# ===========================
BATCH_SIZES = [1, 5, 10, 20, 50]  # Diferentes tamanhos de lote
TEST_RESOLUTIONS = [(320, 240), (640, 480), (1280, 960)]  # Diferentes resoluções

# ===========================
# Conexão com o NAO
# ===========================
ip = "172.15.1.80"
port = 9559

app = qi.Application(["CameraBenchmark", "--qi-url=tcp://{}:{}".format(ip, port)])
app.start()
session = app.session
video_service = session.service("ALVideoDevice")

# ===========================
# Captura de frame
# ===========================
def get_camera_frame(camera_name="top"):
    camera_index = 0 if camera_name == "top" else 1
    resolution = 2  # resolução máxima
    color_space = 11  # RGB
    fps = 15
    name_id = None
    try:
        name_id = video_service.subscribeCamera("python_client", camera_index, resolution, color_space, fps)
        time.sleep(0.3)
        image = video_service.getImageRemote(name_id)
        video_service.unsubscribe(name_id)
        if image is None or len(image) < 7:
            return None
        width, height, channels = image[0], image[1], image[2]
        array = np.frombuffer(image[6], dtype=np.uint8).reshape((height, width, channels))
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        return array
    except Exception as e:
        print("Erro ao capturar frame:", e)
        if name_id is not None:
            try: 
                video_service.unsubscribe(name_id)
            except: 
                pass
        return None

def resize_frame(frame, target_size):
    """Redimensiona frame para tamanho específico"""
    return cv2.resize(frame, target_size)

# ===========================
# Conversão para tensor PyTorch
# ===========================
def to_tensor(img, device='cpu'):
    if len(img.shape) == 4:  # Batch
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    else:  # Single image
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)

def to_image(tensor):
    if tensor.dim() == 4 and tensor.shape[0] > 1:  # Batch
        # Retorna apenas a primeira imagem do batch
        tensor = tensor[0]
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

# ===========================
# Filtros CPU Tradicionais
# ===========================
def cpu_fft_filter(img):
    """FFT-based high-pass filter"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f_transform = fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    
    # High-pass filter
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0
    
    f_shift = f_shift * mask
    f_ishift = np.fft.ifftshift(f_shift)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)

def cpu_batch_gaussian(images):
    """Processa lote de imagens na CPU"""
    results = []
    for img in images:
        results.append(cv2.GaussianBlur(img, (15, 15), 0))
    return np.array(results)

def cpu_batch_sobel(images):
    """Processa lote de imagens na CPU"""
    results = []
    for img in images:
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        results.append(np.clip(sobel, 0, 255).astype(np.uint8))
    return np.array(results)

def cpu_batch_fft(images):
    """FFT em lote na CPU"""
    results = []
    for img in images:
        results.append(cpu_fft_filter(img))
    return np.array(results)

# ===========================
# Filtros GPU Avançados
# ===========================
def gpu_fft_filter(x):
    """FFT-based high-pass filter on GPU"""
    # Converter para grayscale se necessário
    if x.shape[1] == 3:
        gray = x.mean(dim=1, keepdim=True)
    else:
        gray = x
    
    # FFT
    f_transform = torch.fft.fft2(gray.squeeze(1))
    f_shift = torch.fft.fftshift(f_transform)
    
    # High-pass mask
    b, h, w = f_shift.shape
    crow, ccol = h // 2, w // 2
    mask = torch.ones_like(f_shift, dtype=torch.float32)
    r = 30
    mask[:, crow-r:crow+r, ccol-r:ccol+r] = 0
    
    # Apply filter
    f_shift = f_shift * mask
    f_ishift = torch.fft.ifftshift(f_shift)
    img_back = torch.fft.ifft2(f_ishift)
    img_back = torch.abs(img_back)
    
    # Normalize and convert back to 3 channels
    img_back = img_back.unsqueeze(1)
    img_back = img_back / img_back.max() if img_back.max() > 0 else img_back
    return img_back.repeat(1, x.shape[1], 1, 1)

def gpu_batch_gaussian(x, kernel_size=15):
    """Gaussian blur otimizado para lotes"""
    sigma = kernel_size / 6.0
    kernel_1d = torch.exp(-0.5 * (torch.arange(kernel_size, device=x.device, dtype=torch.float32) - kernel_size // 2)**2 / sigma**2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Separable convolution
    kernel_x = kernel_1d.view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, -1, 1).repeat(x.shape[1], 1, 1, 1)
    
    x = F.conv2d(x, kernel_x, padding=(0, kernel_size//2), groups=x.shape[1])
    x = F.conv2d(x, kernel_y, padding=(kernel_size//2, 0), groups=x.shape[1])
    return x

def gpu_batch_sobel(x):
    """Sobel otimizado para lotes"""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device)
    kx = kx.unsqueeze(0).unsqueeze(0).repeat(x.shape[1], 1, 1, 1)
    ky = ky.unsqueeze(0).unsqueeze(0).repeat(x.shape[1], 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, ky, padding=1, groups=x.shape[1])
    return torch.sqrt(gx**2 + gy**2)

# ===========================
# Função de Benchmark
# ===========================
def benchmark_operation(cpu_func, gpu_func, images, batch_size=1, device='cuda'):
    """Executa benchmark de uma operação específica"""
    
    # Preparar dados
    if batch_size == 1:
        cpu_data = images
        gpu_data = [to_tensor(img, device) for img in images]
    else:
        # Criar batches
        cpu_batches = []
        gpu_batches = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            if len(batch) == batch_size:  # Só usar batches completos
                cpu_batches.append(np.array(batch))
                gpu_batches.append(to_tensor(np.array(batch), device))
        cpu_data = cpu_batches
        gpu_data = gpu_batches
    
    # Benchmark CPU
    torch.cuda.synchronize() if device == 'cuda' else None
    start_cpu = time.time()
    
    if batch_size == 1:
        for img in cpu_data:
            _ = cpu_func(img)
    else:
        for batch in cpu_data:
            _ = cpu_func(batch)
    
    end_cpu = time.time()
    cpu_time = (end_cpu - start_cpu) / len(cpu_data)
    
    # Benchmark GPU
    gpu_time = 0
    if gpu_func and torch.cuda.is_available():
        torch.cuda.synchronize()
        start_gpu = time.time()
        
        with torch.no_grad():
            for data in gpu_data:
                _ = gpu_func(data)
        
        torch.cuda.synchronize()
        end_gpu = time.time()
        gpu_time = (end_gpu - start_gpu) / len(gpu_data)
    
    return cpu_time, gpu_time

# ===========================
# Captura e preparação de dados
# ===========================
print("Capturando frame base...")
base_frame = get_camera_frame("top")
if base_frame is None:
    print("Erro: Não foi possível capturar frame!")
    app.stop()
    exit()

print(f"Frame capturado: {base_frame.shape}")

# ===========================
# Benchmark Completo
# ===========================
results = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n=== BENCHMARK COMPLETO (Device: {device}) ===")

# Testar diferentes resoluções e tamanhos de batch
for resolution in TEST_RESOLUTIONS:
    print(f"\n--- Resolução: {resolution[0]}x{resolution[1]} ---")
    
    # Redimensionar frame base
    resized_frame = resize_frame(base_frame, resolution)
    
    for batch_size in BATCH_SIZES:
        print(f"\nBatch Size: {batch_size}")
        
        # Criar conjunto de imagens (duplicar o frame base)
        images = [resized_frame.copy() for _ in range(batch_size * 3)]  # 3 batches para média
        
        # Definir operações para teste
        operations = [
            ("Gaussian", 
             lambda img: cv2.GaussianBlur(img, (15, 15), 0) if batch_size == 1 else cpu_batch_gaussian([img] if batch_size == 1 else img),
             gpu_batch_gaussian),
            ("Sobel", 
             lambda img: cv2.Sobel(img, cv2.CV_64F, 1, 1) if batch_size == 1 else cpu_batch_sobel([img] if batch_size == 1 else img),
             gpu_batch_sobel),
            ("FFT", 
             cpu_fft_filter if batch_size == 1 else cpu_batch_fft,
             gpu_fft_filter)
        ]
        
        for op_name, cpu_func, gpu_func in operations:
            try:
                cpu_time, gpu_time = benchmark_operation(cpu_func, gpu_func, images, batch_size, device)
                
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                
                results.append({
                    'Resolution': f"{resolution[0]}x{resolution[1]}",
                    'Batch_Size': batch_size,
                    'Operation': op_name,
                    'CPU_Time_ms': cpu_time * 1000,
                    'GPU_Time_ms': gpu_time * 1000,
                    'Speedup': speedup,
                    'Total_Pixels': resolution[0] * resolution[1] * batch_size
                })
                
                print(f"  {op_name:8}: CPU={cpu_time*1000:6.2f}ms, GPU={gpu_time*1000:6.2f}ms, Speedup={speedup:.1f}x")
                
            except Exception as e:
                print(f"  {op_name:8}: ERRO - {e}")

# ===========================
# Análise dos Resultados
# ===========================
df = pd.DataFrame(results)

print("\n=== ANÁLISE DE RESULTADOS ===")

# Encontrar ponto de breakeven
print("\n1. Operações onde GPU é mais rápida:")
gpu_wins = df[df['Speedup'] > 1.0]
if not gpu_wins.empty:
    print(gpu_wins[['Resolution', 'Batch_Size', 'Operation', 'Speedup']].to_string(index=False))
else:
    print("GPU não foi mais rápida em nenhuma operação testada")

print("\n2. Melhor speedup por operação:")
best_speedups = df.loc[df.groupby('Operation')['Speedup'].idxmax()]
print(best_speedups[['Operation', 'Resolution', 'Batch_Size', 'Speedup']].to_string(index=False))

print("\n3. Threshold de eficiência (pixels necessários para GPU compensar):")
for operation in df['Operation'].unique():
    op_data = df[df['Operation'] == operation]
    efficient = op_data[op_data['Speedup'] > 1.0]
    if not efficient.empty:
        min_pixels = efficient['Total_Pixels'].min()
        print(f"  {operation}: {min_pixels:,} pixels")
    else:
        print(f"  {operation}: GPU nunca foi mais eficiente nos testes")

# ===========================
# Visualização
# ===========================
plt.figure(figsize=(15, 10))

# Subplot 1: Speedup por batch size
plt.subplot(2, 3, 1)
for operation in df['Operation'].unique():
    op_data = df[(df['Operation'] == operation) & (df['Resolution'] == '640x480')]
    plt.plot(op_data['Batch_Size'], op_data['Speedup'], marker='o', label=operation)
plt.xlabel('Batch Size')
plt.ylabel('Speedup (CPU/GPU)')
plt.title('Speedup vs Batch Size (640x480)')
plt.legend()
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

# Subplot 2: Tempo absoluto GPU vs CPU
plt.subplot(2, 3, 2)
for operation in df['Operation'].unique():
    op_data = df[(df['Operation'] == operation) & (df['Batch_Size'] == 20)]
    plt.scatter(op_data['CPU_Time_ms'], op_data['GPU_Time_ms'], label=operation, s=100)
plt.xlabel('CPU Time (ms)')
plt.ylabel('GPU Time (ms)')
plt.title('GPU vs CPU Time (Batch=20)')
plt.legend()
plt.plot([0, df['CPU_Time_ms'].max()], [0, df['CPU_Time_ms'].max()], 'r--', alpha=0.5)
plt.grid(True, alpha=0.3)

# Subplot 3: Speedup por resolução
plt.subplot(2, 3, 3)
for operation in df['Operation'].unique():
    op_data = df[(df['Operation'] == operation) & (df['Batch_Size'] == 20)]
    resolutions = [320*240, 640*480, 1280*960]
    speedups = []
    for res in TEST_RESOLUTIONS:
        res_str = f"{res[0]}x{res[1]}"
        res_data = op_data[op_data['Resolution'] == res_str]
        if not res_data.empty:
            speedups.append(res_data['Speedup'].iloc[0])
        else:
            speedups.append(0)
    plt.plot(resolutions, speedups, marker='o', label=operation)
plt.xlabel('Total Pixels')
plt.ylabel('Speedup')
plt.title('Speedup vs Resolution (Batch=20)')
plt.legend()
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
plt.xscale('log')
plt.grid(True, alpha=0.3)

# Mostrar exemplo de FFT
plt.subplot(2, 3, 4)
fft_result = cpu_fft_filter(base_frame)
plt.imshow(cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(fft_result, cv2.COLOR_BGR2RGB))
plt.title('FFT High-pass Filter')
plt.axis('off')

# Heatmap de speedup
plt.subplot(2, 3, 6)
pivot_data = df.pivot_table(index='Batch_Size', columns='Operation', values='Speedup', aggfunc='max')
im = plt.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)
plt.colorbar(im, label='Speedup')
plt.xlabel('Operation')
plt.ylabel('Batch Size')
plt.title('Speedup Heatmap')
plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
plt.yticks(range(len(pivot_data.index)), pivot_data.index)

plt.tight_layout()
plt.savefig("comprehensive_benchmark_results.png", dpi=300, bbox_inches='tight')
print("\nGráficos salvos em 'comprehensive_benchmark_results.png'")

# Salvar dados para artigo
df.to_csv("benchmark_results.csv", index=False)
print("Dados salvos em 'benchmark_results.csv'")



app.stop()

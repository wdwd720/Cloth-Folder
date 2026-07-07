import modal

app = modal.App("terafold-gpu-smoke")

image = modal.Image.debian_slim().pip_install("torch", "numpy")

@app.function(gpu="L40S", image=image, timeout=600)
def run():
    import torch
    import platform
    print("python_platform", platform.platform())
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("gpu_name", torch.cuda.get_device_name(0))
        x = torch.randn((4096, 4096), device="cuda")
        y = x @ x
        print("result_mean", float(y.mean().detach().cpu()))
    return {"ok": torch.cuda.is_available()}

@app.local_entrypoint()
def main():
    print(run.remote())

# 🚀 NVIDIA GPU Setup for Running AI Models (Windows + PyTorch + CUDA 13.0.1)

This guide explains how to **set up your NVIDIA GPU** on **Windows** for running **AI models** using **PyTorch** (via `pip`) with **CUDA 13.0.1** support.

---

## 🧠 Prerequisites

Before installing, ensure you have:

1. ✅ **NVIDIA GPU** (with CUDA Compute Capability ≥ 6.0)  
   → Check supported GPUs here: [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)

2. 🧩 **Windows 10/11** (64-bit)

3. 🧰 **Python 3.8–3.12**  
   → Recommended: **Python 3.10 or 3.11**

4. 💻 **Administrator Access**

---

## ⚙️ Step 1. Install NVIDIA GPU Driver

- Download and install the **latest NVIDIA GPU driver** from:  
  🔗 [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)

- During installation, choose:
  - ✅ *Custom Installation*
  - ✅ *Perform a clean installation*

🧭 **Verify Installation**  
Open **Command Prompt** and run:
```bash
nvidia-smi
````

You should see GPU details listed.

---

## 🧩 Step 2. Install CUDA 13.0.1 (Windows)

* Download CUDA 13.0.1 Windows Installer:
  📦 `cuda_13.0.1_windows.exe`

* Run the installer:

  ```bash
  cuda_13.0.1_windows.exe
  ```

* During setup:

  * Choose **Express Installation**
  * Ensure **CUDA Toolkit**, **Driver**, and **Samples** are selected

🧭 **Verify Installation**

```bash
nvcc --version
```

Expected output:

```
Cuda compilation tools, release 13.0, V13.0.1
```

---

## 🧩 Step 3. Install cuDNN (Optional but Recommended)

1. Go to: [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
2. Download **cuDNN for CUDA 13.0**
3. Extract and copy files into:

   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
   ```

Make sure `bin`, `include`, `lib` folders are merged.

---

## 🐍 Step 4. Create Python Virtual Environment

Open **Command Prompt / PowerShell** and run:

```bash
python -m venv venv
venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

## 🔥 Step 5. Install PyTorch with CUDA 13.0 Support

> ⚠️ PyTorch might not yet have direct binaries for CUDA 13.0 (as of now).
> Use the **closest supported version** (e.g., CUDA 12.4) or build from source.

Check the latest install command at:
🔗 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Example (for CUDA 12.4, adjust when 13.0 is supported):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

If you manually built for CUDA 13.0, use:

```bash
pip install torch torchvision torchaudio
```

---

## 🧪 Step 6. Test GPU with PyTorch

Run Python:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0))
```

✅ If `torch.cuda.is_available()` returns `True`, your GPU is ready!

---

## 🧰 Optional: Install Additional AI Tools

You can now install your preferred AI frameworks:

```bash
pip install transformers accelerate diffusers
```

Or tools like:

```bash
pip install jupyterlab matplotlib numpy pandas
```

---

## 🧹 Troubleshooting

| Issue                               | Solution                                                   |
| ----------------------------------- | ---------------------------------------------------------- |
| `nvcc` not found                    | Ensure CUDA was installed successfully                     |
| `torch.cuda.is_available()` = False | Check CUDA + Driver compatibility                          |
| Mismatched CUDA versions            | Ensure PyTorch CUDA version matches installed CUDA toolkit |
| cuDNN errors                        | Copy cuDNN files into CUDA directory correctly             |

---

## 🎉 You’re All Set!

You’ve successfully set up your **NVIDIA GPU** with **CUDA 13.0.1** and **PyTorch** for AI development on Windows.

> 🧠 You can now train models, run inference, or experiment with GenAI projects effortlessly.

---

**Author:** *S.Earnest*

**Date:** *October 2025*


# Edge AI RAG System Deployment on NVIDIA Jetson: Storage Management

During the deployment of the Edge AI RAG system on NVIDIA Jetson, a critical storage limitation was encountered. The Jetson's internal storage (typically 15.7 GB) was insufficient for:

*   JetPack SDK components (~10 GB)
*   Python libraries (~6 GB)
*   AI models (~2-3 GB)
*   Operating system and cache (~5 GB)

## Solution Implemented: Symbolic Links (Symlinks)

To overcome the storage limitation, symbolic links (symlinks) were created to redirect large directories to external storage. This strategy enabled the system to function effectively despite the limited internal storage.

### Concept: Symbolic Link (Symlink)

A symbolic link acts as a shortcut or pointer to another location. The system perceives the files as being in the symlink's location, but they are physically stored on the external drive.

```
Internal Storage (Limited)         External Storage (Large)
┌─────────────────────┐            ┌──────────────────────┐
│ /usr/python3.8/     │──────────▶ | /media/jetson/lib/  |
│ (symlink - 0 bytes) │            │ (actual files - 6GB) │
└─────────────────────┘            └──────────────────────┘



System thinks files are in /usr/python3.8/
But they're actually stored on external drive! named lib
```

### Result

Successfully deployed the complete system by offloading approximately 15 GB to external storage while maintaining full functionality.

## Driver Installation and `.bashrc` Configuration

All necessary drivers were installed on a USB drive, and their symbolic links were added to the `.bashrc` file to ensure the system could locate them.

### Example `.bashrc` Entries:

```bash
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH

# CUDA paths
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH

# Optional: cuDNN / TensorRT if installed in USB
export LD_LIBRARY_PATH=/media/jetson/lib/sdk/cuda-11.4/targets/aarch64-linux/lib:$LD_LIBRARY_PATH

# CUDA environment setup
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.4/targets/aarch64-linux/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Hugging Face cache directories redirected to external storage
export HF_HOME=/media/jetson/lib/huggingface_cache
export TRANSFORMERS_CACHE=/media/jetson/lib/huggingface_cache
export HF_DATASETS_CACHE=/media/jetson/lib/huggingface_cache
export HF_HOME=/media/jetson/lib/huggingface
```

## PyTorch with CUDA on Jetson

Unlike a regular PC, you cannot install PyTorch directly using `pip install torch` on Jetson devices. This is because Jetson boards use ARM64 architecture, and the available precompiled PyTorch wheels from PyPI are meant for x86_64 machines.

On Jetson, PyTorch must be installed in a way that matches:

*   The JetPack version (which defines CUDA + cuDNN versions).
*   The Python version available on the device.

### My Setup includes:

*   NVIDIA Jetson Xavier
*   JetPack: 5.1.5
*   CUDA: 11.4
*   Python: 3.8.10
*   GPU Shared RAM: 6.7GB

For this configuration, PyTorch was installed from the [Jetson Zoo](https://elinux.org/Jetson_Zoo) (a community-maintained repository of wheels for Jetson devices) instead of the standard PyPI channel.

I am using PyTorch v2.1.0 for this setup.

## System Architecture

```
┌─────────────────────────────────────────────┐
│            Streamlit Web Interface          │
│         (User uploads docs, asks questions) │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼───────────────────────────┐
│          LangChain RAG Pipeline            │
│  ┌─────────────────────────────────────┐   │
│  │ 1. Document Processing & Chunking   │   │
│  │ 2. Embedding Generation (GPU)       │   │
│  │ 3. Vector Store (FAISS)             │   │
│  │ 4. Semantic Search & Retrieval      │   │
│  │ 5. LLM Response Generation (GPU)    │   │
│  └─────────────────────────────────────┘   │
└────────────────┬───────────────────────────┘
                 │
┌────────────────▼───────────────────────────┐
│         AI Models (Running on Jetson GPU)  │
│  • Qwen 2.5 0.5B Instruct (LLM)            │
│  • all-MiniLM-L6-v2 (Embeddings)           │
└──────────────────────────────────── ───────┘
                 │
┌────────────────▼─────────────────────┐
│          NVIDIA Jetson Hardware      │
│  • CUDA-enabled GPU                  │
│  • ARM CPU                           │
│  • Local Storage                     │
└──────────────────────────────────────┘
```

## Running the System

After setting up the environment:

1.  **Create a Virtual Environment:** Create a virtual environment on the USB drive with any desired name.
2.  **Install Libraries:** Install all necessary libraries listed in `requirements.txt`. Ensure you use the `--no-cache-dir` flag to prevent caching on internal storage.

    ```bash
    pip3 install -r requirements.txt --no-cache-dir
    ```

3.  **Run the Application:** Start the Streamlit application.

    ```bash
    streamlit run Final.py
    ```
```
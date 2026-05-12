# LLaVA-Gemma-2B to GGUF Converter 🚀

A streamlined, memory-safe toolkit for extracting and converting the **Intel/llava-gemma-2b** Vision-Language Model (VLM) into the **GGUF** format for local inference with `llama.cpp`. 

Because VLMs consist of both a base text model and a vision encoder/projector, they must be separated and converted into two distinct GGUF files (`.gguf` for the LLM and `mmproj.gguf` for the vision tower). These Colab-ready notebooks handle the entire "surgery" and conversion process safely without exhausting system RAM.

## 📋 Features
* **Memory-Safe Extraction:** Safely splits intertwined text and vision weights onto CPU RAM using chunked processing and garbage collection to prevent Colab from crashing.
* **Automated Index Alignment:** Automatically rewrites the `safetensors.index.json` to strip `language_model.` prefixes and remove orphaned vision keys.
* **End-to-End Conversion:** Compiles the latest `llama.cpp` from source and executes the correct Python conversion scripts for both the LLM and the Multi-Modal Projector.
* **Precision Handling:** Automatically patches `bfloat16` tensors to `float32` during the vision extraction phase to ensure compatibility with `llama.cpp`'s image encoder script.

## 📁 Repository Contents

1. **`Llava_Gemma2_Base_LLM_FP16.ipynb`**
   * Downloads the base model using high-speed `hf_transfer`.
   * Strips out the vision tower and multi-modal projector weights.
   * Modifies the Hugging Face weight map.
   * Converts the standalone Gemma 2B text model into `llava-gemma-2b-f16.gguf`.

2. **`Llava_Gemma2_mmproj_file.ipynb`**
   * Extracts the `vision_tower` and `mm_projector` weights into a separate directory.
   * Generates a compatible `clip` configuration file (`config.json`) with `quick_gelu` activation.
   * Converts weight formats and casts `bfloat16` tensors to `float32`.
   * Converts the vision components into `mmproj-model-f16.gguf` using the legacy `mtmd` conversion script.

## 🛠️ Prerequisites

* A **Google Colab** environment (a standard T4 GPU instance is sufficient, though CPU RAM is the primary bottleneck).
* A **Hugging Face Account & Access Token** (Required to download the base model).

## 🚀 Usage Guide

### Step 1: Convert the Base LLM
1. Open `Llava_Gemma2_Base_LLM_FP16.ipynb` in Google Colab.
2. In the setup cell, replace the placeholder `HF_TOKEN` with your actual Hugging Face write/read token.
3. Run all cells. 
4. The script will download the model, isolate the text weights, and output: `llava-gemma-2b-f16.gguf`.
5. Download this file to your local machine.

### Step 2: Convert the Multi-Modal Projector (mmproj)
1. Open `Llava_Gemma2_mmproj_file.ipynb` in Google Colab.
2. Again, insert your `HF_TOKEN`.
3. Run all cells.
4. The script will extract the vision weights, format the config, cast the tensors, and output: `mmproj-model-f16.gguf` inside the `/content/vit/` directory.
5. Download this file to your local machine.

## 💻 Running Locally with llama.cpp

Once you have downloaded both files (`llava-gemma-2b-f16.gguf` and `mmproj-model-f16.gguf`), you can run them locally using `llama-cli` or `llama-server`.

**Example CLI Command:**
```bash
./llama-cli -m path/to/llava-gemma-2b-f16.gguf \
       --mmproj path/to/mmproj-model-f16.gguf \
       -p "Describe this image in detail:" \
       --image path/to/your/image.jpg \
       -c 4096

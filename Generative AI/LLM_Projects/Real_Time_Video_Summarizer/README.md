# 🎬 Text-to-Video Synthesis & Summarization

This project demonstrates how to implement generative text-to-video synthesis using pre-trained diffusion models. It provides pipeline notebooks to generate frames, synthesize video tracks, and compress summarizations.

---

## 📂 Key Notebooks

- **`text_to_video.ipynb`**:
  - Implements the text-to-video diffusion pipeline using Hugging Face **Diffusers**.
  - Sets up memory optimizations (such as CPU offloading and float16 calculations) to run models on standard GPU resources.
  - Converts text prompts into dynamic video sequences.

---

## 🛠️ Tech Stack & Requirements

- **Models**: Damo-vilab Text-to-Video Synthesis, Stable Diffusion
- **Frameworks**: PyTorch, Diffusers, Transformers, Accelerate
- **Utilities**: ImageIO, OpenCV (for frame encoding)

---

## 🚀 Usage Guide

1. Install Diffusers and PyTorch:
   ```bash
   pip install torch diffusers transformers accelerate imageio[ffmpeg] opencv-python
   ```
2. Open `text_to_video.ipynb` in Jupyter Notebooks.
3. Run the cells to load the pipeline, define prompts, and compile the final video tracks.

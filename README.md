# ⚙️ Restoration Station

**The Intelligent Historical Archive Restoration Engine.**

Restoration Station is a deployed Deep Learning application designed to breathe new life into black & white media. It utilizes a 3-stage pipeline to automatically remove film grain, hallucinate realistic colors, and sharpen details in both **Images** and **Video**.

## Features

- **AI Colorization:** Uses the Zhang et al. (ECCV 2016) CNN trained on 1.3 million images to predict realistic colors in the CIELAB color space.
- **Video Support:** Batch processes black & white MP4 videos frame-by-frame.
- **Color Confidence Control:** A unique slider to manually adjust the saturation/confidence of the AI's predictions (Pastel vs. Vivid).
- **Full Pipeline:**
  1.  **Denoise:** Non-Local Means Denoising to remove old film grain.
  2.  **Colorize:** Deep Convolutional Neural Network.
  3.  **Sharpen:** Edge enhancement kernel.
- **Cloud Ready:** Features an automated model downloader that fetches the 129MB AI weights from Intel's OpenVINO storage on runtime, allowing lightweight deployment.

## Tech Stack

- **Python 3.9+**
- **Streamlit:** For the interactive web interface.
- **OpenCV (DNN Module):** For image processing and running the Caffe model.
- **NumPy:** For matrix operations and channel manipulation.

# Client-Side Neural TTS (Kokoro-82M)

A high-quality, privacy-focused Text-to-Speech application that runs entirely in your browser. Powered by **Kokoro-82M** and **WebAssembly**.

## 🚀 Technology Stack

- **Frontend Framework**: [React](https://react.dev/) + [Vite](https://vitejs.dev/)
- **Language**: [TypeScript](https://www.typescriptlang.org/)
- **AI Model**: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (82 million parameters)
- **Inference Engine**: [ONNX Runtime Web](https://onnxruntime.ai/docs/execution-providers/WebAssembly-ExecutionProvider.html) via [Transformers.js](https://huggingface.co/docs/transformers.js/index) or direct `kokoro-js` wrapper.
- **Styling**: Vanilla CSS (CSS Variables, Glassmorphism)

## 🧠 How It Works

Unlike traditional TTS systems that rely on cloud APIs (like Google Cloud TTS or AWS Polly) or robotic operating system APIs (Web Speech API), this application runs a **Deep Learning model directly in your browser**.

1.  **Model Loading**:
    - On first load, the app downloads the **quantized ONNX model (~80MB)**.
    - The model is cached in the browser for subsequent visits.

2.  **WebAssembly (WASM) Execution**:
    - The model inference runs on the client's CPU (via WASM) or GPU (via WebGPU if available).
    - We use `onnxruntime-web` to execute the neural network operations efficiently in JavaScript.

3.  **Synthesis Pipeline**:
    - **Text Normalization**: Converts raw text into phonemes.
    - **Inference**: The Kokoro model predicts the audio waveform from phonemes.
    - **Vocoder**: Generates the final audio samples at 24kHz.
    - **Playback**: The raw float data is converted to a WAV blob and played via the HTML5 Audio element.

## 🛠️ Setup & Running

1.  **Install Dependencies**:
    ```bash
    npm install
    ```

2.  **Start Development Server**:
    ```bash
    npm run dev
    ```

3.  **Build for Production**:
    ```bash
    npm run build
    ```
    *Note: The build process copies necessary `.wasm` files to the output directory.*

## 🔒 Privacy

Because the inference happens locally:
- **No text is sent to any server.**
- Your data stays on your device.
- It works offline after the initial model download.

## 🧩 Deployment

This is a static site. When deploying (e.g., to Vercel or Netlify), ensure the `.wasm` files are served with the `application/wasm` MIME type.

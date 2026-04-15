import { KokoroTTS } from 'kokoro-js';

let tts: KokoroTTS | null = null;

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data as
    | { type: 'load' }
    | { type: 'generate'; text: string; voice: string };

  if (msg.type === 'load') {
    try {
      tts = await KokoroTTS.from_pretrained('onnx-community/Kokoro-82M-v1.0-ONNX', {
        dtype: 'fp32',
      });
      self.postMessage({ type: 'ready' });
    } catch (err: any) {
      self.postMessage({ type: 'error', message: err.message || String(err) });
    }

  } else if (msg.type === 'generate') {
    if (!tts) {
      self.postMessage({ type: 'error', message: 'Model not loaded yet' });
      return;
    }
    try {
      const audio: any = await tts.generate(msg.text, { voice: msg.voice as any });
      const audioData: Float32Array | undefined = audio.audio || audio.data;
      if (!audioData) throw new Error('Audio data missing from model output');
      const sampleRate: number = audio.sampling_rate || audio.sampleRate || 24000;

      // Transfer the underlying ArrayBuffer for zero-copy delivery to the main thread.
      self.postMessage(
        { type: 'chunk_done', audio: audioData, sampleRate },
        [audioData.buffer],
      );
    } catch (err: any) {
      self.postMessage({ type: 'error', message: err.message || String(err) });
    }
  }
};

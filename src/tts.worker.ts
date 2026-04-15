import { KokoroTTS } from 'kokoro-js';

// The project tsconfig uses lib:DOM which types `self` as Window. The DOM lib
// doesn't include DedicatedWorkerGlobalScope, so we use a local typed wrapper
// for postMessage that matches the actual worker runtime signature.
const workerPost = (msg: unknown, transfer?: Transferable[]) =>
  (self as unknown as { postMessage(m: unknown, t?: Transferable[]): void })
    .postMessage(msg, transfer);

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
      workerPost({ type: 'ready' });
    } catch (err: any) {
      workerPost({ type: 'error', message: err.message || String(err) });
    }

  } else if (msg.type === 'generate') {
    if (!tts) {
      workerPost({ type: 'error', message: 'Model not loaded yet' });
      return;
    }
    try {
      const audio: any = await tts.generate(msg.text, { voice: msg.voice as any });
      const audioData: Float32Array | undefined = audio.audio || audio.data;
      if (!audioData) throw new Error('Audio data missing from model output');
      const sampleRate: number = audio.sampling_rate || audio.sampleRate || 24000;

      // Transfer the underlying ArrayBuffer for zero-copy delivery to the main thread.
      workerPost(
        { type: 'chunk_done', audio: audioData, sampleRate },
        [audioData.buffer],
      );
    } catch (err: any) {
      workerPost({ type: 'error', message: err.message || String(err) });
    }
  }
};

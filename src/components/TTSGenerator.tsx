import React, { useState, useEffect, useRef } from 'react';

const VOICES = [
  { id: 'af_bella', name: 'Bella (US Female)' },
  { id: 'af_nicole', name: 'Nicole (US Female)' },
  { id: 'am_michael', name: 'Michael (US Male)' },
  { id: 'am_adam', name: 'Adam (US Male)' },
  { id: 'bf_emma', name: 'Emma (UK Female)' },
  { id: 'bm_george', name: 'George (UK Male)' },
];

const FACTS = [
  'Kokoro-82M has just 82 million parameters — roughly 40× smaller than most neural TTS models.',
  'Everything runs in your browser via ONNX Runtime. No audio ever leaves your device.',
  'Long text is split at sentence boundaries to stay within the model\'s 510-token limit per pass.',
  'Neural TTS models prosody and phonemes end-to-end — no pre-recorded audio clips involved.',
  'The ONNX model weights are cached after the first download, so future runs start instantly.',
];

type Stage = 'idle' | 'splitting' | 'generating' | 'stitching' | 'done';

const STAGE_LABELS: Record<Stage, string> = {
  idle: '',
  splitting: 'Splitting text…',
  generating: 'Generating audio…',
  stitching: 'Stitching audio…',
  done: 'Done!',
};

// Messages the worker can send back to the main thread.
type WorkerOutMsg =
  | { type: 'ready' }
  | { type: 'chunk_done'; audio: Float32Array; sampleRate: number }
  | { type: 'error'; message: string };

export const TTSGenerator: React.FC = () => {
  // Uncontrolled textarea — no re-renders while typing.
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [voice, setVoice] = useState<string>('af_bella');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isModelLoading, setIsModelLoading] = useState<boolean>(false);
  const [workerReady, setWorkerReady] = useState<boolean>(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stage, setStage] = useState<Stage>('idle');
  const [chunkIndex, setChunkIndex] = useState(0);
  const [chunkTotal, setChunkTotal] = useState(1);
  const [factIndex, setFactIndex] = useState(0);

  const workerRef = useRef<Worker | null>(null);
  // Holds the resolve/reject for the currently in-flight chunk postMessage.
  const pendingRef = useRef<{
    resolve: (v: { audio: Float32Array; sampleRate: number }) => void;
    reject: (e: Error) => void;
  } | null>(null);
  const prevAudioUrlRef = useRef<string | null>(null);

  // Rotate facts every 3.5s while loading.
  useEffect(() => {
    if (!isLoading) return;
    const id = setInterval(() => {
      setFactIndex(i => (i + 1) % FACTS.length);
    }, 3500);
    return () => clearInterval(id);
  }, [isLoading]);

  // Spawn worker once on mount; terminate on unmount.
  useEffect(() => {
    setIsModelLoading(true);

    const worker = new Worker(
      new URL('../tts.worker.ts', import.meta.url),
      { type: 'module' },
    );

    worker.onmessage = (e: MessageEvent<WorkerOutMsg>) => {
      const msg = e.data;
      if (msg.type === 'ready') {
        setIsModelLoading(false);
        setWorkerReady(true);
      } else if (msg.type === 'chunk_done') {
        pendingRef.current?.resolve({ audio: msg.audio, sampleRate: msg.sampleRate });
        pendingRef.current = null;
      } else if (msg.type === 'error') {
        if (pendingRef.current) {
          // Error during inference — reject the in-flight chunk promise.
          pendingRef.current.reject(new Error(msg.message));
          pendingRef.current = null;
        } else {
          // Error during model load.
          setError('Failed to load TTS model: ' + msg.message);
          setIsModelLoading(false);
        }
      }
    };

    worker.onerror = (e) => {
      const msg = e.message || 'Unknown worker error';
      if (pendingRef.current) {
        pendingRef.current.reject(new Error(msg));
        pendingRef.current = null;
      } else {
        setError('Worker error: ' + msg);
        setIsModelLoading(false);
      }
    };

    workerRef.current = worker;
    worker.postMessage({ type: 'load' });

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  // Send one chunk to the worker and await its Float32Array result.
  function generateChunk(text: string, voice: string): Promise<{ audio: Float32Array; sampleRate: number }> {
    return new Promise((resolve, reject) => {
      pendingRef.current = { resolve, reject };
      workerRef.current!.postMessage({ type: 'generate', text, voice });
    });
  }

  const handleGenerate = async () => {
    const text = textareaRef.current?.value.trim() ?? '';
    if (!workerRef.current || !workerReady || !text) return;

    setIsLoading(true);
    setError(null);
    setAudioUrl(null);
    setFactIndex(0);

    try {
      setStage('splitting');
      const chunks = splitIntoChunks(text);
      setChunkTotal(chunks.length);
      setChunkIndex(0);

      const audioBuffers: Float32Array[] = [];
      let sampleRate = 24000;

      setStage('generating');
      for (let i = 0; i < chunks.length; i++) {
        setChunkIndex(i + 1);
        const result = await generateChunk(chunks[i], voice);
        sampleRate = result.sampleRate;
        audioBuffers.push(result.audio);
      }

      setStage('stitching');
      const totalLength = audioBuffers.reduce((sum, buf) => sum + buf.length, 0);
      const combined = new Float32Array(totalLength);
      let offset = 0;
      for (const buf of audioBuffers) {
        combined.set(buf, offset);
        offset += buf.length;
      }

      const wavBlob = audioToWav(combined, sampleRate);
      if (prevAudioUrlRef.current) URL.revokeObjectURL(prevAudioUrlRef.current);
      const url = URL.createObjectURL(wavBlob);
      prevAudioUrlRef.current = url;
      setAudioUrl(url);
      setStage('done');
    } catch (err: any) {
      console.error(err);
      setError('Generation failed: ' + (err.message || String(err)));
      setStage('idle');
    } finally {
      setIsLoading(false);
    }
  };

  const progressPct =
    stage === 'splitting' ? 5
    : stage === 'generating' ? 10 + (chunkIndex / chunkTotal) * 75
    : stage === 'stitching' ? 90
    : stage === 'done' ? 100
    : 0;

  const stageLabel =
    stage === 'generating' && chunkTotal > 1
      ? `Generating audio (chunk ${chunkIndex} of ${chunkTotal})…`
      : STAGE_LABELS[stage];

  return (
    <div className="card">
      <h2>Client-Side AI Voice</h2>

      <div className="controls">
        <select value={voice} onChange={(e) => setVoice(e.target.value)} disabled={isLoading}>
          {VOICES.map(v => (
            <option key={v.id} value={v.id}>{v.name}</option>
          ))}
        </select>
      </div>

      <textarea
        ref={textareaRef}
        defaultValue="Hello! This is a test of high-quality client-side text to speech."
        placeholder="Type something here…"
        disabled={isLoading}
      />

      <div className="controls">
        <button onClick={handleGenerate} disabled={isLoading || isModelLoading || !workerReady}>
          {isModelLoading ? 'Loading model…' : isLoading ? 'Generating…' : 'Generate Voice'}
        </button>
      </div>

      {isLoading && (
        <div className="gen-panel">
          <div className="gen-stage-row">
            <span className="gen-stage-label">{stageLabel}</span>
            {chunkTotal > 1 && stage === 'generating' && (
              <span className="gen-chunk-count">{chunkIndex}/{chunkTotal}</span>
            )}
          </div>
          <div className="gen-progress-track">
            <div className="gen-progress-fill" style={{ width: `${progressPct}%` }} />
          </div>
          <div className="facts-panel">
            <span className="facts-eyebrow">Did you know?</span>
            <p className="facts-text" key={factIndex}>{FACTS[factIndex]}</p>
          </div>
        </div>
      )}

      {stage === 'done' && !isLoading && (
        <div className="gen-done-row">
          <span className="gen-done-label">Done!</span>
          <div className="gen-progress-track gen-progress-track--done">
            <div className="gen-progress-fill" style={{ width: '100%' }} />
          </div>
        </div>
      )}

      {error && (
        <div style={{ color: '#ff6b6b', marginTop: '1rem', fontSize: '0.9em' }}>
          {error}
        </div>
      )}

      {audioUrl && (
        <audio key={audioUrl} controls src={audioUrl} autoPlay />
      )}
    </div>
  );
};

const MAX_CHUNK_CHARS = 500;

function splitIntoChunks(input: string): string[] {
  const sentences = input.match(/[^.!?;]+[.!?;]*/g) ?? [input];
  const chunks: string[] = [];
  let current = '';

  for (const sentence of sentences) {
    const trimmed = sentence.trim();
    if (!trimmed) continue;
    if (current.length + trimmed.length + 1 <= MAX_CHUNK_CHARS) {
      current = current ? `${current} ${trimmed}` : trimmed;
    } else {
      if (current) chunks.push(current);
      current = trimmed;
    }
  }
  if (current) chunks.push(current);
  return chunks;
}

function audioToWav(channels: Float32Array, sampleRate: number) {
  const buffer = new ArrayBuffer(44 + channels.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + channels.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, channels.length * 2, true);

  let offset = 44;
  for (let i = 0; i < channels.length; i++) {
    const s = Math.max(-1, Math.min(1, channels[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

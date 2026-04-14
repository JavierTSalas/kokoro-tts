import React, { useState, useEffect, useRef } from 'react';
import { KokoroTTS } from 'kokoro-js';

// Define voice options based on Kokoro's available voices (commonly supported ones)
const VOICES = [
  { id: 'af_bella', name: 'Bella (US Female)' },
  { id: 'af_nicole', name: 'Nicole (US Female)' },
  { id: 'am_michael', name: 'Michael (US Male)' },
  { id: 'am_adam', name: 'Adam (US Male)' },
  { id: 'bf_emma', name: 'Emma (UK Female)' },
  { id: 'bm_george', name: 'George (UK Male)' },
];

export const TTSGenerator: React.FC = () => {
  const [text, setText] = useState<string>('Hello! This is a test of high-quality client-side text to speech.');
  const [voice, setVoice] = useState<string>('af_bella');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isModelLoading, setIsModelLoading] = useState<boolean>(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string>('');

  const ttsRef = useRef<KokoroTTS | null>(null);
  const prevAudioUrlRef = useRef<string | null>(null);

  useEffect(() => {
    // Initialize TTS model
    const initTTS = async () => {
      try {
        setIsModelLoading(true);
        setProgress('Loading model (approx 80MB)...');
        // Correct initialization using static method
        const tts = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", {
          dtype: "fp32", // fp32 is safer for compatibility
        });
        ttsRef.current = tts;
        setIsModelLoading(false);
        setProgress('Model ready!');
      } catch (err: any) {
        console.error("Failed to initialize TTS:", err);
        setError("Failed to load TTS model. " + (err.message || String(err)));
        setIsModelLoading(false);
      }
    };
    
    // Check if we need simple init or complex one. 
    // The kokoro-js documentation is sparse, but usually standard init works.
    // If this fails, we might need a specific correct import.
    initTTS();

    return () => {
      // Cleanup if needed
    };
  }, []);

  // Kokoro-82M has a hard limit of 510 phoneme tokens per inference pass.
  // ~500 characters of English text is a safe proxy for staying under ~450 tokens,
  // leaving headroom before the limit where quality degrades.
  const MAX_CHUNK_CHARS = 500;

  // Split text at sentence boundaries, then group into chunks that each stay
  // under MAX_CHUNK_CHARS. Splitting on sentence endings keeps prosody natural.
  function splitIntoChunks(input: string): string[] {
    // Split on sentence-ending punctuation, keeping the delimiter attached.
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
        // A single sentence that exceeds the limit must still be sent as-is;
        // the model will handle it as best it can.
        current = trimmed;
      }
    }
    if (current) chunks.push(current);
    return chunks;
  }

  const handleGenerate = async () => {
    if (!ttsRef.current || !text) return;

    setIsLoading(true);
    setError(null);
    setAudioUrl(null);

    try {
      const chunks = splitIntoChunks(text);
      const audioBuffers: Float32Array[] = [];
      let sampleRate = 24000;

      for (let i = 0; i < chunks.length; i++) {
        setProgress(`Generating speech… (part ${i + 1} of ${chunks.length})`);
        const audio: any = await ttsRef.current.generate(chunks[i], {
          voice: voice as any,
        });

        const audioData: Float32Array | undefined = audio.audio || audio.data;
        if (!audioData) {
          throw new Error(`Generated audio data is missing for chunk ${i + 1}`);
        }
        sampleRate = audio.sampling_rate || audio.sampleRate || 24000;
        audioBuffers.push(audioData);
      }

      // Concatenate all Float32Array chunks into one buffer.
      const totalLength = audioBuffers.reduce((sum, buf) => sum + buf.length, 0);
      const combined = new Float32Array(totalLength);
      let offset = 0;
      for (const buf of audioBuffers) {
        combined.set(buf, offset);
        offset += buf.length;
      }

      const wavBlob = audioToWav(combined, sampleRate);
      if (prevAudioUrlRef.current) {
        URL.revokeObjectURL(prevAudioUrlRef.current);
      }
      const url = URL.createObjectURL(wavBlob);
      prevAudioUrlRef.current = url;
      setAudioUrl(url);
      setProgress('Done!');
    } catch (err: any) {
      console.error(err);
      setError("Generation failed: " + (err.message || String(err)));
    } finally {
      setIsLoading(false);
    }
  };

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
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type something here..."
        disabled={isLoading}
      />

      <div className="controls">
        <button onClick={handleGenerate} disabled={isLoading || isModelLoading || !ttsRef.current}>
          {isLoading ? 'Generating...' : 'Generate Voice'}
        </button>
      </div>

      {progress && (
        <div className="status-bar">
          {(isLoading || isModelLoading) && <span className="loader"></span>}
          <span>{progress}</span>
        </div>
      )}

      {error && (
        <div style={{ color: '#ff6b6b', marginTop: '1rem' }}>
          {error}
        </div>
      )}

      {audioUrl && (
        <audio key={audioUrl} controls src={audioUrl} autoPlay />
      )}
    </div>
  );
};

// Helper function to create WAV file from Float32Array
function audioToWav(channels: Float32Array, sampleRate: number) {
  // Convert float32 to int16 PCM
  const buffer = new ArrayBuffer(44 + channels.length * 2);
  const view = new DataView(buffer);

  // RIFF identifier
  writeString(view, 0, 'RIFF');
  // file length
  view.setUint32(4, 36 + channels.length * 2, true);
  // RIFF type
  writeString(view, 8, 'WAVE');
  // format chunk identifier
  writeString(view, 12, 'fmt ');
  // format chunk length
  view.setUint32(16, 16, true);
  // sample format (raw)
  view.setUint16(20, 1, true);
  // channel count
  view.setUint16(22, 1, true);
  // sample rate
  view.setUint32(24, sampleRate, true);
  // byte rate (sample rate * block align)
  view.setUint32(28, sampleRate * 2, true);
  // block align (channel count * bytes per sample)
  view.setUint16(32, 2, true);
  // bits per sample
  view.setUint16(34, 16, true);
  // data chunk identifier
  writeString(view, 36, 'data');
  // data chunk length
  view.setUint32(40, channels.length * 2, true);

  // Write the PCM samples
  let offset = 44;
  for (let i = 0; i < channels.length; i++) {
    let s = Math.max(-1, Math.min(1, channels[i]));
    // scale to 16-bit signed integer
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

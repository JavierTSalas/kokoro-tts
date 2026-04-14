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

  const handleGenerate = async () => {
    if (!ttsRef.current || !text) return;

    setIsLoading(true);
    setError(null);
    setAudioUrl(null);

    try {
      // Kokoro-82M has a hard limit of ~510 tokens per generation pass.
      // Split text at sentence boundaries into chunks ≤500 chars to stay safely under it.
      const chunks = splitIntoChunks(text, 500);
      const audioSegments: Float32Array[] = [];
      let sampleRate = 24000;

      for (let i = 0; i < chunks.length; i++) {
        setProgress(`Generating speech… (chunk ${i + 1}/${chunks.length})`);
        const audio: any = await ttsRef.current.generate(chunks[i], {
          voice: voice as any,
        });

        const audioData = audio.audio || audio.data;
        if (!audioData) throw new Error("Generated audio data is missing");
        sampleRate = audio.sampling_rate || audio.sampleRate || 24000;
        audioSegments.push(audioData);
      }

      const combined = concatenateFloat32Arrays(audioSegments);
      const wavBlob = audioToWav(combined, sampleRate);
      const url = URL.createObjectURL(wavBlob);
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
        <audio controls src={audioUrl} autoPlay />
      )}
    </div>
  );
};

// Split text at sentence boundaries into chunks not exceeding maxChars characters.
// This keeps each chunk within Kokoro-82M's ~510-token generation limit.
function splitIntoChunks(text: string, maxChars: number): string[] {
  // Match sentences ending with . ! ? or newlines (including trailing whitespace)
  const sentences = text.match(/[^.!?\n]+[.!?\n]*/g) ?? [text];
  const chunks: string[] = [];
  let current = '';

  for (const sentence of sentences) {
    if (current.length > 0 && (current + sentence).length > maxChars) {
      chunks.push(current.trim());
      current = sentence;
    } else {
      current += sentence;
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

// Concatenate multiple Float32Arrays into one (for multi-chunk audio).
function concatenateFloat32Arrays(arrays: Float32Array[]): Float32Array {
  const total = arrays.reduce((sum, a) => sum + a.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const a of arrays) {
    out.set(a, offset);
    offset += a.length;
  }
  return out;
}

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

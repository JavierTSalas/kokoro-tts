
import { TTSGenerator } from './components/TTSGenerator';

function App() {
  return (
    <>
      <h1>Neural Voice </h1>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem' }}>
        Generate high-quality speech directly in your browser using AI.
        <br/>
        <small>Powered by Kokoro-82M & WebAssembly</small>
      </p>
      
      <TTSGenerator />
      
      <footer style={{ marginTop: '4rem', color: '#666', fontSize: '0.8rem' }}>
        Running locally on device using WebGPU/WASM. No data sent to cloud.
      </footer>
    </>
  );
}

export default App;

import React, { useRef, useState } from 'react';
import './App.css';

const EMOJI_MAP = {
  Angry: '😠',
  Disgust: '🤢',
  Fear: '😨',
  Happy: '😄',
  Sad: '😢',
  Surprise: '😲',
  Neutral: '😐',
};

function App() {
  const [page, setPage] = useState('landing');
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const videoRef = useRef();
  const canvasRef = useRef();

  // Camera functions
  const startCamera = async () => {
    setResult(null);
    setImage(null);
    setPreview(null);
    setPage('camera');
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      videoRef.current.play();
    }
  };

  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      setImage(blob);
      setPreview(URL.createObjectURL(blob));
      setPage('result');
      stopCamera();
    }, 'image/jpeg');
  };

  const stopCamera = () => {
    const stream = videoRef.current?.srcObject;
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  // Upload functions
  const handleFileChange = (e) => {
    setResult(null);
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setPage('result');
    }
  };

  // Prediction
  const handlePredict = async () => {
    if (!image) return;
    setLoading(true);
    setResult(null);
    const formData = new FormData();
    formData.append('image', image);
    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: 'Prediction failed. Try again.' });
    }
    setLoading(false);
  };

  // UI
  if (page === 'landing') {
    return (
      <div className="landing-page">
        <h1>Emotion Detection from Facial Image</h1>
        <p>Detect emotions in real-time using your webcam or by uploading a photo.</p>
        <button className="main-btn" onClick={startCamera}>Start Camera</button>
        <span style={{ margin: '0 10px' }}>or</span>
        <label className="upload-btn">
          Upload Image
          <input type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileChange} />
        </label>
      </div>
    );
  }

  if (page === 'camera') {
    return (
      <div className="camera-page">
        <h2>Camera Capture</h2>
        <video ref={videoRef} width="350" height="260" autoPlay playsInline style={{ borderRadius: 12, border: '2px solid #eee' }} />
        <div style={{ margin: '20px 0' }}>
          <button className="main-btn" onClick={capturePhoto}>Capture</button>
          <button className="secondary-btn" onClick={() => { stopCamera(); setPage('landing'); }}>Back</button>
        </div>
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>
    );
  }

  if (page === 'result') {
    return (
      <div className="result-page">
        <h2>Emotion Detection Result</h2>
        <div className="card">
          {preview && <img src={preview} alt="Preview" className="preview-img" />}
          <div className="result-section">
            {loading ? (
              <div className="loader"></div>
            ) : result ? (
              result.error ? (
                <div className="error">{result.error}</div>
              ) : (
                <div className="emotion-display">
                  <span className="emoji" role="img" aria-label={result.emotion}>{EMOJI_MAP[result.emotion] || '🙂'}</span>
                  <div className="emotion-label">{result.emotion}</div>
                  <div className="confidence-bar">
                    <div className="confidence-fill" style={{ width: `${result.confidence}%` }}></div>
                  </div>
                  <div className="confidence-text">Confidence: {result.confidence}%</div>
                </div>
              )
            ) : (
              <button className="main-btn" onClick={handlePredict}>Predict Emotion</button>
            )}
          </div>
        </div>
        <button className="secondary-btn" onClick={() => setPage('landing')}>Back to Home</button>
      </div>
    );
  }

  return null;
}

export default App;

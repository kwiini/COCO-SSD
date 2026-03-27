'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

/* ========= TF.js ========= */
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

// Detection class color mapping
const CLASS_COLORS: Record<string, string> = {
  person: '#00f5d4',
  car: '#ff006e',
  truck: '#ff006e',
  bus: '#ff006e',
  bicycle: '#fee440',
  motorcycle: '#fee440',
  dog: '#9b5de5',
  cat: '#9b5de5',
  bird: '#00bbf9',
  default: '#00f5d4',
};

interface DetectionStats {
  fps: number;
  count: number;
  detections: cocoSsd.DetectedObject[];
}

interface SystemStatus {
  model: 'loading' | 'ready' | 'error';
  camera: 'idle' | 'active' | 'error';
  detection: 'idle' | 'running' | 'paused';
  error: string | null;
}

export default function Home() {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [status, setStatus] = useState<SystemStatus>({
    model: 'loading',
    camera: 'idle',
    detection: 'idle',
    error: null,
  });
  const [stats, setStats] = useState<DetectionStats>({ fps: 0, count: 0, detections: [] });
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const detectRef = useRef(false);
  const lastDetectTimeRef = useRef(0);
  const lastFrameTimeRef = useRef<number | null>(null);
  const detectFrameRef = useRef<() => Promise<void>>(() => Promise.resolve());

  /* ========= Load Model ========= */
  useEffect(() => {
    let mounted = true;

    const loadModel = async () => {
      try {
        await tf.setBackend('cpu');
        await tf.ready();
        const loadedModel = await cocoSsd.load();
        
        if (mounted) {
          setModel(loadedModel);
          setStatus(prev => ({ ...prev, model: 'ready' }));
        }
      } catch (err) {
        console.error('Model load failed:', err);
        if (mounted) {
          setStatus(prev => ({
            ...prev,
            model: 'error',
            error: 'Model loading failed, please refresh the page',
          }));
        }
      }
    };

    loadModel();
    return () => {
      mounted = false;
    };
  }, []);

  /* ========= Responsive Canvas ========= */
  const updateCanvasSize = useCallback(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!container || !canvas || !video) return;

    const rect = container.getBoundingClientRect();
    const aspectRatio = 4 / 3;
    let width = rect.width;
    let height = width / aspectRatio;

    if (height > window.innerHeight * 0.6) {
      height = window.innerHeight * 0.6;
      width = height * aspectRatio;
    }

    canvas.width = width;
    canvas.height = height;
    video.width = width;
    video.height = height;
    
    setCanvasSize({ width: Math.round(width), height: Math.round(height) });
  }, []);

  useEffect(() => {
    // 使用 requestAnimationFrame 延迟执行，避免在 effect 中同步调用 setState
    requestAnimationFrame(() => {
      updateCanvasSize();
    });
    window.addEventListener('resize', updateCanvasSize);
    return () => window.removeEventListener('resize', updateCanvasSize);
  }, [updateCanvasSize]);

  /* ========= Camera ========= */
  const startCamera = async () => {
    if (!videoRef.current) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      
      setStatus(prev => ({ ...prev, camera: 'active', error: null }));
      updateCanvasSize();
    } catch (err) {
      console.error('Camera start failed:', err);
      setStatus(prev => ({
        ...prev,
        camera: 'error',
        error: 'Cannot access camera, please check permissions',
      }));
    }
  };

  /* ========= Canvas ========= */
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const getColorForClass = (className: string): string => {
    return CLASS_COLORS[className] || CLASS_COLORS.default;
  };

  const drawPredictions = useCallback((preds: cocoSsd.DetectedObject[]) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const validDetections: cocoSsd.DetectedObject[] = [];

    preds.forEach(prediction => {
      if (prediction.score < 0.5) return;
      validDetections.push(prediction);

      let [x, y, w, h] = prediction.bbox;
      const color = getColorForClass(prediction.class);

      // Mirror X coordinate
      x = canvas.width - x - w;

      // Clip to bounds
      x = Math.max(0, x);
      y = Math.max(0, y);
      w = Math.min(w, canvas.width - x);
      h = Math.min(h, canvas.height - y);

      // Draw detection box - HUD style
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      ctx.strokeRect(x, y, w, h);
      ctx.shadowBlur = 0;

      // Draw corner brackets
      const cornerSize = Math.min(15, w / 4, h / 4);
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      // Top-left
      ctx.moveTo(x, y + cornerSize);
      ctx.lineTo(x, y);
      ctx.lineTo(x + cornerSize, y);
      // Top-right
      ctx.moveTo(x + w - cornerSize, y);
      ctx.lineTo(x + w, y);
      ctx.lineTo(x + w, y + cornerSize);
      // Bottom-left
      ctx.moveTo(x, y + h - cornerSize);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x + cornerSize, y + h);
      // Bottom-right
      ctx.moveTo(x + w - cornerSize, y + h);
      ctx.lineTo(x + w, y + h);
      ctx.lineTo(x + w, y + h - cornerSize);
      ctx.stroke();

      // Draw label
      const label = `${prediction.class} ${(prediction.score * 100).toFixed(0)}%`;
      ctx.font = '600 12px var(--font-geist-mono), monospace';
      const textWidth = ctx.measureText(label).width;
      const padding = 8;
      const labelHeight = 20;
      const labelY = y - labelHeight - 4 < 0 ? y + h + 4 : y - labelHeight - 4;

      // Label background
      ctx.fillStyle = color + '40';
      ctx.fillRect(x, labelY, textWidth + padding * 2, labelHeight);
      
      // Label border
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.strokeRect(x, labelY, textWidth + padding * 2, labelHeight);

      // Label text
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x + padding, labelY + 14);
    });

    setStats(prev => ({ ...prev, count: validDetections.length, detections: validDetections }));
  }, []);

  /* ========= Detection Loop ========= */
  const detectFrame = useCallback(async () => {
    if (!model || !videoRef.current || !detectRef.current) return;

    const now = performance.now();
    const delta = now - (lastFrameTimeRef?.current || 0);
    lastFrameTimeRef.current = now;

    const currentFps = delta > 0 ? Math.round(1000 / delta) : 0;

    if (now - lastDetectTimeRef.current > 100) {
      lastDetectTimeRef.current = now;
      try {
        const preds = await model.detect(videoRef.current);
        drawPredictions(preds);
      } catch (err) {
        console.error('Detection error:', err);
      }
    }

    setStats(prev => ({ ...prev, fps: currentFps }));

    if (detectRef.current) {
      animationFrameRef.current = requestAnimationFrame(() => {
        detectFrameRef.current?.();
      });
    }
  }, [drawPredictions, model]);

  // Store detectFrame in ref for recursive calls
  useEffect(() => {
    detectFrameRef.current = detectFrame;
  }, [detectFrame]);

  /* ========= Camera Control ========= */
  const stopCamera = useCallback(() => {
    // Stop detection first
    detectRef.current = false;
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    // Stop camera stream
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    
    // Clear canvas
    clearCanvas();
    
    // Reset state
    setStatus({ model: 'ready', camera: 'idle', detection: 'idle', error: null });
    setStats({ fps: 0, count: 0, detections: [] });
  }, []);

  const startDetect = () => {
    if (!model || status.camera !== 'active') return;
    detectRef.current = true;
    setStatus(prev => ({ ...prev, detection: 'running' }));
    detectFrame();
  };

  const stopDetect = () => {
    detectRef.current = false;
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    setStatus(prev => ({ ...prev, detection: 'idle' }));
    setStats(prev => ({ ...prev, count: 0, detections: [] }));
    clearCanvas();
  };

  // Cleanup
  useEffect(() => {
    return () => {
      detectRef.current = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      streamRef.current?.getTracks().forEach(t => t.stop());
    };
  }, []);

  /* ========= Status Indicator Component ========= */
  const StatusIndicator = ({ state, label }: { state: string; label: string }) => {
    const getStatusClass = () => {
      switch (state) {
        case 'ready':
        case 'active':
        case 'running':
          return 'online';
        case 'loading':
          return 'loading';
        default:
          return 'offline';
      }
    };

    return (
      <div className="flex items-center gap-2 text-xs font-mono">
        <span className={`status-dot ${getStatusClass()}`} />
        <span className="text-gray-400">{label}</span>
        <span className={state === 'error' ? 'text-accent-magenta' : 'text-accent-cyan'}>
          {state.toUpperCase()}
        </span>
      </div>
    );
  };

  /* ========= UI ========= */
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-4 md:p-8">
      {/* Header */}
      <header className="w-full max-w-5xl mb-8">
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <h1 className="text-3xl md:text-5xl font-bold tracking-tighter glow-cyan mb-2">
              <span className="text-accent-cyan">COCO</span>
              <span className="text-white">-</span>
              <span className="text-accent-magenta">SSD</span>
              <span className="text-white ml-3">VISION</span>
            </h1>
            <p className="text-gray-500 font-mono text-sm tracking-widest">
              REAL-TIME OBJECT DETECTION SYSTEM
            </p>
          </div>
          
          {/* Status Panel */}
          <div className="flex flex-col gap-2 bg-black/30 p-4 border border-(--accent-cyan)/20 hud-corner">
            <div className="flex items-center gap-2 text-xs font-mono">
              <span className={`status-dot ${status.model === 'ready' ? 'online' : status.model === 'loading' ? 'loading' : 'offline'}`} />
              <span className="text-gray-400">MODEL</span>
              <span className={status.model === 'error' ? 'text-accent-magenta' : 'text-accent-cyan'}>
                {status.model.toUpperCase()}
              </span>
            </div>
            <div className="flex items-center gap-2 text-xs font-mono">
              <span className={`status-dot ${status.camera === 'active' ? 'online' : status.camera === 'idle' ? 'offline' : 'offline'}`} />
              <span className="text-gray-400">CAMERA</span>
              <span className={status.camera === 'error' ? 'text-accent-magenta' : 'text-accent-cyan'}>
                {status.camera.toUpperCase()}
              </span>
            </div>
            <div className="flex items-center gap-2 text-xs font-mono">
              <span className={`status-dot ${status.detection === 'running' ? 'online' : status.detection === 'idle' ? 'offline' : 'offline'}`} />
              <span className="text-gray-400">DETECTION</span>
              <span className={status.detection === 'paused' ? 'text-accent-yellow' : 'text-accent-cyan'}>
                {status.detection.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Error Message */}
      {status.error && (
        <div className="w-full max-w-5xl mb-4 p-4 bg-(--accent-magenta)/10 border border-accent-magenta text-accent-magenta font-mono text-sm">
          ⚠ ERROR: {status.error}
        </div>
      )}

      {/* Video Container */}
      <div 
        ref={containerRef}
        className="relative w-full max-w-5xl aspect-4/3 bg-black/50 border border-(--accent-cyan)/30 hud-corner scanlines overflow-hidden"
      >
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="absolute inset-0 w-full h-full object-cover scale-x-[-1] opacity-80"
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
        />

        {/* HUD Overlay */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Grid Overlay */}
          <div className="absolute inset-0 opacity-10"
            style={{
              backgroundImage: `
                linear-gradient(to right, var(--accent-cyan) 1px, transparent 1px),
                linear-gradient(to bottom, var(--accent-cyan) 1px, transparent 1px)
              `,
              backgroundSize: '50px 50px',
            }}
          />

          {/* Top Bar */}
          <div className="absolute top-0 left-0 right-0 h-8 bg-linear-to-b from-black/60 to-transparent flex items-center justify-between px-4">
            <span className="text-accent-cyan font-mono text-xs tracking-widest">
              FEED: CAM_01
            </span>
            <span className="text-accent-cyan font-mono text-xs">
              {new Date().toLocaleTimeString()}
            </span>
          </div>

          {/* Stats Panel */}
          <div className="absolute top-12 left-4 bg-black/70 border border-(--accent-cyan)/30 p-3 font-mono text-xs">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-gray-500">FPS:</span>
              <span className={`text-accent-cyan ${stats.fps > 0 ? 'glow-cyan' : ''}`}>
                {stats.fps.toString().padStart(2, '0')}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-gray-500">OBJ:</span>
              <span className={`text-accent-magenta ${stats.count > 0 ? 'glow-magenta' : ''}`}>
                {stats.count.toString().padStart(2, '0')}
              </span>
            </div>
          </div>

          {/* Detection List */}
          {stats.detections.length > 0 && (
            <div className="absolute top-12 right-4 bg-black/70 border border-(--accent-cyan)/30 p-3 max-w-[200px]">
              <div className="text-accent-cyan font-mono text-xs mb-2 border-b border-(--accent-cyan)/30 pb-1">
                DETECTED
              </div>
              <div className="space-y-1">
                {stats.detections.slice(0, 5).map((det, i) => (
                  <div key={i} className="flex items-center justify-between text-xs font-mono">
                    <span style={{ color: getColorForClass(det.class) }}>
                      {det.class}
                    </span>
                    <span className="text-gray-500">
                      {(det.score * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
                {stats.detections.length > 5 && (
                  <div className="text-gray-500 text-xs font-mono">
                    +{stats.detections.length - 5} more
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Center Crosshair */}
          {status.detection === 'running' && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="w-20 h-20 border border-(--accent-cyan)/20 rounded-full flex items-center justify-center">
                <div className="w-1 h-1 bg-accent-cyan rounded-full" />
              </div>
            </div>
          )}

          {/* Bottom Info */}
          <div className="absolute bottom-4 left-4 right-4 flex justify-between items-end">
            <div className="font-mono text-xs text-gray-500">
              <div>RES: {canvasSize.width}x{canvasSize.height}</div>
              <div>THRESHOLD: 50%</div>
            </div>
            
            {!status.detection && status.camera === 'active' && (
              <div className="text-accent-yellow font-mono text-sm animate-pulse">
                ▶ READY TO DETECT
              </div>
            )}
          </div>
        </div>

        {/* Loading Overlay */}
        {status.model === 'loading' && (
          <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center loading-scan">
            <div className="text-accent-cyan font-mono text-lg mb-4 glow-cyan">
              INITIALIZING MODEL...
            </div>
            <div className="w-64 h-1 bg-gray-800 overflow-hidden">
              <div className="h-full bg-accent-cyan animate-pulse w-3/4" />
            </div>
          </div>
        )}
      </div>

      {/* Control Panel */}
      <div className="w-full max-w-5xl mt-8">
        <div className="flex flex-wrap justify-center gap-4 p-6 bg-black/20 border border-(--accent-cyan)/10 hud-corner">
          <button
            onClick={startCamera}
            disabled={status.model !== 'ready' || status.camera === 'active'}
            className={`btn-cyber ${status.camera === 'active' ? 'active' : ''}`}
          >
            <span className="mr-2">◉</span>
            启动摄像头
          </button>
          
          <button
            onClick={startDetect}
            disabled={status.camera !== 'active' || status.detection === 'running'}
            className={`btn-cyber ${status.detection === 'running' ? 'active' : ''}`}
          >
            <span className="mr-2">▶</span>
            开始检测
          </button>
          
          <button
            onClick={stopDetect}
            disabled={status.detection !== 'running'}
            className="btn-cyber"
          >
            <span className="mr-2">⏸</span>
            暂停检测
          </button>
          
          <button
            onClick={stopCamera}
            disabled={status.camera === 'idle'}
            className="btn-cyber btn-cyber-danger"
          >
            <span className="mr-2">◉</span>
            关闭系统
          </button>
        </div>

        {/* Legend */}
        <div className="mt-6 flex flex-wrap justify-center gap-6 text-xs font-mono text-gray-500">
          {Object.entries(CLASS_COLORS).filter(([k]) => k !== 'default').map(([className, color]) => (
            <div key={className} className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: color }} />
              <span className="uppercase">{className}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-12 text-center text-gray-600 text-xs font-mono">
        <p>POWERED BY TENSORFLOW.JS • COCO-SSD MODEL</p>
        <p className="mt-1">© 2025 COCO-SSD VISION</p>
      </footer>
    </main>
  );
}

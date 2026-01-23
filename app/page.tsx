'use client';

import { useEffect, useRef, useState } from 'react';

/* ========= TF.js ========= */
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

export default function Home() {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [loading, setLoading] = useState(true);

  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [stats, setStats] = useState({ fps: 0, count: 0 });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const detectRef = useRef(false);
  const lastDetectTimeRef = useRef(0);
  const lastFrameTimeRef = useRef(performance.now());

  /* ========= 加载模型 ========= */
  useEffect(() => {
    let mounted = true;

    const load = async () => {
      await tf.setBackend('cpu');
      await tf.ready();
      const m = await cocoSsd.load();
      if (mounted) {
        setModel(m);
        setLoading(false);
      }
    };

    load();
    return () => {
      mounted = false;
    };
  }, []);

  /* ========= 摄像头 ========= */
  const startCamera = async () => {
    if (!videoRef.current) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });

    streamRef.current = stream;
    videoRef.current.srcObject = stream;
    await videoRef.current.play();
    setIsCameraOn(true);
  };

  const stopCamera = () => {
    stopDetect();
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    clearCanvas();
    setIsCameraOn(false);
  };

  /* ========= Canvas ========= */
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const drawPredictions = (preds: cocoSsd.DetectedObject[]) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let validCount = 0;

    preds.forEach(p => {
      if (p.score < 0.5) return;
      validCount++;

      let [x, y, w, h] = p.bbox;

      // 镜像 X
      x = canvas.width - x - w;

      // 裁剪
      x = Math.max(0, x);
      y = Math.max(0, y);
      w = Math.min(w, canvas.width - x);
      h = Math.min(h, canvas.height - y);

      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      const label = `${p.class} ${(p.score * 100).toFixed(1)}%`;
      ctx.font = '14px Arial';

      const textWidth = ctx.measureText(label).width;
      const textHeight = 16;
      const textY = y - textHeight < 0 ? y + textHeight + 4 : y - 4;

      ctx.fillStyle = '#00ff00';
      ctx.fillRect(x, textY - textHeight, textWidth + 6, textHeight);
      ctx.fillStyle = '#000';
      ctx.fillText(label, x + 3, textY - 4);
    });

    setStats(s => ({ ...s, count: validCount }));
  };

  /* ========= 检测循环 ========= */
  const detectFrame = async () => {
    if (!model || !videoRef.current || !detectRef.current) return;

    const now = performance.now();
    const delta = now - lastFrameTimeRef.current;
    lastFrameTimeRef.current = now;

    setStats(s => ({
      ...s,
      fps: Math.round(1000 / delta),
    }));

    if (now - lastDetectTimeRef.current > 200) {
      lastDetectTimeRef.current = now;
      const preds = await model.detect(videoRef.current);
      drawPredictions(preds);
    }

    requestAnimationFrame(detectFrame);
  };

  const startDetect = () => {
    if (!model || !isCameraOn) return;
    detectRef.current = true;
    setIsDetecting(true);
    detectFrame();
  };

  const stopDetect = () => {
    detectRef.current = false;
    setIsDetecting(false);
    setStats(s => ({ ...s, count: 0 }));
    clearCanvas();
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  /* ========= UI ========= */
  return (
    <main className="min-h-screen bg-gray-100 flex flex-col items-center pt-8">
      <h1 className="text-2xl font-semibold">实时目标检测</h1>

      <div className="relative mt-6 w-[640px] h-[480px] bg-black rounded overflow-hidden">
        <video
          ref={videoRef}
          width={640}
          height={480}
          autoPlay
          muted
          playsInline
          className="absolute inset-0 w-full h-full object-cover scale-x-[-1]"
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          className="absolute inset-0"
        />

        {/* 状态面板 */}
        <div className="absolute left-2 top-2 bg-black/60 text-white text-xs px-2 py-1 rounded">
          <div>FPS: {stats.fps}</div>
          <div>Objects: {stats.count}</div>
        </div>

        {!isDetecting && isCameraOn && (
          <div className="absolute inset-0 flex items-center justify-center text-white text-sm">
            点击「开始检测」
          </div>
        )}

        {isDetecting && stats.count === 0 && (
          <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-white text-xs opacity-80">
            未检测到目标
          </div>
        )}
      </div>

      <p className="mt-3 text-sm text-gray-600">
        {loading ? '模型加载中…' : '模型就绪'}
      </p>

      <div className="mt-5 flex gap-3">
        <button
          onClick={startCamera}
          disabled={loading || isCameraOn}
          className="px-4 py-2 rounded bg-green-600 text-white disabled:opacity-40"
        >
          打开摄像头
        </button>
        <button
          onClick={startDetect}
          disabled={!isCameraOn || isDetecting}
          className="px-4 py-2 rounded bg-blue-600 text-white disabled:opacity-40"
        >
          开始检测
        </button>
        <button
          onClick={stopDetect}
          disabled={!isDetecting}
          className="px-4 py-2 rounded bg-yellow-500 text-white disabled:opacity-40"
        >
          停止检测
        </button>
        <button
          onClick={stopCamera}
          disabled={!isCameraOn}
          className="px-4 py-2 rounded bg-red-600 text-white disabled:opacity-40"
        >
          关闭摄像头
        </button>
      </div>
    </main>
  );
}

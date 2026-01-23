# Real-time Object Detection with COCO-SSD (TensorFlow.js)

一个基于 **Next.js + React + TensorFlow.js** 的实时目标检测 Demo。

项目通过调用浏览器摄像头，使用 **COCO-SSD 模型** 对视频流进行实时推理，并在画面中绘制目标边框、类别和置信度，同时展示 FPS 与检测数量。

> 这是一个简单的模型调用示例。

## 功能特性

- 🎥 浏览器摄像头实时视频流
- 📦 基于 COCO-SSD 的目标检测
- 🟩 实时绘制检测框、类别与置信度
- 🔁 正确处理摄像头镜像（左右不反）
- 📊 FPS（帧率）与目标数量统计
- 🕹️ 完整的用户控制（开始 / 停止检测、关闭摄像头）
- ⚙️ 资源释放完善，避免内存与摄像头占用问题

## 技术栈

- **Next.js (App Router)**
- **React 18**
- **Tailwind CSS**
- **TensorFlow.js**
- **@tensorflow-models/coco-ssd**
- **HTML5 getUserMedia API**

## 安装与运行

### 克隆项目

```bash
git clone https://github.com/kwiini/COCO-SSD.git
```

### 安装依赖

```bash
npm install
# 或
pnpm install
```

### 启动开发环境

```bash
npm run dev
```

浏览器访问：

```
http://localhost:3000
```

## 核心实现说明

### 1. 模型加载

- 使用 `@tensorflow-models/coco-ssd`
- 模型只加载一次，避免重复初始化
- 显式指定 **CPU backend**，保证稳定性

```ts
await tf.setBackend('cpu');
await tf.ready();
const model = await cocoSsd.load();
```

------

### 2. 摄像头接入

- 使用 `navigator.mediaDevices.getUserMedia`
- 分辨率限制为 `640x480`，平衡性能与清晰度
- 组件卸载 / 停止时正确释放流资源

------

### 3. 镜像与坐标处理（重点）

- **视频使用 CSS 镜像**（`scaleX(-1)`），符合用户直觉
- **Canvas 不做镜像**，避免文字反转
- 手动修正 bbox 的 X 坐标：

```ts
mirroredX = canvasWidth - x - width;
```

这是一个常见但容易踩坑的前端 AI 细节。

------

### 4. 实时检测与性能控制

- 使用 `requestAnimationFrame` 进行渲染循环
- 通过时间间隔节流检测频率（≈ 5 FPS 推理）
- FPS 采用轻量计算方式，避免性能负担

## 界面说明

- **FPS**：当前页面渲染帧率
- **Objects**：当前帧中检测到的目标数量
- **开始检测**：开启模型推理
- **停止检测**：暂停推理但保留摄像头
- **关闭摄像头**：释放摄像头与资源

## 可扩展方向

- 📸 截图并保存检测结果
- 📱 切换前 / 后摄像头（移动端）
- 🧵 使用 Web Worker / WASM backend 优化性能
- 📦 封装为通用 React Hook
- 🌐 部署到 Vercel 作为在线 Demo

## 注意事项

- 首次访问需要 **允许浏览器摄像头权限**
- 建议使用 Chrome / Edge 等现代浏览器
- 本项目为前端推理示例，不涉及后端服务


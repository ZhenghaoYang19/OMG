# OMG (Online Meeting Guru)

[English](README.md) | 简体中文

OMG 是一个用于处理会议录像(或屏幕截图)的工具，可以提取关键帧并生成文字记录。它使用 Gradio 构建了一个轻量化的现代界面。

## 功能特点

- 两种输入模式：
  - 视频文件上传：处理预录制的视频
  - 屏幕捕获：实时捕获和处理屏幕内容
- 基于内容相似度提取视频关键帧
- 使用 OpenAI Whisper 模型生成文字记录
- 导出前支持手动编辑提取的图片和文字记录
- 支持导出 PDF、音频和文本文件
- 使用 Gradio 构建的轻量化的现代界面
- 极致压缩，实测将2.48G视频压缩到321M

## 系统要求

- Python>=3.8 (在3.12下测试) 
- CUDA 兼容的 GPU（可选，用于加快处理速度）
- FFmpeg（用于音频处理）

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/ZhenghaoYang19/omg.git
cd omg
```

2. 创建虚拟环境并安装依赖：

   ### 使用 uv（推荐）
   ```bash
   # 安装 uv
   pip install uv
   
   # 创建并激活虚拟环境
   uv venv
   .venv\Scripts\activate  # Windows系统
   source .venv/bin/activate  # Linux/macOS系统
   
   # 安装依赖
   uv pip install -r requirements.txt
   ```

   ### 使用 pip
   ```bash
   pip install -r requirements.txt
   ```

3. 安装 FFmpeg：
- Windows：从 [FFmpeg 官网](https://ffmpeg.org/download.html) 下载
- Linux：`sudo apt-get install ffmpeg`
- macOS：`brew install ffmpeg`

## 使用方法

1. 启动 Web 界面：
```bash
python app.py
```

2. 在浏览器中访问 `http://localhost:7860`

3. 选择输入模式：

   ### 视频上传
   - 上传视频文件
   - 调整配置参数（可选）
   - 点击"Process Video"
   - 等待处理完成
   - 切换到"Results & Export"标签页查看结果

   ### 屏幕捕获
   - 选择捕获类型（显示器或窗口）
   - 显示器捕获：从下拉菜单选择显示器
   - 窗口捕获：输入窗口标题（或部分标题）(例如：chrome, edge, 腾讯会议)
   - 调整配置参数（可选）
   - 点击"Start Capture"开始捕获
   - 完成后点击"Stop Capture"停止捕获
   - 切换到"Results & Export"标签页查看结果

4. 导出结果：
   - 选择要导出的内容（PDF/音频/文字记录）
   - 点击"Export Selected Files"
   - 下载导出的文件

## 配置说明

你可以调整以下参数：

- **相似度阈值**（0.0-1.0）：控制帧与帧之间需要多大的差异才被视为关键帧，建议值在 0.6 到 0.7 之间
- **每秒采样帧数**：每秒钟采样的帧数，建议值在0.2-5之间，如果采样过多就降低，反之亦然
- **开始/结束时间**（仅视频上传）：只处理视频的特定部分
- **ASR 模型**：选择不同的 Whisper 模型
- **ASR 设备**：选择处理设备（建议：auto）
- **帧比较方法**：选择不同的帧比较算法

默认值可以在 `config.json` 中修改。

## 项目结构

```
omg/
├── app.py              # Web 界面
├── omg.py              # 核心处理逻辑
├── utils/
│   ├── compare.py      # 帧比较函数
│   └── images2pdf.py   # PDF 生成
├── config.json         # 配置文件
├── requirements.txt    # Python 依赖
└── output/             # 处理结果
```

## 输出目录结构

视频上传模式：
```
output/
└── video_name/
    ├── images/        # 提取的关键帧
    ├── audio.wav      # 提取的音频
    ├── transcript.txt # 生成的文字记录
    └── slides.pdf     # 生成的 PDF（可选）
```

屏幕捕获模式：
```
output/
└── screen_capture/
    └── YYYYMMDD_HHMMSS/  # 捕获的时间戳
        ├── images/        # 捕获的帧
        ├── audio.wav      # 录制的音频
        ├── transcript.txt # 生成的文字记录
        └── slides.pdf     # 生成的 PDF（可选）
```

## 许可证

本项目采用 [Apache License 2.0](LICENSE) - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- [wudududu/extract-video-ppt](https://github.com/wudududu/extract-video-ppt/tree/master) 提供了灵感和参考
- [Gradio](https://www.gradio.app/) 提供了 Web 界面框架
- [OpenAI](https://openai.com/) 提供了 Whisper ASR 模型
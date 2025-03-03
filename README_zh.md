# TransMeet

[English](README.md) | 简体中文

TransMeet 是一个用于处理会议录像的工具，可以提取关键帧并生成文字记录。它提供了基于 Gradio 构建的友好的 Web 界面。

## 功能特点

- 基于内容相似度提取视频关键帧
- 使用 OpenAI Whisper 模型生成文字记录
- 支持导出 PDF、音频和文本文件
- 用户友好的 Web 界面
- 可配置的处理参数
- 支持多种视频格式

## 系统要求

- Python 3.8 或更高版本
- CUDA 兼容的 GPU（可选，用于加速处理）
- FFmpeg（用于音频处理）

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/ZhenghaoYang19/transmeet.git
cd transmeet
```

2. 安装依赖：
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

3. 处理视频：
   - 上传视频文件
   - 调整配置参数（可选）
   - 点击"Process Video"
   - 等待处理完成
   - 切换到"Results & Export"标签页查看结果

4. 导出结果：
   - 选择要导出的内容（PDF/音频/文字记录）
   - 点击"Export Selected Files"
   - 下载导出的文件

## 配置说明

你可以调整以下参数：

- **相似度阈值**（0.0-1.0）：控制帧与帧之间需要多大的差异才被认为是关键帧，建议值在 0.5 到 0.8 之间
- **每秒采样帧数**：每秒钟采样的帧数
- **开始/结束时间**：只处理视频的特定部分
- **ASR 模型**：选择不同的 Whisper 模型
- **ASR 设备**：选择处理设备（建议：auto）
- **帧比较方法**：选择不同的帧比较算法

默认值可以在 `config.json` 中修改。

## 项目结构

```
transmeet/
├── app.py              # Web 界面
├── transmeet.py        # 核心处理逻辑
├── compare.py          # 帧比较函数
├── images2pdf.py       # PDF 生成
├── config.json         # 配置文件
├── requirements.txt    # Python 依赖
└── output/            # 处理结果
```

## 输出目录结构

```
output/
└── video_name/
    ├── images/        # 提取的关键帧
    ├── audio.wav      # 提取的音频
    ├── transcript.txt # 生成的文字记录
    └── slides.pdf     # 生成的 PDF（可选）
```

## 开源许可

本项目采用 [MIT 许可证](LICENSE) - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- [wudududu/extract-video-ppt](https://github.com/wudududu/extract-video-ppt/tree/master) 提供了灵感和参考
- [Gradio](https://www.gradio.app/) 提供了 Web 界面框架
- [OpenAI](https://openai.com/) 提供了 Whisper ASR 模型 
# GPU Monitor Telegram Setup Guide

## Quick Start

### 1. Get Telegram Bot Token
1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow instructions
3. Copy the bot token (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### 2. Get Your Chat ID
1. Search for `@userinfobot` on Telegram
2. Start a chat and it will show your chat ID (a number like: `123456789`)

### 3. Configure the Script
Edit `gpu_monitor_telegram.py`:
```python
TELEGRAM_BOT_TOKEN = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"  # Your bot token
TELEGRAM_CHAT_ID = "123456789"                                # Your chat ID
```

### 4. Install Dependencies
```bash
pip install requests
```

### 5. Run the Monitor
```bash
python gpu_monitor_telegram.py
```

## Configuration Options

```python
# Alert threshold (GB)
VRAM_THRESHOLD_GB = 25.0

# Polling interval (seconds)
POLL_INTERVAL = 1

# Cooldown between alerts (seconds) - prevents spam
ALERT_COOLDOWN = 300  # 5 minutes
```

## Features

✅ Monitors GPU 0 and GPU 1  
✅ Polls every second  
✅ Sends Telegram alert when VRAM > 25GB  
✅ 5-minute cooldown to prevent spam  
✅ Edge-triggered (only alerts once when threshold crossed)  
✅ Background status updates every 10 seconds  

## Example Output

```
============================================================
GPU Memory Monitor with Telegram Alerts
============================================================
Threshold: 25.0 GB
Poll interval: 1s
Alert cooldown: 300s
Monitoring GPUs: 0, 1
============================================================

[07:35:10] GPU 0: 12.45 GB
[07:35:10] GPU 1: 38.23 GB

🚨 ALERT: GPU 1 exceeded threshold!
   VRAM: 38.23 GB > 25.0 GB
   ✓ Telegram alert sent!
```

## Alert Message Format

You'll receive a Telegram message like:

```
🚨 GPU Alert! 🚨

📅 Time: 2026-01-20 07:35:10
🎮 GPU: 1
💾 VRAM: 38.23 GB
⚠️  Threshold: 25.0 GB

Status: Memory usage exceeded threshold!
```

## Run in Background

```bash
# Run in background with nohup
nohup python gpu_monitor_telegram.py > gpu_monitor.log 2>&1 &

# Or use screen/tmux
screen -S gpu_monitor
python gpu_monitor_telegram.py
# Press Ctrl+A, D to detach
```

## Troubleshooting

### "Failed to send Telegram message"
- Check your bot token is correct
- Check your chat ID is correct
- Make sure you've started a chat with your bot (send `/start`)

### "No GPU found"
- Make sure CUDA is available: `nvidia-smi`
- Check PyTorch can see GPUs: `python -c "import torch; print(torch.cuda.is_available())"`

### Monitor different GPUs
Edit line 139 to change which GPUs to monitor:
```python
for gpu_id in [0, 1, 2, 3]:  # Monitor GPUs 0, 1, 2, 3
```

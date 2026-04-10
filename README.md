# VeloGuard 🛡️
**Adaptive AI-Driven API Rate Limiting Using Bidirectional LSTM with Temporal Attention**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-00a393.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Redis](https://img.shields.io/badge/Redis-7.0%2B-dc382d.svg)](https://redis.io/)

## Stop Guessing. Start Predicting.
Static rate limiters are dead. Classical algorithms like token buckets and fixed windows can't tell the difference between a DDoS attack and a server that is just recovering from a traffic spike—resulting in up to 35% of legitimate users getting falsely blocked during queue drainage.

**VeloGuard** is an AI-driven, closed-loop API rate limiter designed for FastAPI. Powered by a Bidirectional LSTM with temporal attention, it doesn't just count instantaneous requests; it reads the 60-second temporal trajectory of your server's telemetry. It predicts crashes *before* they happen and automatically applies a 4-tier graduated throttling policy via Redis in under 1 second.

![Live Demo](fig6_live_demo.png)

## ✨ Key Features
* 🧠 **BiLSTM with Temporal Attention:** Learns the characteristic signatures of imminent crashes vs. genuine recovery, completely eliminating false-positive 429 errors.
* 📉 **Graduated Throttling:** Drops limits proportionally based on AI probability thresholds rather than hard-blocking all traffic:
  * 🟢 **SAFE:** 100 RPS 
  * 🟡 **WARNING:** 60 RPS 
  * 🟠 **CRITICAL:** 20 RPS 
  * 🔴 **BLACKHOLE:** 0 RPS 
* ⚡ **Sub-Millisecond Overhead:** A custom Starlette middleware intercepts traffic and enforces AI-derived Redis limits instantly.
* 🧪 **Built-in DDoS Simulator:** Includes a 6-vector Locust state machine (Flash Crowd, Sniper, Wave, Ramp, Pulse, Stochastic) to test your own infrastructure.

---

## 🏗️ System Architecture
VeloGuard evaluates 60-second rolling windows of four key telemetry features:
1. **RPS (Requests Per Second)**
2. **Average Response Latency**
3. **CPU Utilization**
4. **Memory Utilization**

The AI Control Plane processes this window through the PyTorch model, outputs a crash probability, and updates the `global_ai_limit` key in Redis. The FastAPI middleware reads this key on every incoming request.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
* Python 3.10+
* Docker & Docker Compose (for Redis)

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/hardikahujaa/Load-Guard-.git](https://github.com/hardikahujaa/Load-Guard-.git)
cd Load-Guard-
pip install -r requirements.txt
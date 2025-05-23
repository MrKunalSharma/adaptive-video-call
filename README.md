# Intelligent Video Call Optimizer

![Demo](docs/images/demo.gif)

A WebRTC-based video calling application that uses machine learning to optimize video quality based on real-time network conditions.

## Features

- **Room-based Video Calling**: Create or join private video chat rooms
- **Network-Aware Optimization**: Automatically adjusts video quality for optimal experience
- **Real-time Statistics**: Monitor bandwidth, latency, jitter, and packet loss
- **Auto Recovery**: Intelligent reconnection when network conditions change
- **Cross-platform Support**: Works on desktop and mobile browsers

## Technology Stack

- **Frontend**: JavaScript, WebRTC, WebSockets
- **Backend**: Python, Flask, asyncio, websockets
- **ML/AI**: TensorFlow for network condition prediction
- **DevOps**: Docker, GitHub Actions

## Architecture
The system uses WebRTC for peer-to-peer communication and a central signaling server for connection establishment. The ML model analyzes network metrics to predict optimal video quality settings.

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+ (for development)
- Modern web browser with WebRTC support

### Installation

1. Clone the repository

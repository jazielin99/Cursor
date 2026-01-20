# PSA Card Grader iOS App

AI-powered PSA card grading prediction app for iOS.

## Features

- 📸 **Camera Capture**: Take photos directly from your phone
- 🖼️ **Photo Library**: Select existing images from your library
- 🤖 **AI Prediction**: Get instant PSA grade predictions (1-10)
- 📊 **Confidence Scores**: See prediction confidence percentages
- 📝 **Grading Notes**: Detailed analysis of centering, corners, surface
- ⚡ **Fast Results**: Predictions in seconds

## Architecture

```
┌─────────────────────┐
│     iOS App         │
│   (SwiftUI)         │
│                     │
│  ┌───────────────┐  │
│  │ Camera/Photos │  │
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │  API Client   │  │
│  └───────┬───────┘  │
└──────────┼──────────┘
           │ HTTPS
           │
┌──────────▼──────────┐
│   Backend API       │
│   (FastAPI)         │
│                     │
│  ┌───────────────┐  │
│  │ Feature       │  │
│  │ Extraction    │  │
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │ R Prediction  │  │
│  │ Engine        │  │
│  └───────────────┘  │
└─────────────────────┘
```

## Setup

### 1. Backend Server

First, start the API server:

```bash
# Install dependencies
pip install fastapi uvicorn python-multipart

# Navigate to backend directory
cd ios_app/backend

# Start server
python api_server.py
```

The server will start on `http://localhost:8000`.

### 2. iOS App (Xcode)

1. Open Xcode
2. Create a new iOS App project
3. Choose "SwiftUI" as the interface
4. Copy the Swift files from `ios_app/PSAGrader/` to your project:
   - `PSAGraderApp.swift`
   - `ContentView.swift`
   - `GradingViewModel.swift`
   - `CameraView.swift`
   - `SettingsView.swift`
5. Add the camera permission to Info.plist (already included)
6. Build and run on your device

### 3. Configuration

In the app settings, configure the API URL:
- **Local testing**: `http://YOUR_MAC_IP:8000`
- **Production**: Your deployed server URL

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Server status |
| `/predict` | POST | Predict grade from image |
| `/grades` | GET | List possible grades |

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@card.jpg"
```

### Example Response

```json
{
  "success": true,
  "grade": "PSA_9",
  "grade_confidence": 0.782,
  "tier": "NearMint_8_10",
  "tier_confidence": 0.923,
  "grade_probabilities": {
    "PSA_8": 0.15,
    "PSA_9": 0.78,
    "PSA_10": 0.07
  },
  "grading_notes": {
    "centering": "Good centering: 52/48 L/R, 48/52 T/B",
    "summary": "Near Mint-Mint condition"
  },
  "upgrade_hint": null
}
```

## Development

### Requirements

- **iOS App**: Xcode 14+, iOS 16+
- **Backend**: Python 3.8+, R 4.0+

### Project Structure

```
ios_app/
├── README.md
├── backend/
│   └── api_server.py       # FastAPI server
└── PSAGrader/
    ├── PSAGraderApp.swift  # App entry point
    ├── ContentView.swift   # Main UI
    ├── GradingViewModel.swift  # State management
    ├── CameraView.swift    # Camera/photo picker
    ├── SettingsView.swift  # Settings UI
    └── Info.plist          # App configuration
```

### Running Locally

1. Start the backend server on your Mac
2. Find your Mac's local IP address:
   ```bash
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```
3. In the iOS app settings, set the API URL to `http://YOUR_MAC_IP:8000`
4. Run the app on your iPhone (same WiFi network)

### Deploying the Backend

For production, deploy the backend to a cloud server:

```bash
# Using Docker
docker build -t psa-grader-api .
docker run -p 8000:8000 psa-grader-api

# Using Gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

## Troubleshooting

### "Network error" in app
- Check that the backend server is running
- Verify the API URL in settings
- Ensure your phone is on the same WiFi network as the server

### Camera not working
- Check that camera permissions are granted in iOS Settings
- Restart the app

### Slow predictions
- First prediction may be slow (model loading)
- Subsequent predictions should be faster
- Consider using a more powerful server for production

## Future Improvements

- [ ] Offline mode with Core ML model
- [ ] Batch prediction support
- [ ] History of past predictions
- [ ] Share predictions to social media
- [ ] Card collection tracking

## License

MIT License

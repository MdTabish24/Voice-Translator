<<<<<<< HEAD
# Voice-Translator
=======
# Real-Time Translator

A web-based real-time language translator with voice and camera modes for educational demonstrations.

## Features

- **Voice Mode**: Speak in your native language and get real-time translation with audio output
- **Camera Mode**: Point camera at objects to get their names translated in real-time
- **Free Tools Only**: Uses only free APIs and libraries, no paid services required
- **Local Deployment**: Runs completely locally for presentations and demos

## Tech Stack

### Backend
- Python Flask server
- googletrans library for translation
- CORS enabled for frontend communication

### Frontend
- HTML/CSS/JavaScript
- Web Speech API for speech recognition
- SpeechSynthesis API for text-to-speech
- TensorFlow.js with Coco-SSD for object detection

## Quick Start

### Prerequisites
- Python 3.7 or higher
- Modern web browser (Chrome, Firefox, Edge recommended)
- Internet connection (required for translation service)

### Installation & Running

1. **Clone or download the project files**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Flask backend:**
   ```bash
   python app.py
   ```
   
   You should see output like:
   ```
   INFO - Starting Real-Time Translator API server...
   INFO - Translation service initialized successfully
   * Running on http://127.0.0.1:5000
   ```

4. **Test the backend (optional):**
   ```bash
   python test_app.py
   ```

5. **Open the frontend:**
   - Open your web browser
   - Navigate to: `http://127.0.0.1:5000/` or `http://127.0.0.1:5000/static/index.html`
   - Grant microphone and camera permissions when prompted

### Testing the Application

#### Backend API Testing
Test the backend endpoints directly:

1. **Health check:**
   ```bash
   curl http://127.0.0.1:5000/health
   ```

2. **Translation test:**
   ```bash
   curl -X POST http://127.0.0.1:5000/translate \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world", "target_language": "es"}'
   ```

3. **Languages list:**
   ```bash
   curl http://127.0.0.1:5000/languages
   ```

### Usage

#### Voice Mode Testing
1. Click "Voice Mode" tab
2. Select target language (e.g., Spanish)
3. Click "Start Listening"
4. Say something like "Hello, how are you?"
5. Verify:
   - Original text appears in left panel
   - Translation appears in right panel
   - Audio plays the translation

#### Camera Mode Testing
1. Click "Camera Mode" tab
2. Select target language (e.g., French)
3. Click "Start Camera"
4. Point camera at common objects (cup, book, phone, etc.)
5. Verify:
   - Green bounding boxes appear around detected objects
   - Object names and translations appear below video
   - Audio plays the translated object names

#### Mode Switching Testing
1. Start with Voice Mode active and listening
2. Switch to Camera Mode
3. Verify voice recognition stops
4. Start camera, then switch back to Voice Mode
5. Verify camera stops and voice mode reactivates

## Project Structure

```
real-time-translator/
├── app.py              # Flask backend server
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── static/
    ├── index.html     # Main HTML interface
    ├── style.css      # Responsive CSS styling
    └── script.js      # JavaScript application logic
```

## Browser Permissions

The application requires:
- **Microphone access** for Voice Mode
- **Camera access** for Camera Mode

Grant permissions when prompted by your browser.

## Troubleshooting

### Common Issues

#### Backend Issues
- **Backend not starting**: 
  - Ensure Python 3.7+ is installed: `python --version`
  - Install dependencies: `pip install -r requirements.txt`
  - Check if port 5000 is available

- **Translation not working**: 
  - Check internet connection (googletrans requires internet)
  - Verify backend is running: visit `http://127.0.0.1:5000/health`
  - Check backend logs for error messages

#### Frontend Issues
- **Camera/microphone not working**: 
  - Grant permissions when browser prompts
  - Use HTTPS in production (required for camera/mic access)
  - Try Chrome or Edge browsers for best compatibility

- **Object detection not loading**:
  - Check internet connection (TensorFlow.js models download from CDN)
  - Wait for "Camera active - Detecting objects..." status
  - Try refreshing the page

- **CORS errors**: 
  - Ensure backend is running on port 5000
  - Check that you're accessing via `http://127.0.0.1:5000/static/index.html`

#### Browser Compatibility
- **Chrome/Edge**: Full support for all features
- **Firefox**: Voice and camera work, some TTS voices may be limited
- **Safari**: Basic support, may have voice recognition limitations
- **Mobile browsers**: Limited camera/microphone support

### Error Messages
The application provides helpful error messages:
- **Red error bars**: Show specific issues with retry buttons
- **Status indicators**: Show current state of voice/camera modes
- **Console logs**: Check browser developer tools for detailed errors

## Development Notes

This is a demonstration application built for educational purposes.

### Architecture Overview
- **Backend**: Flask server with googletrans library
- **Frontend**: Vanilla JavaScript with Web APIs
- **Translation**: Google Translate (free tier via googletrans)
- **Object Detection**: TensorFlow.js with COCO-SSD model
- **Speech**: Browser's built-in Web Speech API and SpeechSynthesis

### Key Features Implemented
✅ Voice Mode with continuous speech recognition  
✅ Camera Mode with real-time object detection  
✅ Text-to-speech in multiple languages  
✅ Responsive design for presentations  
✅ Comprehensive error handling  
✅ Long text chunking for better translation  
✅ Visual feedback and status indicators  
✅ Browser compatibility checks  

### For Production Use, Consider:
- Adding authentication and rate limiting
- Using more robust translation services (Google Cloud Translate API)
- Implementing proper error logging and monitoring
- Adding comprehensive testing suite
- Using HTTPS for secure camera/microphone access
- Implementing offline fallbacks
- Adding user preferences and settings storage

### Performance Notes
- Object detection runs every 1 second to balance accuracy and performance
- Translation requests are debounced to avoid API spam
- TensorFlow.js model is cached by the browser after first load
- Long text is automatically chunked for better translation quality
>>>>>>> b81045f (Initial commit)

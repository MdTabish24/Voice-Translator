// Real-Time Translator - Complete Implementation
const APP_CONFIG = {
    API_BASE: window.location.hostname.includes('onrender.com') ? '' : 'http://127.0.0.1:5000',
    DETECTION_INTERVAL: 1000,
    MAX_TEXT_LENGTH: 5000,
    SPEECH_TIMEOUT: 10000
};

class RealTimeTranslator {
    constructor() {
        this.state = {
            currentMode: 'voice',
            targetLanguage: 'es',
            isListening: false,
            isCameraActive: false,
            isOcrCameraActive: false,
            recognition: null,
            detectionModel: null,
            detectionInterval: null,
            currentStream: null,
            ocrStream: null
        };

        this.elements = {
            // Mode buttons
            voiceMode: document.getElementById('voiceMode'),
            textMode: document.getElementById('textMode'),
            cameraMode: document.getElementById('cameraMode'),
            ocrMode: document.getElementById('ocrMode'),
            
            // Interfaces
            voiceInterface: document.getElementById('voiceInterface'),
            textInterface: document.getElementById('textInterface'),
            cameraInterface: document.getElementById('cameraInterface'),
            ocrInterface: document.getElementById('ocrInterface'),
            
            // Common elements
            targetLanguage: document.getElementById('targetLanguage'),
            errorMessage: document.getElementById('errorMessage'),
            
            // Voice mode elements
            startListening: document.getElementById('startListening'),
            stopListening: document.getElementById('stopListening'),
            originalText: document.getElementById('originalText'),
            translatedText: document.getElementById('translatedText'),
            voiceStatus: document.getElementById('voiceStatus'),
            
            // Text mode elements
            textInput: document.getElementById('textInput'),
            translateText: document.getElementById('translateText'),
            clearText: document.getElementById('clearText'),
            textTranslationOutput: document.getElementById('textTranslationOutput'),
            
            // Camera mode elements
            startCamera: document.getElementById('startCamera'),
            stopCamera: document.getElementById('stopCamera'),
            videoElement: document.getElementById('videoElement'),
            detectedObjects: document.getElementById('detectedObjects'),
            cameraStatus: document.getElementById('cameraStatus'),
            
            // OCR mode elements
            startOcrCamera: document.getElementById('startOcrCamera'),
            captureOcrPhoto: document.getElementById('captureOcrPhoto'),
            stopOcrCamera: document.getElementById('stopOcrCamera'),
            ocrVideoElement: document.getElementById('ocrVideoElement'),
            ocrOutput: document.getElementById('ocrOutput'),
            ocrStatus: document.getElementById('ocrStatus')
        };

        this.initializeApp();
    }

    async initializeApp() {
        console.log('🚀 Initializing Real-Time Translator...');
        
        try {
            this.setupEventListeners();
            await this.testBackendConnection();
            await this.loadLanguages();
            this.initializeSpeechRecognition();
            this.switchMode('voice');
            
            console.log('✅ App initialized successfully');
        } catch (error) {
            console.error('❌ App initialization failed:', error);
            this.showError('Failed to initialize app: ' + error.message);
        }
    }

    setupEventListeners() {
        // Mode switching
        this.elements.voiceMode?.addEventListener('click', () => this.switchMode('voice'));
        this.elements.textMode?.addEventListener('click', () => this.switchMode('text'));
        this.elements.cameraMode?.addEventListener('click', () => this.switchMode('camera'));
        this.elements.ocrMode?.addEventListener('click', () => this.switchMode('ocr'));

        // Language selection
        this.elements.targetLanguage?.addEventListener('change', (e) => {
            this.state.targetLanguage = e.target.value;
            console.log('Target language changed to:', e.target.value);
        });

        // Voice controls
        this.elements.startListening?.addEventListener('click', () => this.startListening());
        this.elements.stopListening?.addEventListener('click', () => this.stopListening());

        // Text controls
        this.elements.translateText?.addEventListener('click', () => this.translateTextInput());
        this.elements.clearText?.addEventListener('click', () => this.clearTextInput());
        this.elements.textInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.translateTextInput();
            }
        });

        // Camera controls
        this.elements.startCamera?.addEventListener('click', () => this.startCamera());
        this.elements.stopCamera?.addEventListener('click', () => this.stopCamera());

        // OCR controls
        this.elements.startOcrCamera?.addEventListener('click', () => this.startOcrCamera());
        this.elements.captureOcrPhoto?.addEventListener('click', () => this.captureOcrPhoto());
        this.elements.stopOcrCamera?.addEventListener('click', () => this.stopOcrCamera());
    }

    switchMode(mode) {
        console.log(`Switching to ${mode} mode`);
        
        // Stop current activities
        this.stopAllActivities();
        
        this.state.currentMode = mode;

        // Update button states
        document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
        this.elements[mode + 'Mode']?.classList.add('active');

        // Show/hide interfaces
        const interfaces = ['voice', 'text', 'camera', 'ocr'];
        interfaces.forEach(iface => {
            const element = this.elements[iface + 'Interface'];
            if (element) {
                element.classList.toggle('hidden', iface !== mode);
            }
        });

        // Update status
        this.updateStatus(`${mode.charAt(0).toUpperCase() + mode.slice(1)} mode active`);
    }

    stopAllActivities() {
        // Stop voice recognition
        if (this.state.isListening) {
            this.stopListening();
        }
        
        // Stop camera
        if (this.state.isCameraActive) {
            this.stopCamera();
        }
        
        // Stop OCR camera
        if (this.state.isOcrCameraActive) {
            this.stopOcrCamera();
        }
    }

    // ==================== VOICE MODE ====================

    initializeSpeechRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.warn('Speech recognition not supported');
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.state.recognition = new SpeechRecognition();
        
        this.state.recognition.continuous = true;
        this.state.recognition.interimResults = true;
        this.state.recognition.lang = 'auto';

        this.state.recognition.onstart = () => {
            console.log('🎤 Speech recognition started');
            this.updateVoiceStatus('Listening... Speak now');
        };

        this.state.recognition.onresult = (event) => {
            let finalTranscript = '';
            let interimTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }

            if (this.elements.originalText) {
                this.elements.originalText.textContent = finalTranscript || interimTranscript;
            }

            if (finalTranscript) {
                this.translateAndSpeak(finalTranscript);
            }
        };

        this.state.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.updateVoiceStatus('Error: ' + event.error);
            this.state.isListening = false;
            this.updateVoiceButtons();
        };

        this.state.recognition.onend = () => {
            console.log('🎤 Speech recognition ended');
            if (this.state.isListening) {
                // Restart if still supposed to be listening
                setTimeout(() => {
                    if (this.state.isListening) {
                        this.state.recognition.start();
                    }
                }, 100);
            } else {
                this.updateVoiceStatus('Stopped listening');
                this.updateVoiceButtons();
            }
        };
    }

    startListening() {
        if (!this.state.recognition) {
            this.showError('Speech recognition not available');
            return;
        }

        try {
            this.state.isListening = true;
            this.state.recognition.start();
            this.updateVoiceButtons();
            console.log('🎤 Starting speech recognition');
        } catch (error) {
            console.error('Failed to start listening:', error);
            this.showError('Failed to start listening: ' + error.message);
            this.state.isListening = false;
            this.updateVoiceButtons();
        }
    }

    stopListening() {
        if (this.state.recognition && this.state.isListening) {
            this.state.isListening = false;
            this.state.recognition.stop();
            this.updateVoiceButtons();
            console.log('🎤 Stopping speech recognition');
        }
    }

    updateVoiceButtons() {
        if (this.elements.startListening) {
            this.elements.startListening.disabled = this.state.isListening;
        }
        if (this.elements.stopListening) {
            this.elements.stopListening.disabled = !this.state.isListening;
        }
    }

    updateVoiceStatus(message) {
        if (this.elements.voiceStatus) {
            this.elements.voiceStatus.textContent = message;
        }
    }

    async translateAndSpeak(text) {
        try {
            console.log('🔄 Translating:', text);
            
            const translation = await this.translateText(text, this.state.targetLanguage);
            
            if (this.elements.translatedText) {
                this.elements.translatedText.textContent = translation.translated_text;
            }
            
            // Speak the translation
            this.speakText(translation.translated_text, this.state.targetLanguage);
            
        } catch (error) {
            console.error('Translation failed:', error);
            this.showError('Translation failed: ' + error.message);
        }
    }

    speakText(text, language) {
        if (!('speechSynthesis' in window)) {
            console.warn('Text-to-speech not supported');
            return;
        }

        try {
            // Cancel any ongoing speech
            speechSynthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = this.getVoiceLanguage(language);
            utterance.rate = 0.9;
            utterance.pitch = 1;
            
            utterance.onstart = () => console.log('🔊 Speaking:', text);
            utterance.onerror = (event) => console.error('Speech error:', event.error);
            
            speechSynthesis.speak(utterance);
        } catch (error) {
            console.error('Text-to-speech failed:', error);
        }
    }

    getVoiceLanguage(langCode) {
        const voiceMap = {
            'es': 'es-ES',
            'fr': 'fr-FR',
            'de': 'de-DE',
            'it': 'it-IT',
            'pt': 'pt-PT',
            'ru': 'ru-RU',
            'ja': 'ja-JP',
            'ko': 'ko-KR',
            'zh': 'zh-CN',
            'ar': 'ar-SA',
            'hi': 'hi-IN',
            'en': 'en-US'
        };
        return voiceMap[langCode] || 'en-US';
    }

    // ==================== TEXT MODE ====================

    async translateTextInput() {
        const text = this.elements.textInput?.value.trim();
        if (!text) {
            this.showError('Please enter text to translate');
            return;
        }

        if (text.length > APP_CONFIG.MAX_TEXT_LENGTH) {
            this.showError(`Text too long (max ${APP_CONFIG.MAX_TEXT_LENGTH} characters)`);
            return;
        }

        try {
            console.log('🔄 Translating text input:', text);
            
            if (this.elements.textTranslationOutput) {
                this.elements.textTranslationOutput.innerHTML = '<div class="loading">Translating...</div>';
            }
            
            const translation = await this.translateText(text, this.state.targetLanguage);
            
            if (this.elements.textTranslationOutput) {
                this.elements.textTranslationOutput.innerHTML = `
                    <div class="translation-result">
                        <div class="original">
                            <strong>Original (${translation.source_language}):</strong>
                            <p>${translation.original_text}</p>
                        </div>
                        <div class="translated">
                            <strong>Translation (${translation.target_language}):</strong>
                            <p>${translation.translated_text}</p>
                        </div>
                        <button onclick="window.translator.speakText('${translation.translated_text.replace(/'/g, "\\'")}', '${translation.target_language}')" class="speak-btn">🔊 Speak</button>
                    </div>
                `;
            }
            
        } catch (error) {
            console.error('Text translation failed:', error);
            this.showError('Translation failed: ' + error.message);
            
            if (this.elements.textTranslationOutput) {
                this.elements.textTranslationOutput.innerHTML = '<div class="error">Translation failed. Please try again.</div>';
            }
        }
    }

    clearTextInput() {
        if (this.elements.textInput) {
            this.elements.textInput.value = '';
        }
        if (this.elements.textTranslationOutput) {
            this.elements.textTranslationOutput.innerHTML = '<div class="placeholder">Translation will appear here...</div>';
        }
    }

    // ==================== CAMERA MODE ====================

    async startCamera() {
        try {
            console.log('📷 Starting camera...');
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            
            this.state.currentStream = stream;
            this.elements.videoElement.srcObject = stream;
            this.state.isCameraActive = true;
            
            this.updateCameraButtons();
            this.updateCameraStatus('Camera starting...');
            
            // Wait for video to load, then start object detection
            this.elements.videoElement.onloadedmetadata = () => {
                this.elements.videoElement.play();
                this.loadObjectDetectionModel();
            };
            
        } catch (error) {
            console.error('Camera failed:', error);
            this.showError('Camera access failed: ' + error.message);
            this.state.isCameraActive = false;
            this.updateCameraButtons();
        }
    }

    stopCamera() {
        console.log('📷 Stopping camera...');
        
        if (this.state.currentStream) {
            this.state.currentStream.getTracks().forEach(track => track.stop());
            this.state.currentStream = null;
        }
        
        if (this.elements.videoElement) {
            this.elements.videoElement.srcObject = null;
        }
        
        if (this.state.detectionInterval) {
            clearInterval(this.state.detectionInterval);
            this.state.detectionInterval = null;
        }
        
        this.state.isCameraActive = false;
        this.updateCameraButtons();
        this.updateCameraStatus('Camera stopped');
        
        if (this.elements.detectedObjects) {
            this.elements.detectedObjects.innerHTML = '<div class="placeholder">Start camera to detect objects</div>';
        }
    }

    updateCameraButtons() {
        if (this.elements.startCamera) {
            this.elements.startCamera.disabled = this.state.isCameraActive;
        }
        if (this.elements.stopCamera) {
            this.elements.stopCamera.disabled = !this.state.isCameraActive;
        }
    }

    updateCameraStatus(message) {
        if (this.elements.cameraStatus) {
            this.elements.cameraStatus.textContent = message;
        }
    }

    async loadObjectDetectionModel() {
        try {
            console.log('🤖 Loading object detection model...');
            this.updateCameraStatus('Loading AI model...');
            
            // Load TensorFlow.js and COCO-SSD model
            if (typeof cocoSsd === 'undefined') {
                throw new Error('TensorFlow.js COCO-SSD not loaded');
            }
            
            this.state.detectionModel = await cocoSsd.load();
            console.log('✅ Object detection model loaded');
            
            this.updateCameraStatus('Camera active - Detecting objects...');
            this.startObjectDetection();
            
        } catch (error) {
            console.error('Model loading failed:', error);
            this.updateCameraStatus('Model loading failed');
            this.showError('AI model loading failed: ' + error.message);
        }
    }

    startObjectDetection() {
        if (this.state.detectionInterval) {
            clearInterval(this.state.detectionInterval);
        }
        
        this.state.detectionInterval = setInterval(async () => {
            if (this.state.isCameraActive && this.state.detectionModel && this.elements.videoElement.readyState === 4) {
                await this.detectObjects();
            }
        }, APP_CONFIG.DETECTION_INTERVAL);
    }

    async detectObjects() {
        try {
            const predictions = await this.state.detectionModel.detect(this.elements.videoElement);
            
            if (predictions.length > 0) {
                await this.displayDetections(predictions);
            } else {
                if (this.elements.detectedObjects) {
                    this.elements.detectedObjects.innerHTML = '<div class="no-objects">No objects detected</div>';
                }
            }
            
        } catch (error) {
            console.error('Object detection failed:', error);
        }
    }

    async displayDetections(predictions) {
        const detectedItems = [];
        
        for (const prediction of predictions.slice(0, 3)) { // Limit to top 3
            if (prediction.score > 0.5) {
                try {
                    const translation = await this.translateText(prediction.class, this.state.targetLanguage);
                    detectedItems.push({
                        original: prediction.class,
                        translated: translation.translated_text,
                        confidence: Math.round(prediction.score * 100)
                    });
                } catch (error) {
                    console.error('Translation failed for:', prediction.class);
                    detectedItems.push({
                        original: prediction.class,
                        translated: prediction.class,
                        confidence: Math.round(prediction.score * 100)
                    });
                }
            }
        }
        
        if (detectedItems.length > 0 && this.elements.detectedObjects) {
            this.elements.detectedObjects.innerHTML = detectedItems.map(item => `
                <div class="detection-item">
                    <div class="object-name">
                        <span class="original">${item.original}</span>
                        <span class="arrow">→</span>
                        <span class="translated">${item.translated}</span>
                    </div>
                    <div class="confidence">${item.confidence}% confidence</div>
                    <button onclick="window.translator.speakText('${item.translated.replace(/'/g, "\\'")}', '${this.state.targetLanguage}')" class="speak-btn-small">🔊</button>
                </div>
            `).join('');
            
            // Auto-speak the first detection
            if (detectedItems[0]) {
                this.speakText(detectedItems[0].translated, this.state.targetLanguage);
            }
        }
    }

    // ==================== OCR MODE ====================

    async startOcrCamera() {
        try {
            console.log('📷 Starting OCR camera...');
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            
            this.state.ocrStream = stream;
            this.elements.ocrVideoElement.srcObject = stream;
            this.state.isOcrCameraActive = true;
            
            this.updateOcrButtons();
            this.updateOcrStatus('Camera ready - Take a photo');
            
            this.elements.ocrVideoElement.onloadedmetadata = () => {
                this.elements.ocrVideoElement.play();
            };
            
        } catch (error) {
            console.error('OCR camera failed:', error);
            this.showError('Camera access failed: ' + error.message);
            this.state.isOcrCameraActive = false;
            this.updateOcrButtons();
        }
    }

    async captureOcrPhoto() {
        if (!this.state.isOcrCameraActive) {
            this.showError('Camera not active');
            return;
        }

        try {
            console.log('📸 Capturing photo for OCR...');
            this.updateOcrStatus('Processing image...');
            
            // Create canvas and capture frame
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = this.elements.ocrVideoElement.videoWidth;
            canvas.height = this.elements.ocrVideoElement.videoHeight;
            
            context.drawImage(this.elements.ocrVideoElement, 0, 0);
            
            // Convert to blob
            canvas.toBlob(async (blob) => {
                try {
                    const ocrResult = await this.performOCR(blob);
                    await this.displayOcrResult(ocrResult);
                } catch (error) {
                    console.error('OCR processing failed:', error);
                    this.showError('OCR processing failed: ' + error.message);
                    this.updateOcrStatus('OCR failed - Try again');
                }
            }, 'image/jpeg', 0.8);
            
        } catch (error) {
            console.error('Photo capture failed:', error);
            this.showError('Photo capture failed: ' + error.message);
            this.updateOcrStatus('Capture failed - Try again');
        }
    }

    async performOCR(imageBlob) {
        const formData = new FormData();
        formData.append('image', imageBlob, 'capture.jpg');
        formData.append('target_language', this.state.targetLanguage);

        const response = await fetch(`${APP_CONFIG.API_BASE}/ocr`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`OCR request failed: ${response.status}`);
        }

        return await response.json();
    }

    async displayOcrResult(result) {
        console.log('📝 OCR result:', result);
        
        if (result.success && result.extracted_text) {
            this.elements.ocrOutput.innerHTML = `
                <div class="ocr-result">
                    <h4>📷 OCR Result</h4>
                    <div class="ocr-text">
                        <div class="extracted">
                            <strong>Extracted Text:</strong>
                            <p>${result.extracted_text}</p>
                        </div>
                        ${result.translated_text ? `
                            <div class="translated">
                                <strong>Translation (${result.target_language}):</strong>
                                <p>${result.translated_text}</p>
                            </div>
                            <button onclick="window.translator.speakText('${result.translated_text.replace(/'/g, "\\'")}', '${result.target_language}')" class="speak-btn">🔊 Speak Translation</button>
                        ` : ''}
                    </div>
                </div>
            `;
            
            this.updateOcrStatus('OCR completed successfully');
            
            // Auto-speak translation if available
            if (result.translated_text) {
                this.speakText(result.translated_text, result.target_language);
            }
            
        } else {
            this.elements.ocrOutput.innerHTML = `
                <div class="ocr-result error">
                    <h4>📷 OCR Result</h4>
                    <p>No text detected in the image. Try again with clearer text.</p>
                </div>
            `;
            this.updateOcrStatus('No text detected - Try again');
        }
    }

    stopOcrCamera() {
        console.log('📷 Stopping OCR camera...');
        
        if (this.state.ocrStream) {
            this.state.ocrStream.getTracks().forEach(track => track.stop());
            this.state.ocrStream = null;
        }
        
        if (this.elements.ocrVideoElement) {
            this.elements.ocrVideoElement.srcObject = null;
        }
        
        this.state.isOcrCameraActive = false;
        this.updateOcrButtons();
        this.updateOcrStatus('Camera stopped');
    }

    updateOcrButtons() {
        if (this.elements.startOcrCamera) {
            this.elements.startOcrCamera.disabled = this.state.isOcrCameraActive;
        }
        if (this.elements.captureOcrPhoto) {
            this.elements.captureOcrPhoto.disabled = !this.state.isOcrCameraActive;
        }
        if (this.elements.stopOcrCamera) {
            this.elements.stopOcrCamera.disabled = !this.state.isOcrCameraActive;
        }
    }

    updateOcrStatus(message) {
        if (this.elements.ocrStatus) {
            this.elements.ocrStatus.textContent = message;
        }
    }

    // ==================== API COMMUNICATION ====================

    async translateText(text, targetLanguage, sourceLang = 'auto') {
        const response = await fetch(`${APP_CONFIG.API_BASE}/translate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                target_language: targetLanguage,
                source_language: sourceLang
            })
        });

        if (!response.ok) {
            throw new Error(`Translation request failed: ${response.status}`);
        }

        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Translation failed');
        }

        return result;
    }

    async loadLanguages() {
        try {
            const response = await fetch(`${APP_CONFIG.API_BASE}/languages`);
            if (response.ok) {
                const languages = await response.json();
                this.populateLanguageSelect(languages);
            }
        } catch (error) {
            console.warn('Failed to load languages:', error);
            // Use default languages if API fails
            this.populateLanguageSelect({
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'it': 'Italian',
                'pt': 'Portuguese',
                'ru': 'Russian',
                'ja': 'Japanese',
                'ko': 'Korean',
                'zh': 'Chinese',
                'ar': 'Arabic',
                'hi': 'Hindi'
            });
        }
    }

    populateLanguageSelect(languages) {
        if (!this.elements.targetLanguage) return;
        
        this.elements.targetLanguage.innerHTML = '';
        
        Object.entries(languages).forEach(([code, name]) => {
            const option = document.createElement('option');
            option.value = code;
            option.textContent = name;
            if (code === this.state.targetLanguage) {
                option.selected = true;
            }
            this.elements.targetLanguage.appendChild(option);
        });
    }

    async testBackendConnection() {
        try {
            const response = await fetch(`${APP_CONFIG.API_BASE}/health`);
            if (response.ok) {
                console.log('✅ Backend connection successful');
                return true;
            } else {
                throw new Error(`Backend responded with status: ${response.status}`);
            }
        } catch (error) {
            console.warn('⚠️ Backend connection failed:', error);
            this.showError('Backend connection failed. Some features may not work.');
            return false;
        }
    }

    // ==================== UTILITY METHODS ====================

    updateStatus(message) {
        console.log('📊 Status:', message);
        
        // Update mode-specific status elements
        if (this.state.currentMode === 'voice' && this.elements.voiceStatus) {
            this.elements.voiceStatus.textContent = message;
        } else if (this.state.currentMode === 'camera' && this.elements.cameraStatus) {
            this.elements.cameraStatus.textContent = message;
        } else if (this.state.currentMode === 'ocr' && this.elements.ocrStatus) {
            this.elements.ocrStatus.textContent = message;
        }
    }

    showError(message) {
        console.error('❌ Error:', message);
        
        if (this.elements.errorMessage) {
            this.elements.errorMessage.textContent = message;
            this.elements.errorMessage.classList.remove('hidden');
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                this.elements.errorMessage.classList.add('hidden');
            }, 5000);
        }
    }

    hideError() {
        if (this.elements.errorMessage) {
            this.elements.errorMessage.classList.add('hidden');
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Starting Real-Time Translator...');
    window.translator = new RealTimeTranslator();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden && window.translator) {
        // Stop activities when page is hidden
        window.translator.stopAllActivities();
    }
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.translator) {
        window.translator.stopAllActivities();
    }
});
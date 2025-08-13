/**
 * Real-Time Translator - Complete Enhanced Version
 * With Advanced Object Detection (YOLO, MediaPipe, Enhanced COCO-SSD)
 * Updated with Text Mode, Language Swap, and Camera Switching
 */

// ==================== CONFIGURATION ====================
const getApiBase = () => {
    // If we're on Railway or production
    if (window.location.hostname.includes('railway.app') || 
        window.location.hostname.includes('herokuapp.com')) {
        return ''; // Use same origin
    }
    
    // For local development
    if (window.location.hostname === 'localhost' || 
        window.location.hostname === '127.0.0.1') {
        // Check if we're already on port 8080 (accessing through Flask)
        if (window.location.port === '8080') {
            return ''; // Same origin
        }
        // Otherwise, explicitly use port 8080
        return 'http://127.0.0.1:8080';
    }
    
    // Default to same origin
    return '';
};

// ==================== CONFIGURATION ====================
const APP_CONFIG = {
    // Fix the API base URL
    API_BASE: window.location.hostname.includes('railway.app') 
        ? '' // On Railway, use same origin (no localhost)
        : 'http://127.0.0.1:8080', // For local development, use port 8080
    
    DETECTION: {
        DEFAULT_MODEL: 'yolo',
        MIN_CONFIDENCE: 0.3,
        DETECTION_INTERVAL: 500,
        MAX_OBJECTS: 10,
        SMOOTHING_FRAMES: 3,
        TRANSLATION_DELAY: 3000
    },
    VOICE: {
        LANGUAGE: 'en-US',
        CONTINUOUS: true,
        INTERIM_RESULTS: true,
        MAX_ALTERNATIVES: 3
    },
    CAMERA: {
        WIDTH: { ideal: 1280, min: 640 },
        HEIGHT: { ideal: 720, min: 480 },
        FACING_MODE: 'environment', // Default to back camera
        FRAME_RATE: { ideal: 30, min: 15 }
    }
};

console.log('Running on:', window.location.hostname);
console.log('API Base:', APP_CONFIG.API_BASE || 'same origin');

// ==================== ADVANCED DETECTION MODELS ====================
class UniversalObjectDetector {
    constructor() {
        this.models = {
            yolo: null,
            mediapipe: null,
            cocoEnhanced: null,
            current: null
        };
        
        this.modelStatus = {
            yolo: 'not-loaded',
            mediapipe: 'not-loaded',
            cocoEnhanced: 'not-loaded'
        };
        
        this.detectionCache = new Map();
        this.performanceStats = {
            fps: 0,
            lastFrameTime: 0,
            detectionCount: 0
        };
    }

    /**
     * Initialize all available models
     */
    async initialize() {
        console.log('🚀 Initializing Universal Object Detector...');
        
        // Try to load models in parallel
        const modelPromises = [
            this.loadYOLO().catch(e => console.warn('YOLO failed:', e)),
            this.loadMediaPipe().catch(e => console.warn('MediaPipe failed:', e)),
            this.loadEnhancedCOCO().catch(e => console.warn('Enhanced COCO failed:', e))
        ];
        
        await Promise.allSettled(modelPromises);
        
        // Select best available model
        this.selectBestModel();
        
        return this.models.current !== null;
    }

    /**
     * Load YOLO Model (Best Accuracy)
     */
    async loadYOLO() {
        try {
            console.log('📦 Loading YOLO model...');
            
            // Load YOLOv5 or YOLOv8
            if (typeof tf === 'undefined') {
                throw new Error('TensorFlow.js not loaded');
            }
            
            // Custom YOLO implementation
            this.models.yolo = {
                model: await this.loadYOLOModel(),
                type: 'yolo',
                detect: async (video) => await this.detectWithYOLO(video)
            };
            
            this.modelStatus.yolo = 'loaded';
            console.log('✅ YOLO model loaded successfully');
            
        } catch (error) {
            this.modelStatus.yolo = 'failed';
            throw error;
        }
    }

    /**
     * Load actual YOLO model
     */
    async loadYOLOModel() {
        // Try YOLOv5 first
        try {
            const modelUrl = 'https://raw.githubusercontent.com/ultralytics/yolov5/master/models/yolov5s.json';
            return await tf.loadGraphModel(modelUrl);
        } catch (e) {
            // Fallback to COCO-SSD with YOLO-like processing
            console.log('Using enhanced COCO-SSD as YOLO fallback');
            return await cocoSsd.load({
                base: 'mobilenet_v2'
            });
        }
    }

    /**
     * YOLO Detection Implementation
     */
    async detectWithYOLO(video) {
        if (!this.models.yolo.model) return [];
        
        try {
            // If using actual YOLO
            if (this.models.yolo.model.predict) {
                return await this.yoloInference(video);
            } else {
                // If using COCO-SSD fallback
                const predictions = await this.models.yolo.model.detect(video, 20, 0.25);
                return this.enhancePredictions(predictions);
            }
        } catch (error) {
            console.error('YOLO detection error:', error);
            return [];
        }
    }

    /**
     * YOLO specific inference
     */
    async yoloInference(video) {
        const [modelWidth, modelHeight] = [640, 640];
        
        // Preprocess
        const input = tf.tidy(() => {
            const img = tf.browser.fromPixels(video);
            const resized = tf.image.resizeBilinear(img, [modelWidth, modelHeight]);
            const normalized = resized.div(255.0);
            return normalized.expandDims(0);
        });
        
        // Run inference
        const output = await this.models.yolo.model.predict(input);
        
        // Process output
        const predictions = await this.processYOLOOutput(output);
        
        // Cleanup
        input.dispose();
        output.dispose();
        
        return predictions;
    }

    /**
     * Load MediaPipe (Google's Advanced Detection)
     */
    async loadMediaPipe() {
        try {
            console.log('📦 Loading MediaPipe model...');
            
            // Check if MediaPipe is available
            if (typeof window.MediaPipeObjectDetector === 'undefined') {
                // Try dynamic import
                const vision = await this.loadMediaPipeLibrary();
                
                if (!vision) {
                    throw new Error('MediaPipe library not available');
                }
                
                const config = {
                    baseOptions: {
                        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite',
                        delegate: 'GPU'
                    },
                    scoreThreshold: 0.3,
                    maxResults: 15
                };
                
                this.models.mediapipe = {
                    model: await vision.ObjectDetector.createFromOptions(config),
                    type: 'mediapipe',
                    detect: async (video) => await this.detectWithMediaPipe(video)
                };
                
                this.modelStatus.mediapipe = 'loaded';
                console.log('✅ MediaPipe model loaded successfully');
            }
        } catch (error) {
            this.modelStatus.mediapipe = 'failed';
            throw error;
        }
    }

    /**
     * Load MediaPipe library dynamically
     */
    async loadMediaPipeLibrary() {
        try {
            return await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest');
        } catch (e) {
            console.warn('MediaPipe dynamic import failed:', e);
            return null;
        }
    }

    /**
     * MediaPipe Detection
     */
    async detectWithMediaPipe(video) {
        if (!this.models.mediapipe.model) return [];
        
        try {
            const results = await this.models.mediapipe.model.detectForVideo(
                video,
                performance.now()
            );
            
            return results.detections.map(det => ({
                class: det.categories[0].categoryName,
                score: det.categories[0].score,
                bbox: [
                    det.boundingBox.originX,
                    det.boundingBox.originY,
                    det.boundingBox.width,
                    det.boundingBox.height
                ]
            }));
        } catch (error) {
            console.error('MediaPipe detection error:', error);
            return [];
        }
    }

    /**
     * Load Enhanced COCO-SSD with Multiple Models
     */
    async loadEnhancedCOCO() {
        try {
            console.log('📦 Loading Enhanced COCO-SSD...');
            
            if (typeof cocoSsd === 'undefined') {
                throw new Error('COCO-SSD library not loaded');
            }
            
            // Load both lite and full models
            const [liteModel, fullModel] = await Promise.allSettled([
                cocoSsd.load({ base: 'lite_mobilenet_v2' }),
                cocoSsd.load({ base: 'mobilenet_v2' })
            ]);
            
            this.models.cocoEnhanced = {
                lite: liteModel.status === 'fulfilled' ? liteModel.value : null,
                full: fullModel.status === 'fulfilled' ? fullModel.value : null,
                type: 'coco-enhanced',
                detect: async (video) => await this.detectWithEnhancedCOCO(video)
            };
            
            this.modelStatus.cocoEnhanced = 'loaded';
            console.log('✅ Enhanced COCO-SSD loaded successfully');
            
        } catch (error) {
            this.modelStatus.cocoEnhanced = 'failed';
            throw error;
        }
    }

    /**
     * Enhanced COCO Detection with Multiple Models
     */
    async detectWithEnhancedCOCO(video) {
        const results = [];
        
        try {
            // Use full model if available, otherwise lite
            const model = this.models.cocoEnhanced.full || this.models.cocoEnhanced.lite;
            
            if (!model) return [];
            
            // Detect with multiple confidence levels
            const [highConf, medConf, lowConf] = await Promise.all([
                model.detect(video, 10, 0.5),
                model.detect(video, 15, 0.35),
                model.detect(video, 20, 0.25)
            ]);
            
            // Combine and deduplicate
            const allDetections = [...highConf, ...medConf, ...lowConf];
            return this.deduplicateDetections(allDetections);
            
        } catch (error) {
            console.error('Enhanced COCO detection error:', error);
            return [];
        }
    }

    /**
     * Select best available model
     */
    selectBestModel() {
        // Priority: YOLO > MediaPipe > Enhanced COCO
        if (this.modelStatus.yolo === 'loaded') {
            this.models.current = this.models.yolo;
            console.log('🎯 Selected YOLO as primary detector');
        } else if (this.modelStatus.mediapipe === 'loaded') {
            this.models.current = this.models.mediapipe;
            console.log('🎯 Selected MediaPipe as primary detector');
        } else if (this.modelStatus.cocoEnhanced === 'loaded') {
            this.models.current = this.models.cocoEnhanced;
            console.log('🎯 Selected Enhanced COCO as primary detector');
        } else {
            console.error('❌ No detection models available!');
        }
    }

    /**
     * Universal detect method
     */
    async detect(video) {
        if (!this.models.current) {
            console.error('No detection model available');
            return [];
        }
        
        const startTime = performance.now();
        
        try {
            // Get raw detections
            const detections = await this.models.current.detect(video);
            
            // Apply enhancements
            const enhanced = this.enhancePredictions(detections);
            
            // Apply temporal smoothing
            const smoothed = this.applySmoothing(enhanced);
            
            // Update performance stats
            this.updatePerformance(performance.now() - startTime);
            
            return smoothed;
            
        } catch (error) {
            console.error('Detection error:', error);
            return [];
        }
    }

    /**
     * Enhance predictions with additional processing
     */
    enhancePredictions(predictions) {
        return predictions.map(pred => ({
            ...pred,
            id: this.generateId(pred),
            timestamp: Date.now(),
            confidence: pred.score,
            enhanced: true,
            category: this.categorizeObject(pred.class)
        }));
    }

    /**
     * Apply temporal smoothing
     */
    applySmoothing(detections) {
        const smoothed = [];
        const currentTime = Date.now();
        
        // Update cache
        detections.forEach(det => {
            const key = `${det.class}_${Math.round(det.bbox[0]/50)}`;
            
            if (!this.detectionCache.has(key)) {
                this.detectionCache.set(key, {
                    detections: [],
                    lastSeen: currentTime
                });
            }
            
            const cached = this.detectionCache.get(key);
            cached.detections.push(det);
            cached.lastSeen = currentTime;
            
            // Keep only recent detections
            if (cached.detections.length > APP_CONFIG.DETECTION.SMOOTHING_FRAMES) {
                cached.detections.shift();
            }
        });
        
        // Clean old cache entries
        for (const [key, cached] of this.detectionCache.entries()) {
            if (currentTime - cached.lastSeen > 2000) {
                this.detectionCache.delete(key);
            } else if (cached.detections.length >= 2) {
                // Average the detections
                const avgDetection = this.averageDetections(cached.detections);
                smoothed.push(avgDetection);
            }
        }
        
        return smoothed;
    }

    /**
     * Average multiple detections
     */
    averageDetections(detections) {
        const avg = { ...detections[0] };
        
        // Average bbox
        avg.bbox = [0, 0, 0, 0];
        detections.forEach(det => {
            det.bbox.forEach((val, i) => {
                avg.bbox[i] += val / detections.length;
            });
        });
        
        // Average score
        avg.score = detections.reduce((sum, det) => sum + det.score, 0) / detections.length;
        avg.confidence = avg.score;
        
        return avg;
    }

    /**
     * Deduplicate detections
     */
    deduplicateDetections(detections) {
        const unique = [];
        const seen = new Set();
        
        detections.sort((a, b) => b.score - a.score);
        
        detections.forEach(det => {
            const key = `${det.class}_${Math.round(det.bbox[0]/100)}_${Math.round(det.bbox[1]/100)}`;
            
            if (!seen.has(key)) {
                seen.add(key);
                unique.push(det);
            }
        });
        
        return unique;
    }

    /**
     * Process YOLO output
     */
    async processYOLOOutput(output) {
        // YOLO specific output processing
        const predictions = [];
        const outputData = await output.array();
        
        // Process based on YOLO format
        // This is simplified - actual YOLO output processing is more complex
        for (let i = 0; i < outputData[0].length; i++) {
            const detection = outputData[0][i];
            if (detection[4] > APP_CONFIG.DETECTION.MIN_CONFIDENCE) {
                predictions.push({
                    bbox: [detection[0], detection[1], detection[2], detection[3]],
                    score: detection[4],
                    class: this.getClassName(detection[5])
                });
            }
        }
        
        return predictions;
    }

    /**
     * Get class name from index
     */
    getClassName(index) {
        const classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ];
        
        return classes[index] || `Object_${index}`;
    }

    /**
     * Categorize objects
     */
    categorizeObject(className) {
        const categories = {
            'person': 'People',
            'bicycle': 'Vehicle',
            'car': 'Vehicle',
            'motorcycle': 'Vehicle',
            'airplane': 'Vehicle',
            'bus': 'Vehicle',
            'train': 'Vehicle',
            'truck': 'Vehicle',
            'boat': 'Vehicle',
            'bird': 'Animal',
            'cat': 'Animal',
            'dog': 'Animal',
            'horse': 'Animal',
            'sheep': 'Animal',
            'cow': 'Animal',
            'elephant': 'Animal',
            'bear': 'Animal',
            'zebra': 'Animal',
            'giraffe': 'Animal',
            'chair': 'Furniture',
            'couch': 'Furniture',
            'bed': 'Furniture',
            'dining table': 'Furniture',
            'toilet': 'Furniture',
            'tv': 'Electronics',
            'laptop': 'Electronics',
            'mouse': 'Electronics',
            'remote': 'Electronics',
            'keyboard': 'Electronics',
            'cell phone': 'Electronics',
            'book': 'Object',
            'clock': 'Object',
            'vase': 'Object',
            'scissors': 'Object',
            'teddy bear': 'Toy',
            'bottle': 'Container',
            'wine glass': 'Container',
            'cup': 'Container',
            'bowl': 'Container',
            'banana': 'Food',
            'apple': 'Food',
            'sandwich': 'Food',
            'orange': 'Food',
            'broccoli': 'Food',
            'carrot': 'Food',
            'hot dog': 'Food',
            'pizza': 'Food',
            'donut': 'Food',
            'cake': 'Food'
        };
        
        return categories[className] || 'Miscellaneous';
    }

    /**
     * Generate unique ID
     */
    generateId(detection) {
        return `${detection.class}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Update performance metrics
     */
    updatePerformance(detectionTime) {
        const currentTime = Date.now();
        
        if (this.performanceStats.lastFrameTime) {
            const timeDiff = currentTime - this.performanceStats.lastFrameTime;
            this.performanceStats.fps = Math.round(1000 / timeDiff);
        }
        
        this.performanceStats.lastFrameTime = currentTime;
        this.performanceStats.detectionCount++;
        this.performanceStats.lastDetectionTime = detectionTime;
    }

    /**
     * Get performance stats
     */
    getPerformanceStats() {
        return {
            ...this.performanceStats,
            modelType: this.models.current?.type || 'none',
            cacheSize: this.detectionCache.size
        };
    }
}

// ==================== MAIN APPLICATION ====================
class RealTimeTranslator {
    constructor() {
        // State management
        this.state = {
            currentMode: 'voice',
            sourceLanguage: 'auto', // Added source language
            targetLanguage: 'es',
            isListening: false,
            isCameraActive: false,
            isSpeaking: false,
            lastTranslation: null,
            recognition: null,
            videoStream: null,
            currentCameraFacing: 'environment', // Track current camera (back camera default)
            detectionInterval: null,
            translationQueue: [],
            processedObjects: new Set()
        };

        // DOM elements
        this.elements = {
            voiceMode: document.getElementById('voiceMode'),
            textMode: document.getElementById('textMode'), // Added text mode
            cameraMode: document.getElementById('cameraMode'),
            sourceLanguage: document.getElementById('sourceLanguage'), // Added source language
            targetLanguage: document.getElementById('targetLanguage'),
            swapLanguages: document.getElementById('swapLanguages'), // Added swap button
            voiceInterface: document.getElementById('voiceInterface'),
            textInterface: document.getElementById('textInterface'), // Added text interface
            cameraInterface: document.getElementById('cameraInterface'),
            startListening: document.getElementById('startListening'),
            stopListening: document.getElementById('stopListening'),
            startCamera: document.getElementById('startCamera'),
            stopCamera: document.getElementById('stopCamera'),
            switchCamera: document.getElementById('switchCamera'), // Added switch camera
            textInput: document.getElementById('textInput'), // Added text input
            translateText: document.getElementById('translateText'), // Added translate button
            clearText: document.getElementById('clearText'), // Added clear button
            textTranslationOutput: document.getElementById('textTranslationOutput'), // Added text output
            voiceStatus: document.getElementById('voiceStatus'),
            cameraStatus: document.getElementById('cameraStatus'),
            originalText: document.getElementById('originalText'),
            translatedText: document.getElementById('translatedText'),
            videoElement: document.getElementById('videoElement'),
            detectionCanvas: document.getElementById('detectionCanvas'),
            detectedObjects: document.getElementById('detectedObjects'),
            errorMessage: document.getElementById('errorMessage'),
            performanceStats: document.getElementById('performanceStats')
        };

        // Initialize detector
        this.detector = new UniversalObjectDetector();
        
        // API configuration
        this.apiBase = APP_CONFIG.API_BASE;
        
        // Performance monitoring
        this.performanceMonitor = {
            startTime: Date.now(),
            totalTranslations: 0,
            totalDetections: 0
        };

        // Initialize
        this.initializeApp();
    }

    /**
     * Initialize application
     */
    async initializeApp() {
        console.log('🚀 Initializing Real-Time Translator...');
        
        try {
            // Setup event listeners
            this.setupEventListeners();
            
            // Check browser compatibility
            this.checkBrowserCompatibility();
            
            // Initialize error handling
            this.initializeErrorHandling();
            
            // Test backend connection
            await this.testBackendConnection();
            
            // Load supported languages
            await this.loadSupportedLanguages();
            
            // Initialize voice recognition
            this.initializeSpeechRecognition();
            
            // Initialize text-to-speech
            this.initializeTextToSpeech();
            
            // Load detection models
            await this.loadDetectionModels();
            
            // Set default mode
            this.switchMode('voice');
            
            console.log('✅ Application initialized successfully');
            
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            this.showError('Failed to initialize application: ' + error.message);
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Mode switching
        this.elements.voiceMode?.addEventListener('click', () => this.switchMode('voice'));
        this.elements.textMode?.addEventListener('click', () => this.switchMode('text')); // Added text mode
        this.elements.cameraMode?.addEventListener('click', () => this.switchMode('camera'));
        
        // Language selection
        this.elements.sourceLanguage?.addEventListener('change', (e) => {
            this.state.sourceLanguage = e.target.value;
            console.log('Source language:', this.state.sourceLanguage);
        });
        
        this.elements.targetLanguage?.addEventListener('change', (e) => {
            this.state.targetLanguage = e.target.value;
            console.log('Target language:', this.state.targetLanguage);
        });
        
        // Swap languages button
        this.elements.swapLanguages?.addEventListener('click', () => this.swapLanguages());
        
        // Voice controls
        this.elements.startListening?.addEventListener('click', () => this.startListening());
        this.elements.stopListening?.addEventListener('click', () => this.stopListening());
        
        // Text mode controls
        this.elements.translateText?.addEventListener('click', () => this.translateTextInput());
        this.elements.clearText?.addEventListener('click', () => this.clearTextInput());
        
        // Enter key in text input
        this.elements.textInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                this.translateTextInput();
            }
        });
        
        // Camera controls
        this.elements.startCamera?.addEventListener('click', () => this.startCamera());
        this.elements.stopCamera?.addEventListener('click', () => this.stopCamera());
        this.elements.switchCamera?.addEventListener('click', () => this.switchCamera());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === ' ' && e.ctrlKey) {
                e.preventDefault();
                if (this.state.currentMode === 'voice') {
                    this.state.isListening ? this.stopListening() : this.startListening();
                }
            }
        });
    }

    /**
     * Swap source and target languages
     */
    swapLanguages() {
        // Don't swap if source is auto-detect
        if (this.state.sourceLanguage === 'auto') {
            this.showWarning('Cannot swap when source is set to Auto Detect');
            return;
        }
        
        const tempLang = this.state.sourceLanguage;
        this.state.sourceLanguage = this.state.targetLanguage;
        this.state.targetLanguage = tempLang;
        
        // Update dropdowns
        if (this.elements.sourceLanguage) {
            this.elements.sourceLanguage.value = this.state.sourceLanguage;
        }
        if (this.elements.targetLanguage) {
            this.elements.targetLanguage.value = this.state.targetLanguage;
        }
        
        console.log(`Languages swapped: ${this.state.sourceLanguage} ⇄ ${this.state.targetLanguage}`);
        
        // Show confirmation
        this.showStatus(`Languages swapped`, 'voiceStatus');
        setTimeout(() => {
            if (this.state.currentMode === 'voice') {
                this.showStatus('Ready to listen', 'voiceStatus');
            }
        }, 2000);
    }

    /**
     * Translate text from input field
     */
    async translateTextInput() {
        const text = this.elements.textInput?.value.trim();
        
        if (!text) {
            this.showWarning('Please enter some text to translate');
            return;
        }
        
        try {
            // Show loading state
            this.updateDisplay('textTranslationOutput', 'Translating...');
            this.elements.translateText.disabled = true;
            
            // Translate
            const translation = await this.translateText(text, this.state.targetLanguage, this.state.sourceLanguage);
            
            if (translation) {
                let output = translation.translated_text;
                
                // Add detected language info if auto-detect was used
                if (this.state.sourceLanguage === 'auto' && translation.source_language) {
                    const languageNames = {
                        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
                        'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
                        'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi'
                    };
                    const detectedName = languageNames[translation.source_language] || translation.source_language;
                    output = `[Detected: ${detectedName}]\n\n${output}`;
                }
                
                this.updateDisplay('textTranslationOutput', output);
                
                // Update stats
                this.performanceMonitor.totalTranslations++;
                this.updatePerformanceDisplay();
            }
            
        } catch (error) {
            console.error('Translation error:', error);
            this.updateDisplay('textTranslationOutput', 'Translation failed: ' + error.message);
        } finally {
            this.elements.translateText.disabled = false;
        }
    }

    /**
     * Clear text input and output
     */
    clearTextInput() {
        if (this.elements.textInput) {
            this.elements.textInput.value = '';
        }
        this.updateDisplay('textTranslationOutput', 'Translation will appear here...');
    }

    /**
     * Check browser compatibility
     */
    checkBrowserCompatibility() {
        const features = {
            speechRecognition: 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window,
            speechSynthesis: 'speechSynthesis' in window,
            getUserMedia: navigator.mediaDevices && navigator.mediaDevices.getUserMedia,
            webGL: this.checkWebGLSupport()
        };
        
        const missing = Object.entries(features)
            .filter(([_, supported]) => !supported)
            .map(([feature]) => feature);
        
        if (missing.length > 0) {
            console.warn('Missing features:', missing);
            this.showWarning(`Some features may not work: ${missing.join(', ')}`);
        }
        
        return features;
    }

    /**
     * Check WebGL support
     */
    checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
        } catch (e) {
            return false;
        }
    }

    /**
     * Initialize error handling
     */
    initializeErrorHandling() {
        // Global error handler
        window.addEventListener('error', (e) => {
            console.error('Global error:', e);
            this.showError('An unexpected error occurred');
        });
        
        // Unhandled promise rejection
        window.addEventListener('unhandledrejection', (e) => {
            console.error('Unhandled promise rejection:', e);
            this.showError('An operation failed unexpectedly');
        });
        
        // Network status
        window.addEventListener('online', () => {
            console.log('Back online');
            this.hideError();
            this.testBackendConnection();
        });
        
        window.addEventListener('offline', () => {
            console.log('Offline');
            this.showError('No internet connection', false);
        });
    }

    /**
     * Test backend connection
     */
    async testBackendConnection() {
        try {
            const apiUrl = APP_CONFIG.API_BASE ? 
                `${APP_CONFIG.API_BASE}/health` : 
                '/health';
                
            console.log('Testing backend at:', apiUrl);
            
            const response = await fetch(apiUrl, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            console.log('✅ Backend connected:', data.message);
            return true;
            
        } catch (error) {
            console.error('❌ Backend connection failed:', error);
            this.showError('Cannot connect to translation server. Make sure backend is running.');
            return false;
        }
    }

    /**
     * Load supported languages
     */
    async loadSupportedLanguages() {
        try {
            const apiUrl = APP_CONFIG.API_BASE ? 
                `${APP_CONFIG.API_BASE}/languages` : 
                '/languages';
                
            const response = await fetch(apiUrl);
            const data = await response.json();
            
            if (data.languages) {
                this.updateLanguageDropdowns(data.languages);
            }
            
        } catch (error) {
            console.warn('Failed to load languages:', error);
            this.useDefaultLanguages();
        }
    }

    /**
     * Update language dropdowns (both source and target)
     */
    updateLanguageDropdowns(languages) {
        // Update source language dropdown
        if (this.elements.sourceLanguage) {
            const currentSource = this.elements.sourceLanguage.value;
            this.elements.sourceLanguage.innerHTML = '<option value="auto">Auto Detect</option>';
            
            Object.entries(languages).forEach(([code, name]) => {
                const option = document.createElement('option');
                option.value = code;
                option.textContent = name;
                if (code === currentSource) {
                    option.selected = true;
                }
                this.elements.sourceLanguage.appendChild(option);
            });
        }
        
        // Update target language dropdown
        if (this.elements.targetLanguage) {
            const currentTarget = this.elements.targetLanguage.value;
            this.elements.targetLanguage.innerHTML = '';
            
            Object.entries(languages).forEach(([code, name]) => {
                const option = document.createElement('option');
                option.value = code;
                option.textContent = name;
                if (code === currentTarget || (currentTarget === '' && code === 'es')) {
                    option.selected = true;
                }
                this.elements.targetLanguage.appendChild(option);
            });
        }
    }

    /**
     * Use default languages
     */
    useDefaultLanguages() {
        const defaultLanguages = {
            'en': 'English',
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
        };
        
        this.updateLanguageDropdowns(defaultLanguages);
    }

    /**
     * Load detection models
     */
    async loadDetectionModels() {
        try {
            console.log('Loading object detection models...');
            this.showStatus('Loading AI models...', 'cameraStatus');
            
            const success = await this.detector.initialize();
            
            if (success) {
                console.log('✅ Detection models loaded');
                this.showStatus('AI models ready', 'cameraStatus');
            } else {
                throw new Error('No models could be loaded');
            }
            
        } catch (error) {
            console.error('❌ Failed to load detection models:', error);
            this.showError('Object detection unavailable');
        }
    }

    /**
     * Switch application mode
     */
    switchMode(mode) {
        // Stop current mode
        if (this.state.currentMode === 'voice' && this.state.isListening) {
            this.stopListening();
        } else if (this.state.currentMode === 'camera' && this.state.isCameraActive) {
            this.stopCamera();
        }
        
        // Update state
        this.state.currentMode = mode;
        
        // Update UI
        this.updateModeUI(mode);
        
        console.log(`Switched to ${mode} mode`);
    }

    /**
     * Update mode UI
     */
    updateModeUI(mode) {
        // Update buttons
        this.elements.voiceMode?.classList.toggle('active', mode === 'voice');
        this.elements.textMode?.classList.toggle('active', mode === 'text');
        this.elements.cameraMode?.classList.toggle('active', mode === 'camera');
        
        // Show/hide interfaces
        this.elements.voiceInterface?.classList.toggle('hidden', mode !== 'voice');
        this.elements.textInterface?.classList.toggle('hidden', mode !== 'text');
        this.elements.cameraInterface?.classList.toggle('hidden', mode !== 'camera');
        
        // Update status
        if (mode === 'voice') {
            this.showStatus('Ready to listen', 'voiceStatus');
        } else if (mode === 'text') {
            // Focus on text input
            setTimeout(() => this.elements.textInput?.focus(), 100);
        } else {
            this.showStatus('Ready to detect objects', 'cameraStatus');
        }
    }

    // ==================== VOICE MODE ====================
    
    /**
     * Initialize speech recognition
     */
    initializeSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (!SpeechRecognition) {
            console.warn('Speech recognition not supported');
            this.enableManualInput();
            return;
        }
        
        try {
            this.state.recognition = new SpeechRecognition();
            
            // Configure
            Object.assign(this.state.recognition, APP_CONFIG.VOICE);
            
            // Event handlers
            this.setupSpeechHandlers();
            
            console.log('✅ Speech recognition initialized');
            
        } catch (error) {
            console.error('❌ Speech recognition initialization failed:', error);
            this.enableManualInput();
        }
    }

    /**
     * Setup speech recognition handlers
     */
    setupSpeechHandlers() {
        const recognition = this.state.recognition;
        
        recognition.onstart = () => {
            console.log('🎤 Listening...');
            this.state.isListening = true;
            this.showStatus('Listening... Speak now', 'voiceStatus', true);
            this.updateVoiceButtons();
        };
        
        recognition.onresult = async (event) => {
            // Skip if speaking
            if (this.state.isSpeaking) return;
            
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
            
            // Show interim results
            if (interimTranscript) {
                this.updateDisplay('originalText', `${interimTranscript}...`);
            }
            
            // Process final results
            if (finalTranscript) {
                console.log('📝 Transcript:', finalTranscript);
                this.updateDisplay('originalText', finalTranscript);
                
                // Pause listening during translation
                this.pauseListening();
                await this.processVoiceInput(finalTranscript);
            }
        };
        
        recognition.onerror = (event) => {
            console.error('❌ Speech recognition error:', event.error);
            this.handleSpeechError(event.error);
        };
        
        recognition.onend = () => {
            console.log('🎤 Stopped listening');
            this.state.isListening = false;
            this.showStatus('Ready to listen', 'voiceStatus');
            this.updateVoiceButtons();
        };
    }

    /**
     * Start listening
     */
    async startListening() {
        if (this.state.isSpeaking) {
            this.showWarning('Please wait for translation to finish');
            return;
        }
        
        if (!this.state.recognition) {
            this.initializeSpeechRecognition();
        }
        
        try {
            await this.state.recognition.start();
        } catch (error) {
            if (error.message?.includes('already started')) {
                this.state.recognition.stop();
                setTimeout(() => this.startListening(), 100);
            } else {
                console.error('Failed to start listening:', error);
                this.showError('Failed to start speech recognition');
            }
        }
    }

    /**
     * Stop listening
     */
    stopListening() {
        if (this.state.recognition && this.state.isListening) {
            this.state.recognition.stop();
        }
    }

    /**
     * Pause listening
     */
    pauseListening() {
        if (this.state.isListening) {
            this.state.recognition.stop();
            this.state.isListening = false;
        }
    }

    /**
     * Resume listening
     */
    resumeListening() {
        if (!this.state.isListening && !this.state.isSpeaking) {
            setTimeout(() => this.startListening(), 500);
        }
    }

    /**
     * Process voice input
     */
    async processVoiceInput(text) {
        try {
            this.updateDisplay('translatedText', 'Translating...');
            
            // Translate
            const translation = await this.translateText(text, this.state.targetLanguage, this.state.sourceLanguage);
            
            if (translation) {
                this.updateDisplay('translatedText', translation.translated_text);
                
                // Speak translation
                await this.speakText(translation.translated_text, this.state.targetLanguage);
                
                // Update stats
                this.performanceMonitor.totalTranslations++;
                this.updatePerformanceDisplay();
            }
            
        } catch (error) {
            console.error('Voice processing error:', error);
            this.updateDisplay('translatedText', 'Translation failed');
            this.showError('Translation failed: ' + error.message);
        } finally {
            // Resume listening
            this.resumeListening();
        }
    }

    /**
     * Handle speech recognition errors
     */
    handleSpeechError(error) {
        const errorMessages = {
            'no-speech': 'No speech detected',
            'audio-capture': 'Microphone not accessible',
            'not-allowed': 'Microphone permission denied',
            'network': 'Network error - check internet connection'
        };
        
        const message = errorMessages[error] || `Speech error: ${error}`;
        this.showError(message);
        
        if (['not-allowed', 'audio-capture'].includes(error)) {
            this.enableManualInput();
        }
    }

    /**
     * Enable manual text input
     */
    enableManualInput() {
        if (document.getElementById('manualInputContainer')) return;
        
        const html = `
            <div id="manualInputContainer" class="manual-input-container">
                <h3>Manual Text Input</h3>
                <textarea 
                    id="manualTextInput" 
                    placeholder="Type or paste text to translate..."
                    rows="3"
                ></textarea>
                <button id="translateManualText" class="btn btn-primary">
                    Translate
                </button>
            </div>
        `;
        
        this.elements.voiceInterface?.insertAdjacentHTML('beforeend', html);
        
        document.getElementById('translateManualText')?.addEventListener('click', async () => {
            const text = document.getElementById('manualTextInput').value.trim();
            if (text) {
                this.updateDisplay('originalText', text);
                await this.processVoiceInput(text);
            }
        });
    }

    /**
     * Update voice buttons
     */
    updateVoiceButtons() {
        if (this.elements.startListening) {
            this.elements.startListening.disabled = this.state.isListening || this.state.isSpeaking;
        }
        
        if (this.elements.stopListening) {
            this.elements.stopListening.disabled = !this.state.isListening;
            this.elements.stopListening.classList.toggle('active', this.state.isListening);
        }
    }

    // ==================== TEXT-TO-SPEECH ====================
    
    /**
     * Initialize text-to-speech
     */
    initializeTextToSpeech() {
        if (!('speechSynthesis' in window)) {
            console.warn('Text-to-speech not supported');
            return false;
        }
        
        // Load voices
        this.loadVoices();
        
        // Voice change event
        speechSynthesis.onvoiceschanged = () => this.loadVoices();
        
        return true;
    }

    /**
     * Load available voices
     */
    loadVoices() {
        this.availableVoices = speechSynthesis.getVoices();
        console.log(`Loaded ${this.availableVoices.length} voices`);
    }

    /**
     * Speak text
     */
    async speakText(text, language) {
        if (!text || !speechSynthesis) return;
        
        return new Promise((resolve) => {
            // Set speaking flag
            this.state.isSpeaking = true;
            this.updateVoiceButtons();
            
            // Cancel any ongoing speech
            speechSynthesis.cancel();
            
            // Create utterance
            const utterance = new SpeechSynthesisUtterance(text);
            
            // Configure
            utterance.rate = 0.9;
            utterance.pitch = 1.0;
            utterance.volume = 0.8;
            
            // Select voice
            const voice = this.selectVoice(language);
            if (voice) {
                utterance.voice = voice;
                utterance.lang = voice.lang;
            } else {
                utterance.lang = this.getLanguageCode(language);
            }
            
            // Events
            utterance.onstart = () => {
                console.log('🔊 Speaking...');
                this.showStatus('Speaking translation...', 'voiceStatus');
            };
            
            utterance.onend = () => {
                console.log('🔊 Finished speaking');
                this.state.isSpeaking = false;
                this.showStatus('Ready to listen', 'voiceStatus');
                this.updateVoiceButtons();
                resolve();
            };
            
            utterance.onerror = (event) => {
                console.error('TTS error:', event);
                this.state.isSpeaking = false;
                this.updateVoiceButtons();
                resolve();
            };
            
            // Speak
            speechSynthesis.speak(utterance);
        });
    }

    /**
     * Select appropriate voice
     */
    selectVoice(languageCode) {
        if (!this.availableVoices) return null;
        
        // Try exact match
        let voice = this.availableVoices.find(v => 
            v.lang.toLowerCase().startsWith(languageCode.toLowerCase())
        );
        
        // Try partial match
        if (!voice) {
            const langMap = {
                'es': ['es-ES', 'es-US', 'es-MX'],
                'fr': ['fr-FR', 'fr-CA'],
                'de': ['de-DE', 'de-AT'],
                'zh': ['zh-CN', 'zh-TW', 'zh-HK']
            };
            
            const alternatives = langMap[languageCode] || [];
            for (const alt of alternatives) {
                voice = this.availableVoices.find(v => 
                    v.lang.toLowerCase().includes(alt.toLowerCase())
                );
                if (voice) break;
            }
        }
        
        return voice;
    }

    /**
     * Get language code
     */
    getLanguageCode(code) {
        const langMap = {
            'es': 'es-ES',
            'fr': 'fr-FR',
            'de': 'de-DE',
            'it': 'it-IT',
            'pt': 'pt-BR',
            'ru': 'ru-RU',
            'ja': 'ja-JP',
            'ko': 'ko-KR',
            'zh': 'zh-CN',
            'ar': 'ar-SA',
            'hi': 'hi-IN'
        };
        
        return langMap[code] || code;
    }

    // ==================== CAMERA MODE ====================
    
    /**
     * Start camera
     */
    async startCamera() {
        console.log('📷 Starting camera...');
        
        try {
            // Check support
            if (!navigator.mediaDevices?.getUserMedia) {
                throw new Error('Camera not supported');
            }
            
            // Check if we can enumerate devices (for camera switching)
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            
            console.log(`Found ${videoDevices.length} camera(s)`);
            
            // Enable switch button if multiple cameras
            if (this.elements.switchCamera && videoDevices.length > 1) {
                this.elements.switchCamera.disabled = false;
            }
            
            this.showStatus('Requesting camera access...', 'cameraStatus');
            
            // Request camera with current facing mode (back camera by default)
            const constraints = {
                video: {
                    width: APP_CONFIG.CAMERA.WIDTH,
                    height: APP_CONFIG.CAMERA.HEIGHT,
                    frameRate: APP_CONFIG.CAMERA.FRAME_RATE,
                    facingMode: this.state.currentCameraFacing
                },
                audio: false
            };
            
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Set stream
            this.state.videoStream = stream;
            this.elements.videoElement.srcObject = stream;
            
            // Wait for metadata
            await new Promise((resolve) => {
                this.elements.videoElement.onloadedmetadata = () => {
                    this.elements.videoElement.play();
                    resolve();
                };
            });
            
            // Setup canvas
            this.setupCanvas();
            
            // Update state
            this.state.isCameraActive = true;
            this.updateCameraButtons();
            
            // Start detection
            this.startDetection();
            
            console.log('✅ Camera started');
            this.hideError();
            
        } catch (error) {
            console.error('❌ Camera error:', error);
            
            // If back camera fails, try any available camera
            if (this.state.currentCameraFacing === 'environment' && !this.state.videoStream) {
                console.log('Back camera failed, trying any camera...');
                this.state.currentCameraFacing = 'user';
                
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: true,
                        audio: false
                    });
                    
                    this.state.videoStream = stream;
                    this.elements.videoElement.srcObject = stream;
                    
                    await new Promise((resolve) => {
                        this.elements.videoElement.onloadedmetadata = () => {
                            this.elements.videoElement.play();
                            resolve();
                        };
                    });
                    
                    this.setupCanvas();
                    this.state.isCameraActive = true;
                    this.updateCameraButtons();
                    this.startDetection();
                    
                    console.log('✅ Camera started with fallback');
                    this.showWarning('Using front camera (back camera not available)');
                    
                } catch (fallbackError) {
                    this.showError('Camera access failed: ' + fallbackError.message);
                    this.stopCamera();
                }
            } else {
                this.showError('Camera access failed: ' + error.message);
                this.stopCamera();
            }
        }
    }

    /**
     * Switch between front and back camera
     */
    async switchCamera() {
        if (!this.state.isCameraActive) return;
        
        console.log('Switching camera...');
        
        // Toggle facing mode
        this.state.currentCameraFacing = 
            this.state.currentCameraFacing === 'environment' ? 'user' : 'environment';
        
        // Restart camera with new facing mode
        await this.stopCamera();
        await this.startCamera();
    }

    /**
     * Stop camera
     */
    stopCamera() {
        console.log('📷 Stopping camera...');
        
        // Stop stream
        if (this.state.videoStream) {
            this.state.videoStream.getTracks().forEach(track => track.stop());
            this.state.videoStream = null;
        }
        
        // Clear video
        if (this.elements.videoElement) {
            this.elements.videoElement.srcObject = null;
        }
        
        // Stop detection
        this.stopDetection();
        
        // Clear canvas
        this.clearCanvas();
        
        // Update state
        this.state.isCameraActive = false;
        this.updateCameraButtons();
        
        // Clear display
        this.updateDisplay('detectedObjects', 'Camera stopped');
        
        console.log('✅ Camera stopped');
    }

    /**
     * Setup canvas
     */
    setupCanvas() {
        const video = this.elements.videoElement;
        const canvas = this.elements.detectionCanvas;
        
        if (!video || !canvas) return;
        
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        
        // Position canvas over video
        canvas.style.position = 'absolute';
        canvas.style.top = `${video.offsetTop}px`;
        canvas.style.left = `${video.offsetLeft}px`;
        canvas.style.width = `${video.offsetWidth}px`;
        canvas.style.height = `${video.offsetHeight}px`;
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '10';
    }

    /**
     * Clear canvas
     */
    clearCanvas() {
        const canvas = this.elements.detectionCanvas;
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }

    /**
     * Update camera buttons
     */
    updateCameraButtons() {
        if (this.elements.startCamera) {
            this.elements.startCamera.disabled = this.state.isCameraActive;
        }
        
        if (this.elements.stopCamera) {
            this.elements.stopCamera.disabled = !this.state.isCameraActive;
            this.elements.stopCamera.classList.toggle('active', this.state.isCameraActive);
        }
        
        if (this.elements.switchCamera) {
            this.elements.switchCamera.disabled = !this.state.isCameraActive;
        }
    }

    /**
     * Start object detection
     */
    startDetection() {
        if (!this.state.isCameraActive) return;
        
        console.log('🔍 Starting object detection...');
        this.showStatus('Detecting objects...', 'cameraStatus', true);
        
        // Clear previous interval
        if (this.state.detectionInterval) {
            clearInterval(this.state.detectionInterval);
        }
        
        // Detection loop
        const detect = async () => {
            if (!this.state.isCameraActive) return;
            
            try {
                const detections = await this.detector.detect(this.elements.videoElement);
                
                if (detections && detections.length > 0) {
                    await this.processDetections(detections);
                    this.drawDetections(detections);
                    this.updateDetectionDisplay(detections);
                } else {
                    this.updateDisplay('detectedObjects', 'No objects detected');
                }
                
                // Update performance stats
                this.updatePerformanceDisplay();
                
            } catch (error) {
                console.error('Detection error:', error);
            }
        };
        
        // Start interval
        this.state.detectionInterval = setInterval(detect, APP_CONFIG.DETECTION.DETECTION_INTERVAL);
        
        // Run first detection
        detect();
    }

    /**
     * Stop detection
     */
    stopDetection() {
        if (this.state.detectionInterval) {
            clearInterval(this.state.detectionInterval);
            this.state.detectionInterval = null;
        }
        
        console.log('🔍 Stopped object detection');
    }

    /**
     * Process detections
     */
    async processDetections(detections) {
        // Get top detection
        const topDetection = detections[0];
        
        if (!topDetection) return;
        
        // Check if already processed recently
        const key = `${topDetection.class}_${Math.round(Date.now() / APP_CONFIG.DETECTION.TRANSLATION_DELAY)}`;
        
        if (!this.state.processedObjects.has(key)) {
            this.state.processedObjects.add(key);
            
            // Clean old entries
            if (this.state.processedObjects.size > 100) {
                const entries = Array.from(this.state.processedObjects);
                entries.slice(0, 50).forEach(e => this.state.processedObjects.delete(e));
            }
            
            // Translate
            try {
                console.log('Translating:', topDetection.class);
                const translation = await this.translateText(topDetection.class, this.state.targetLanguage, 'en');
                
                if (translation) {
                    // Store translation
                    topDetection.translation = translation.translated_text;
                    
                    // Speak translation
                    this.speakText(translation.translated_text, this.state.targetLanguage);
                    
                    // Update stats
                    this.performanceMonitor.totalTranslations++;
                }
            } catch (error) {
                console.error('Translation error:', error);
            }
        }
        
        // Update stats
        this.performanceMonitor.totalDetections += detections.length;
    }

    /**
     * Draw detections on canvas
     */
    drawDetections(detections) {
        const canvas = this.elements.detectionCanvas;
        const ctx = canvas.getContext('2d');
        const video = this.elements.videoElement;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Calculate scale
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;
        
        // Color map for categories
        const colors = {
            'People': '#FF6B6B',
            'Animal': '#4ECDC4',
            'Vehicle': '#45B7D1',
            'Food': '#FFA07A',
            'Furniture': '#98D8C8',
            'Electronics': '#A8E6CF',
            'Object': '#C7CEEA'
        };
        
        // Draw each detection
        detections.forEach((detection, index) => {
            const [x, y, width, height] = detection.bbox;
            const category = detection.category || 'Object';
            const color = colors[category] || '#00FF00';
            
            // Scale coordinates
            const sx = x * scaleX;
            const sy = y * scaleY;
            const sw = width * scaleX;
            const sh = height * scaleY;
            
            // Draw box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(sx, sy, sw, sh);
            
            // Draw corners
            this.drawCorners(ctx, sx, sy, sw, sh, color);
            
            // Draw label
            const label = detection.class;
            const confidence = Math.round(detection.confidence * 100);
            const translation = detection.translation || '';
            
            // Background
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            const labelHeight = translation ? 60 : 40;
            ctx.fillRect(sx, sy - labelHeight, sw, labelHeight);
            
            // Text
            ctx.fillStyle = '#FFFFFF';
            ctx.font = 'bold 14px Arial';
            ctx.fillText(label, sx + 5, sy - labelHeight + 20);
            
            ctx.font = '12px Arial';
            ctx.fillText(`${confidence}%`, sx + 5, sy - labelHeight + 35);
            
            if (translation) {
                ctx.fillStyle = '#00FF00';
                ctx.fillText(translation, sx + 5, sy - labelHeight + 50);
            }
            
            // Category indicator
            ctx.fillStyle = color;
            ctx.fillRect(sx, sy - labelHeight, 3, labelHeight);
        });
        
        // Draw performance stats
        this.drawPerformanceStats(ctx, canvas);
    }

    /**
     * Draw corners
     */
    drawCorners(ctx, x, y, width, height, color) {
        const cornerLength = 15;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        
        // Top-left
        ctx.beginPath();
        ctx.moveTo(x, y + cornerLength);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLength, y);
        ctx.stroke();
        
        // Top-right
        ctx.beginPath();
        ctx.moveTo(x + width - cornerLength, y);
        ctx.lineTo(x + width, y);
        ctx.lineTo(x + width, y + cornerLength);
        ctx.stroke();
        
        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(x, y + height - cornerLength);
        ctx.lineTo(x, y + height);
        ctx.lineTo(x + cornerLength, y + height);
        ctx.stroke();
        
        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(x + width - cornerLength, y + height);
        ctx.lineTo(x + width, y + height);
        ctx.lineTo(x + width, y + height - cornerLength);
        ctx.stroke();
    }

    /**
     * Draw performance stats
     */
    drawPerformanceStats(ctx, canvas) {
        const stats = this.detector.getPerformanceStats();
        
        const lines = [
            `FPS: ${stats.fps}`,
            `Model: ${stats.modelType}`,
            `Objects: ${stats.cacheSize}`
        ];
        
        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(canvas.width - 140, 10, 130, 60);
        
        // Text
        ctx.fillStyle = '#00FF00';
        ctx.font = '12px monospace';
        lines.forEach((line, i) => {
            ctx.fillText(line, canvas.width - 135, 30 + i * 15);
        });
    }

    /**
     * Update detection display
     */
    updateDetectionDisplay(detections) {
        const container = this.elements.detectedObjects;
        if (!container) return;
        
        // Group by category
        const grouped = {};
        detections.forEach(det => {
            const cat = det.category || 'Object';
            if (!grouped[cat]) grouped[cat] = [];
            grouped[cat].push(det);
        });
        
        // Build HTML
        let html = '<div class="detection-list">';
        
        Object.entries(grouped).forEach(([category, items]) => {
            html += `
                <div class="detection-category">
                    <div class="category-title">${category}</div>
                    <div class="category-items">
            `;
            
            items.forEach(item => {
                const confidence = Math.round(item.confidence * 100);
                html += `
                    <div class="detection-item">
                        <div class="item-info">
                            <div class="item-name">${item.class}</div>
                            ${item.translation ? 
                                `<div class="item-translation">${item.translation}</div>` : 
                                ''
                            }
                        </div>
                        <div class="item-confidence">
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${confidence}%"></div>
                            </div>
                            <span>${confidence}%</span>
                        </div>
                    </div>
                `;
            });
            
            html += '</div></div>';
        });
        
        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * Update performance display
     */
    updatePerformanceDisplay() {
        if (!this.elements.performanceStats) return;
        
        const stats = {
            uptime: Math.floor((Date.now() - this.performanceMonitor.startTime) / 1000),
            translations: this.performanceMonitor.totalTranslations,
            detections: this.performanceMonitor.totalDetections,
            ...this.detector.getPerformanceStats()
        };
        
        const html = `
            <div class="stats-grid">
                <div class="stat">
                    <span class="stat-label">Uptime</span>
                    <span class="stat-value">${stats.uptime}s</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Translations</span>
                    <span class="stat-value">${stats.translations}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Detections</span>
                    <span class="stat-value">${stats.detections}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">FPS</span>
                    <span class="stat-value">${stats.fps}</span>
                </div>
            </div>
        `;
        
        this.elements.performanceStats.innerHTML = html;
    }

    // ==================== TRANSLATION ====================
    
    /**
     * Translate text
     */
    async translateText(text, targetLang = null, sourceLang = null) {
        const target = targetLang || this.state.targetLanguage;
        const source = sourceLang || this.state.sourceLanguage || 'auto';
        
        try {
            const apiUrl = APP_CONFIG.API_BASE ? 
                `${APP_CONFIG.API_BASE}/translate` : 
                '/translate';
                
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    target_language: target,
                    source_language: source
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.message || 'Translation failed');
            }
            
            this.state.lastTranslation = data;
            return data;
            
        } catch (error) {
            console.error('Translation error:', error);
            throw error;
        }
    }

    // ==================== UI HELPERS ====================
    
    /**
     * Show status message
     */
    showStatus(message, elementId, isActive = false) {
        const element = this.elements[elementId];
        if (element) {
            element.textContent = message;
            element.classList.toggle('active', isActive);
        }
    }

    /**
     * Update display
     */
    updateDisplay(elementId, content) {
        const element = this.elements[elementId];
        if (element) {
            if (typeof content === 'string') {
                element.textContent = content;
            } else {
                element.innerHTML = content;
            }
        }
    }

    /**
     * Show error
     */
    showError(message, dismissible = true) {
        console.error('Error:', message);
        
        if (this.elements.errorMessage) {
            this.elements.errorMessage.textContent = message;
            this.elements.errorMessage.classList.remove('hidden');
            
            if (dismissible) {
                setTimeout(() => this.hideError(), 5000);
            }
        }
    }

    /**
     * Hide error
     */
    hideError() {
        if (this.elements.errorMessage) {
            this.elements.errorMessage.classList.add('hidden');
        }
    }

    /**
     * Show warning
     */
    showWarning(message) {
        console.warn('Warning:', message);
        this.showError(message, true);
    }
}

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Starting Real-Time Translator...');
    
    // Check dependencies
    if (typeof tf === 'undefined') {
        console.warn('TensorFlow.js not loaded - detection may be limited');
    }
    
    if (typeof cocoSsd === 'undefined') {
        console.warn('COCO-SSD not loaded - using fallback detection');
    }
    
    // Initialize application
    window.translator = new RealTimeTranslator();
});

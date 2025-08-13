#!/usr/bin/env python3
"""
Real-Time Translator Backend - Fixed Version
With proper language code handling and error fixes
"""

import logging
import time
import re
import json
import os
import hashlib
from functools import wraps, lru_cache
from datetime import datetime
from typing import Dict, List, Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from googletrans import Translator, LANGUAGES

# ==================== CONFIGURATION ====================

class Config:
    """Application configuration"""
    DEBUG = True
    HOST = '127.0.0.1'
    PORT = 5000
    MAX_TEXT_LENGTH = 10000
    MAX_CHUNK_SIZE = 500
    TRANSLATION_TIMEOUT = 30
    MAX_RETRIES = 3
    CACHE_ENABLED = False  # Disable Redis for now
    LOG_LEVEL = logging.INFO

# ==================== LOGGING ====================

logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== SIMPLE CACHE ====================

class SimpleCache:
    """Simple in-memory cache"""
    def __init__(self):
        self.cache = {}
        self.max_size = 100
        
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = value

# ==================== TRANSLATION MANAGER ====================

class TranslationManager:
    """Translation management"""
    
    def __init__(self):
        self.translator = Translator()
        self.cache = SimpleCache()
        self.stats = {
            'total_translations': 0,
            'total_characters': 0,
            'errors': 0
        }
    
    def translate(self, text: str, target_lang: str, source_lang: str = 'auto') -> Dict:
        """Translate text with caching and error handling"""
        
        try:
            # Validate and clean inputs
            text = self._validate_text(text)
            target_lang = self._normalize_language_code(target_lang)
            source_lang = self._normalize_language_code(source_lang)
            
            # Check cache
            cache_key = f"{text}:{target_lang}:{source_lang}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit for: {text[:50]}...")
                return cached
            
            # Perform translation
            result = self._translate_with_retry(text, target_lang, source_lang)
            
            # Cache result
            self.cache.set(cache_key, result)
            
            # Update stats
            self.stats['total_translations'] += 1
            self.stats['total_characters'] += len(text)
            
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            self.stats['errors'] += 1
            raise
    
    def translate_long(self, text: str, target_lang: str, source_lang: str = 'auto') -> Dict:
        """Translate long text by chunking"""
        
        chunks = self._chunk_text(text, Config.MAX_CHUNK_SIZE)
        translated_chunks = []
        detected_lang = 'auto'
        failed_chunks = 0
        
        logger.info(f"Processing {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            try:
                result = self.translate(chunk, target_lang, source_lang)
                translated_chunks.append(result['translated_text'])
                
                if i == 0:
                    detected_lang = result.get('source_language', 'auto')
                    
            except Exception as e:
                logger.error(f"Chunk {i+1} failed: {e}")
                translated_chunks.append(chunk)
                failed_chunks += 1
        
        success_rate = (len(chunks) - failed_chunks) / len(chunks) if chunks else 0
        
        return {
            'original_text': text,
            'translated_text': ' '.join(translated_chunks),
            'source_language': detected_lang,
            'target_language': target_lang,
            'chunks_processed': len(chunks),
            'chunks_failed': failed_chunks,
            'success_rate': success_rate,
            'confidence': max(0.5, 0.95 * success_rate)
        }
    
    def _translate_with_retry(self, text: str, target_lang: str, source_lang: str) -> Dict:
        """Translate with retry logic"""
        
        last_error = None
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                result = self.translator.translate(text, dest=target_lang, src=source_lang)
                
                if not result or not result.text:
                    raise ValueError("Empty translation result")
                
                return {
                    'original_text': text,
                    'translated_text': result.text,
                    'source_language': result.src,
                    'target_language': target_lang,
                    'confidence': 0.95,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                last_error = e
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(0.5 * (attempt + 1))
        
        raise Exception(f"Translation failed after {Config.MAX_RETRIES} attempts: {last_error}")
    
    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks"""
        if len(text) <= max_length:
            return [text]
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _validate_text(self, text: str) -> str:
        """Validate and sanitize text"""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        
        # Remove dangerous characters but keep regular punctuation
        text = re.sub(r'[<>]', '', text)
        text = text.strip()
        
        if not text:
            raise ValueError("Text cannot be empty")
        
        if len(text) > Config.MAX_TEXT_LENGTH:
            raise ValueError(f"Text too long (max {Config.MAX_TEXT_LENGTH} characters)")
        
        return text
    
    def _normalize_language_code(self, lang_code: str) -> str:
        """Normalize language code to standard format"""
        if not isinstance(lang_code, str):
            raise ValueError("Language code must be a string")
        
        if lang_code == 'auto':
            return lang_code
        
        lang_code = lang_code.lower().strip()
        
        # Handle variants like en-gb, zh-cn, etc.
        if '-' in lang_code:
            base_code = lang_code.split('-')[0]
            
            # Map common variants to googletrans codes
            variant_map = {
                'zh-cn': 'zh-cn',
                'zh-tw': 'zh-tw',
                'pt-br': 'pt',
                'pt-pt': 'pt',
                'en-us': 'en',
                'en-gb': 'en',
                'es-es': 'es',
                'es-mx': 'es',
                'fr-fr': 'fr',
                'fr-ca': 'fr'
            }
            
            if lang_code in variant_map:
                return variant_map[lang_code]
            else:
                lang_code = base_code
        
        # Validate against googletrans languages
        if lang_code not in LANGUAGES:
            # Try to find closest match
            for code in LANGUAGES:
                if code.startswith(lang_code) or lang_code.startswith(code):
                    return code
            
            # If still not found, raise error
            raise ValueError(f"Unsupported language code: {lang_code}. Supported: {', '.join(list(LANGUAGES.keys())[:10])}...")
        
        return lang_code

# ==================== FLASK APPLICATION ====================

# Initialize components
translation_manager = TranslationManager()

# Initialize Flask
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Enable CORS
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])

# ==================== ERROR HANDLERS ====================

def handle_errors(f):
    """Decorator for consistent error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error in {f.__name__}: {e}")
            return jsonify({
                'error': 'Validation error',
                'message': str(e)
            }), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}", exc_info=True)
            return jsonify({
                'error': 'Internal server error',
                'message': str(e)  # Include actual error for debugging
            }), 500
    return decorated_function

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve main application"""
    try:
        return send_from_directory('static', 'index.html')
    except:
        return jsonify({'message': 'API is running. Frontend not found in /static folder'}), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Real-Time Translator API is running',
        'version': '2.0',
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/translate', methods=['POST', 'OPTIONS'])
@handle_errors
def translate_text():
    """Translate text endpoint"""
    
    # Handle OPTIONS request for CORS
    if request.method == 'OPTIONS':
        return '', 204
    
    if not request.is_json:
        return jsonify({
            'error': 'Invalid request',
            'message': 'Request must be JSON'
        }), 400
    
    data = request.get_json()
    
    # Validate required fields
    if not data or 'text' not in data or 'target_language' not in data:
        return jsonify({
            'error': 'Missing required fields',
            'message': 'Both text and target_language are required'
        }), 400
    
    # Extract parameters
    text = data['text']
    target_lang = data['target_language']
    source_lang = data.get('source_language', 'auto')
    
    logger.info(f"Translation request: '{text[:50]}...' from {source_lang} to {target_lang}")
    
    try:
        # Check if long text
        if len(text) > Config.MAX_CHUNK_SIZE:
            result = translation_manager.translate_long(text, target_lang, source_lang)
        else:
            result = translation_manager.translate(text, target_lang, source_lang)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({
            'error': 'Translation failed',
            'message': str(e)
        }), 500

@app.route('/translate-long', methods=['POST', 'OPTIONS'])
@handle_errors
def translate_long_text():
    """Translate long text endpoint"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    if not request.is_json:
        return jsonify({
            'error': 'Invalid request',
            'message': 'Request must be JSON'
        }), 400
    
    data = request.get_json()
    
    if not data or 'text' not in data or 'target_language' not in data:
        return jsonify({
            'error': 'Missing required fields',
            'message': 'Both text and target_language are required'
        }), 400
    
    text = data['text']
    target_lang = data['target_language']
    source_lang = data.get('source_language', 'auto')
    
    logger.info(f"Long text translation: {len(text)} chars -> {target_lang}")
    
    result = translation_manager.translate_long(text, target_lang, source_lang)
    
    if result['success_rate'] >= 0.8:
        return jsonify(result), 200
    elif result['success_rate'] >= 0.5:
        return jsonify(result), 206
    else:
        return jsonify({
            **result,
            'error': 'Translation mostly failed',
            'message': 'Most chunks failed to translate'
        }), 503

@app.route('/languages', methods=['GET'])
@handle_errors
def get_supported_languages():
    """Get supported languages endpoint"""
    
    # Core languages with proper display names
    languages = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh-cn': 'Chinese (Simplified)',
        'zh-tw': 'Chinese (Traditional)',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'no': 'Norwegian',
        'da': 'Danish',
        'fi': 'Finnish',
        'pl': 'Polish',
        'tr': 'Turkish',
        'th': 'Thai',
        'vi': 'Vietnamese',
        'id': 'Indonesian',
        'ms': 'Malay',
        'fa': 'Persian',
        'he': 'Hebrew',
        'ur': 'Urdu',
        'bn': 'Bengali',
        'ta': 'Tamil',
        'te': 'Telugu',
        'mr': 'Marathi',
        'gu': 'Gujarati',
        'pa': 'Punjabi'
    }
    
    logger.info(f"Languages requested, returning {len(languages)} languages")
    
    return jsonify({
        'languages': languages,
        'count': len(languages),
        'default': 'en'
    }), 200

@app.route('/detect', methods=['POST'])
@handle_errors
def detect_language():
    """Detect language of text"""
    
    if not request.is_json:
        return jsonify({
            'error': 'Invalid request',
            'message': 'Request must be JSON'
        }), 400
    
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({
            'error': 'Missing required field',
            'message': 'Text is required'
        }), 400
    
    text = translation_manager._validate_text(data['text'])
    
    try:
        result = translation_manager.translator.detect(text)
        
        return jsonify({
            'detected_language': result.lang,
            'confidence': getattr(result, 'confidence', 0.95),
            'text': text[:100] + '...' if len(text) > 100 else text
        }), 200
        
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return jsonify({
            'error': 'Detection failed',
            'message': 'Could not detect language'
        }), 500

# ==================== ERROR HANDLERS ====================

@app.errorhandler(400)
def bad_request(error):
    logger.warning(f"Bad request: {error}")
    return jsonify({
        'error': 'Bad request',
        'message': 'Invalid request format or parameters'
    }), 400

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"Endpoint not found: {request.url}")
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    logger.info("="*50)
    logger.info("Real-Time Translator API v2.0 - Starting...")
    logger.info("="*50)
    
    try:
        test_result = translation_manager.translator.translate("Hello", dest='es')
        logger.info(f"✅ Translation service working: Hello -> {test_result.text}")
    except Exception as e:
        logger.error(f"❌ Translation service test failed: {e}")
        logger.warning("Server will start but translations may not work")
    
    logger.info("Configuration:")
    logger.info(f"  - Host: 0.0.0.0")
    logger.info(f"  - Port: {os.environ.get('PORT', 5000)}")
    logger.info(f"  - Debug: {Config.DEBUG}")
    logger.info(f"  - CORS: Enabled for all origins")
    logger.info("="*50)
    
    # Run application
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=Config.DEBUG,
        threaded=True
    )




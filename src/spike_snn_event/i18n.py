"""
Internationalization (i18n) and localization support for spike-snn-event-vision-kit.

Provides multi-language support for global deployment across different regions
and cultures, with support for 6 primary languages as specified in requirements.
"""

import os
import json
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class SupportedLanguage(Enum):
    """Supported languages for the neuromorphic vision system."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


@dataclass
class LocalizationConfig:
    """Configuration for localization settings."""
    default_language: SupportedLanguage = SupportedLanguage.ENGLISH
    fallback_language: SupportedLanguage = SupportedLanguage.ENGLISH
    auto_detect_locale: bool = True
    cache_translations: bool = True
    region_code: Optional[str] = None


class I18nManager:
    """Manager for internationalization and localization."""
    
    def __init__(self, config: Optional[LocalizationConfig] = None):
        self.config = config or LocalizationConfig()
        self.current_language = self.config.default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize translation cache
        self._load_translations()
        
        # Auto-detect locale if enabled
        if self.config.auto_detect_locale:
            self._auto_detect_language()
    
    def _load_translations(self):
        """Load translation files for all supported languages."""
        # Define core neuromorphic vision translations
        self.translations = {
            SupportedLanguage.ENGLISH.value: {
                # Core system messages
                "system.started": "Neuromorphic vision system started",
                "system.stopped": "System stopped",
                "system.error": "System error occurred",
                "system.ready": "System ready for processing",
                
                # Event processing
                "events.processing": "Processing event stream",
                "events.detected": "Events detected: {count}",
                "events.filtered": "Events filtered: {count}",
                "events.queue_full": "Event queue is full, dropping events",
                
                # Neural network operations
                "snn.inference": "Running SNN inference",
                "snn.training": "Training spiking neural network",
                "snn.model_loaded": "Model loaded: {model_name}",
                "snn.accuracy": "Model accuracy: {accuracy:.2f}%",
                
                # Hardware integration
                "hardware.camera_connected": "Event camera connected: {camera_type}",
                "hardware.gpu_detected": "GPU detected: {gpu_info}",
                "hardware.loihi_available": "Intel Loihi hardware available",
                
                # Performance metrics
                "perf.latency": "Average latency: {latency:.2f}ms",
                "perf.throughput": "Throughput: {events_per_sec} events/sec",
                "perf.memory_usage": "Memory usage: {percentage:.1f}%",
                
                # Error messages
                "error.validation": "Input validation failed: {details}",
                "error.camera": "Camera error: {error}",
                "error.model": "Model error: {error}",
                "error.memory": "Out of memory",
                
                # Configuration
                "config.loaded": "Configuration loaded from {source}",
                "config.invalid": "Invalid configuration: {details}",
                
                # Security
                "security.threat_detected": "Security threat detected: {threat_type}",
                "security.access_denied": "Access denied: insufficient permissions",
            },
            
            SupportedLanguage.SPANISH.value: {
                "system.started": "Sistema de visión neuromórfica iniciado",
                "system.stopped": "Sistema detenido",
                "system.error": "Error del sistema ocurrido",
                "system.ready": "Sistema listo para procesamiento",
                
                "events.processing": "Procesando flujo de eventos",
                "events.detected": "Eventos detectados: {count}",
                "events.filtered": "Eventos filtrados: {count}",
                "events.queue_full": "Cola de eventos llena, descartando eventos",
                
                "snn.inference": "Ejecutando inferencia SNN",
                "snn.training": "Entrenando red neuronal de impulsos",
                "snn.model_loaded": "Modelo cargado: {model_name}",
                "snn.accuracy": "Precisión del modelo: {accuracy:.2f}%",
                
                "hardware.camera_connected": "Cámara de eventos conectada: {camera_type}",
                "hardware.gpu_detected": "GPU detectada: {gpu_info}",
                "hardware.loihi_available": "Hardware Intel Loihi disponible",
                
                "perf.latency": "Latencia promedio: {latency:.2f}ms",
                "perf.throughput": "Rendimiento: {events_per_sec} eventos/seg",
                "perf.memory_usage": "Uso de memoria: {percentage:.1f}%",
                
                "error.validation": "Validación de entrada falló: {details}",
                "error.camera": "Error de cámara: {error}",
                "error.model": "Error de modelo: {error}",
                "error.memory": "Sin memoria",
                
                "config.loaded": "Configuración cargada desde {source}",
                "config.invalid": "Configuración inválida: {details}",
                
                "security.threat_detected": "Amenaza de seguridad detectada: {threat_type}",
                "security.access_denied": "Acceso denegado: permisos insuficientes",
            },
            
            SupportedLanguage.FRENCH.value: {
                "system.started": "Système de vision neuromorphique démarré",
                "system.stopped": "Système arrêté",
                "system.error": "Erreur système survenue",
                "system.ready": "Système prêt pour le traitement",
                
                "events.processing": "Traitement du flux d'événements",
                "events.detected": "Événements détectés: {count}",
                "events.filtered": "Événements filtrés: {count}",
                "events.queue_full": "File d'événements pleine, abandon d'événements",
                
                "snn.inference": "Exécution de l'inférence SNN",
                "snn.training": "Entraînement du réseau neuronal à impulsions",
                "snn.model_loaded": "Modèle chargé: {model_name}",
                "snn.accuracy": "Précision du modèle: {accuracy:.2f}%",
                
                "hardware.camera_connected": "Caméra d'événements connectée: {camera_type}",
                "hardware.gpu_detected": "GPU détecté: {gpu_info}",
                "hardware.loihi_available": "Matériel Intel Loihi disponible",
                
                "perf.latency": "Latence moyenne: {latency:.2f}ms",
                "perf.throughput": "Débit: {events_per_sec} événements/sec",
                "perf.memory_usage": "Utilisation mémoire: {percentage:.1f}%",
                
                "error.validation": "Validation d'entrée échouée: {details}",
                "error.camera": "Erreur caméra: {error}",
                "error.model": "Erreur modèle: {error}",
                "error.memory": "Mémoire insuffisante",
                
                "config.loaded": "Configuration chargée depuis {source}",
                "config.invalid": "Configuration invalide: {details}",
                
                "security.threat_detected": "Menace de sécurité détectée: {threat_type}",
                "security.access_denied": "Accès refusé: permissions insuffisantes",
            },
            
            SupportedLanguage.GERMAN.value: {
                "system.started": "Neuromorphes Sehsystem gestartet",
                "system.stopped": "System gestoppt",
                "system.error": "Systemfehler aufgetreten",
                "system.ready": "System bereit für Verarbeitung",
                
                "events.processing": "Verarbeitung des Ereignisstroms",
                "events.detected": "Ereignisse erkannt: {count}",
                "events.filtered": "Ereignisse gefiltert: {count}",
                "events.queue_full": "Ereigniswarteschlange voll, verwerfe Ereignisse",
                
                "snn.inference": "SNN-Inferenz ausführen",
                "snn.training": "Training des Spiking Neural Network",
                "snn.model_loaded": "Modell geladen: {model_name}",
                "snn.accuracy": "Modellgenauigkeit: {accuracy:.2f}%",
                
                "hardware.camera_connected": "Ereigniskamera verbunden: {camera_type}",
                "hardware.gpu_detected": "GPU erkannt: {gpu_info}",
                "hardware.loihi_available": "Intel Loihi Hardware verfügbar",
                
                "perf.latency": "Durchschnittliche Latenz: {latency:.2f}ms",
                "perf.throughput": "Durchsatz: {events_per_sec} Ereignisse/Sek",
                "perf.memory_usage": "Speicherverbrauch: {percentage:.1f}%",
                
                "error.validation": "Eingabevalidierung fehlgeschlagen: {details}",
                "error.camera": "Kamerafehler: {error}",
                "error.model": "Modellfehler: {error}",
                "error.memory": "Speicher nicht ausreichend",
                
                "config.loaded": "Konfiguration geladen von {source}",
                "config.invalid": "Ungültige Konfiguration: {details}",
                
                "security.threat_detected": "Sicherheitsbedrohung erkannt: {threat_type}",
                "security.access_denied": "Zugriff verweigert: unzureichende Berechtigungen",
            },
            
            SupportedLanguage.JAPANESE.value: {
                "system.started": "ニューロモルフィックビジョンシステムが開始されました",
                "system.stopped": "システムが停止しました",
                "system.error": "システムエラーが発生しました",
                "system.ready": "システムが処理準備完了",
                
                "events.processing": "イベントストリームを処理中",
                "events.detected": "検出されたイベント: {count}",
                "events.filtered": "フィルタされたイベント: {count}",
                "events.queue_full": "イベントキューが満杯、イベントを破棄",
                
                "snn.inference": "SNN推論を実行中",
                "snn.training": "スパイキングニューラルネットワークを訓練中",
                "snn.model_loaded": "モデルがロードされました: {model_name}",
                "snn.accuracy": "モデル精度: {accuracy:.2f}%",
                
                "hardware.camera_connected": "イベントカメラが接続されました: {camera_type}",
                "hardware.gpu_detected": "GPU検出: {gpu_info}",
                "hardware.loihi_available": "Intel Loihiハードウェアが利用可能",
                
                "perf.latency": "平均レイテンシ: {latency:.2f}ms",
                "perf.throughput": "スループット: {events_per_sec} イベント/秒",
                "perf.memory_usage": "メモリ使用量: {percentage:.1f}%",
                
                "error.validation": "入力検証失敗: {details}",
                "error.camera": "カメラエラー: {error}",
                "error.model": "モデルエラー: {error}",
                "error.memory": "メモリ不足",
                
                "config.loaded": "設定が読み込まれました: {source}",
                "config.invalid": "無効な設定: {details}",
                
                "security.threat_detected": "セキュリティ脅威検出: {threat_type}",
                "security.access_denied": "アクセス拒否: 権限不足",
            },
            
            SupportedLanguage.CHINESE.value: {
                "system.started": "神经形态视觉系统已启动",
                "system.stopped": "系统已停止",
                "system.error": "发生系统错误",
                "system.ready": "系统准备就绪进行处理",
                
                "events.processing": "正在处理事件流",
                "events.detected": "检测到事件: {count}",
                "events.filtered": "过滤事件: {count}",
                "events.queue_full": "事件队列已满，丢弃事件",
                
                "snn.inference": "运行SNN推理",
                "snn.training": "训练脉冲神经网络",
                "snn.model_loaded": "模型已加载: {model_name}",
                "snn.accuracy": "模型精度: {accuracy:.2f}%",
                
                "hardware.camera_connected": "事件相机已连接: {camera_type}",
                "hardware.gpu_detected": "检测到GPU: {gpu_info}",
                "hardware.loihi_available": "Intel Loihi硬件可用",
                
                "perf.latency": "平均延迟: {latency:.2f}ms",
                "perf.throughput": "吞吐量: {events_per_sec} 事件/秒",
                "perf.memory_usage": "内存使用: {percentage:.1f}%",
                
                "error.validation": "输入验证失败: {details}",
                "error.camera": "相机错误: {error}",
                "error.model": "模型错误: {error}",
                "error.memory": "内存不足",
                
                "config.loaded": "配置已从{source}加载",
                "config.invalid": "无效配置: {details}",
                
                "security.threat_detected": "检测到安全威胁: {threat_type}",
                "security.access_denied": "拒绝访问: 权限不足",
            }
        }
        
        self.logger.info(f"Loaded translations for {len(self.translations)} languages")
    
    def _auto_detect_language(self):
        """Auto-detect language from environment."""
        try:
            import locale
            system_locale = locale.getdefaultlocale()[0]
            
            if system_locale:
                lang_code = system_locale.split('_')[0].lower()
                
                # Map to supported languages
                for supported_lang in SupportedLanguage:
                    if supported_lang.value == lang_code:
                        self.current_language = supported_lang
                        self.logger.info(f"Auto-detected language: {lang_code}")
                        return
                        
            # Check environment variables
            for env_var in ['LANG', 'LANGUAGE', 'LC_ALL']:
                env_lang = os.environ.get(env_var, '').lower()
                if env_lang:
                    for supported_lang in SupportedLanguage:
                        if supported_lang.value in env_lang:
                            self.current_language = supported_lang
                            self.logger.info(f"Language from {env_var}: {supported_lang.value}")
                            return
                            
        except Exception as e:
            self.logger.warning(f"Failed to auto-detect language: {e}")
            
    def set_language(self, language: SupportedLanguage):
        """Set current language."""
        self.current_language = language
        self.logger.info(f"Language set to: {language.value}")
        
    def get_language(self) -> SupportedLanguage:
        """Get current language."""
        return self.current_language
        
    def translate(self, key: str, **kwargs) -> str:
        """Translate a message key with optional formatting parameters."""
        try:
            # Get translation for current language
            lang_translations = self.translations.get(self.current_language.value, {})
            
            # Fallback to default language if not found
            if key not in lang_translations:
                lang_translations = self.translations.get(self.config.fallback_language.value, {})
                
            # Get the translation or return key if not found
            translation = lang_translations.get(key, key)
            
            # Format with provided parameters
            if kwargs:
                translation = translation.format(**kwargs)
                
            return translation
            
        except Exception as e:
            self.logger.warning(f"Translation error for key '{key}': {e}")
            return key
    
    def get_available_languages(self) -> List[SupportedLanguage]:
        """Get list of available languages."""
        return list(SupportedLanguage)
        
    def get_language_info(self, language: SupportedLanguage) -> Dict[str, str]:
        """Get information about a language."""
        language_info = {
            SupportedLanguage.ENGLISH: {
                "name": "English",
                "native_name": "English",
                "code": "en",
                "region": "Global"
            },
            SupportedLanguage.SPANISH: {
                "name": "Spanish",
                "native_name": "Español",
                "code": "es", 
                "region": "Latin America, Spain"
            },
            SupportedLanguage.FRENCH: {
                "name": "French",
                "native_name": "Français",
                "code": "fr",
                "region": "France, Canada, Africa"
            },
            SupportedLanguage.GERMAN: {
                "name": "German",
                "native_name": "Deutsch",
                "code": "de",
                "region": "Germany, Austria, Switzerland"
            },
            SupportedLanguage.JAPANESE: {
                "name": "Japanese",
                "native_name": "日本語",
                "code": "ja",
                "region": "Japan"
            },
            SupportedLanguage.CHINESE: {
                "name": "Chinese",
                "native_name": "中文",
                "code": "zh",
                "region": "China, Taiwan, Singapore"
            }
        }
        
        return language_info.get(language, {})
        
    def add_custom_translations(self, language: SupportedLanguage, translations: Dict[str, str]):
        """Add custom translations for a language."""
        lang_code = language.value
        if lang_code not in self.translations:
            self.translations[lang_code] = {}
            
        self.translations[lang_code].update(translations)
        self.logger.info(f"Added {len(translations)} custom translations for {lang_code}")
        
    def export_translations(self, language: SupportedLanguage, file_path: Path):
        """Export translations to JSON file."""
        lang_code = language.value
        translations = self.translations.get(lang_code, {})
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(translations, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Exported {len(translations)} translations for {lang_code} to {file_path}")
        
    def import_translations(self, language: SupportedLanguage, file_path: Path):
        """Import translations from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                
            self.add_custom_translations(language, translations)
            
        except Exception as e:
            self.logger.error(f"Failed to import translations from {file_path}: {e}")


# Global i18n manager instance
_global_i18n = None


def get_i18n_manager() -> I18nManager:
    """Get global i18n manager instance."""
    global _global_i18n
    if _global_i18n is None:
        _global_i18n = I18nManager()
    return _global_i18n


def t(key: str, **kwargs) -> str:
    """Convenience function for translation (shorthand)."""
    return get_i18n_manager().translate(key, **kwargs)


def set_global_language(language: SupportedLanguage):
    """Set global language for the application."""
    get_i18n_manager().set_language(language)


def get_supported_regions() -> Dict[str, List[str]]:
    """Get supported regions and their primary languages."""
    return {
        "North America": ["en"],
        "Latin America": ["es", "en"],
        "Europe": ["en", "fr", "de"],
        "Asia Pacific": ["ja", "zh", "en"],
        "Global": ["en", "es", "fr", "de", "ja", "zh"]
    }


# Localized logging handler
class LocalizedLogHandler(logging.Handler):
    """Logging handler that translates log messages."""
    
    def __init__(self, base_handler: logging.Handler):
        super().__init__()
        self.base_handler = base_handler
        self.i18n = get_i18n_manager()
        
    def emit(self, record):
        """Emit log record with translation."""
        # Try to translate the message if it's a translation key
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if '.' in record.msg and not ' ' in record.msg:
                # Looks like a translation key
                record.msg = self.i18n.translate(record.msg, **(record.args if record.args else {}))
                record.args = ()  # Clear args since we already formatted
                
        self.base_handler.emit(record)
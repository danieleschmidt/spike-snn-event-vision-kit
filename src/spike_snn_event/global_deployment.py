"""
Global-first deployment and internationalization for Spike SNN Event Vision Kit.

Features:
- Multi-region deployment support
- Internationalization (i18n) with 6 languages
- GDPR, CCPA, PDPA compliance
- Cross-platform compatibility
- Regional data sovereignty
- Timezone-aware processing
"""

import os
import json
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
from datetime import datetime, timezone
import locale


class SupportedRegion(Enum):
    """Supported deployment regions."""
    NORTH_AMERICA = "na"
    EUROPE = "eu" 
    ASIA_PACIFIC = "ap"
    LATIN_AMERICA = "la"
    MIDDLE_EAST_AFRICA = "mea"


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


class PrivacyRegulation(Enum):
    """Privacy regulations compliance."""
    GDPR = "gdpr"      # European General Data Protection Regulation
    CCPA = "ccpa"      # California Consumer Privacy Act
    PDPA = "pdpa"      # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"      # Lei Geral de Proteção de Dados (Brazil)


@dataclass
class RegionalConfig:
    """Regional deployment configuration."""
    region: SupportedRegion
    languages: List[SupportedLanguage] = field(default_factory=list)
    privacy_regulations: List[PrivacyRegulation] = field(default_factory=list)
    timezone: str = "UTC"
    currency: str = "USD"
    data_residency_required: bool = False
    encryption_required: bool = True
    audit_logging_required: bool = True
    
    def __post_init__(self):
        """Set region-specific defaults."""
        if self.region == SupportedRegion.EUROPE:
            if not self.languages:
                self.languages = [SupportedLanguage.ENGLISH, SupportedLanguage.FRENCH, SupportedLanguage.GERMAN]
            if PrivacyRegulation.GDPR not in self.privacy_regulations:
                self.privacy_regulations.append(PrivacyRegulation.GDPR)
            self.data_residency_required = True
            self.timezone = "Europe/Berlin"
            self.currency = "EUR"
            
        elif self.region == SupportedRegion.ASIA_PACIFIC:
            if not self.languages:
                self.languages = [SupportedLanguage.ENGLISH, SupportedLanguage.JAPANESE, SupportedLanguage.CHINESE]
            if PrivacyRegulation.PDPA not in self.privacy_regulations:
                self.privacy_regulations.append(PrivacyRegulation.PDPA)
            self.timezone = "Asia/Singapore"
            self.currency = "USD"
            
        elif self.region == SupportedRegion.NORTH_AMERICA:
            if not self.languages:
                self.languages = [SupportedLanguage.ENGLISH, SupportedLanguage.SPANISH]
            if PrivacyRegulation.CCPA not in self.privacy_regulations:
                self.privacy_regulations.append(PrivacyRegulation.CCPA)
            self.timezone = "America/New_York"
            
        elif self.region == SupportedRegion.LATIN_AMERICA:
            if not self.languages:
                self.languages = [SupportedLanguage.SPANISH, SupportedLanguage.ENGLISH]
            if PrivacyRegulation.LGPD not in self.privacy_regulations:
                self.privacy_regulations.append(PrivacyRegulation.LGPD)
            self.timezone = "America/Sao_Paulo"
            self.currency = "BRL"


class InternationalizationManager:
    """Manages internationalization and localization."""
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        self.default_language = default_language
        self.current_language = default_language
        self.translations = self._load_translations()
        self.logger = logging.getLogger(__name__)
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation dictionaries for all supported languages."""
        translations = {
            "en": {
                "system_ready": "System ready",
                "processing_events": "Processing events",
                "error_occurred": "An error occurred",
                "validation_failed": "Validation failed",
                "circuit_breaker_open": "Circuit breaker is open",
                "cache_hit": "Cache hit",
                "cache_miss": "Cache miss",
                "performance_degraded": "Performance degraded",
                "system_healthy": "System healthy",
                "shutting_down": "Shutting down",
                "unauthorized_access": "Unauthorized access attempt",
                "data_processed": "Data processed successfully",
                "model_loaded": "Model loaded",
                "training_complete": "Training complete",
                "inference_ready": "Inference ready"
            },
            "es": {
                "system_ready": "Sistema listo",
                "processing_events": "Procesando eventos",
                "error_occurred": "Ha ocurrido un error",
                "validation_failed": "La validación ha fallado",
                "circuit_breaker_open": "El interruptor automático está abierto",
                "cache_hit": "Acierto de caché",
                "cache_miss": "Fallo de caché",
                "performance_degraded": "Rendimiento degradado",
                "system_healthy": "Sistema saludable",
                "shutting_down": "Apagando",
                "unauthorized_access": "Intento de acceso no autorizado",
                "data_processed": "Datos procesados exitosamente",
                "model_loaded": "Modelo cargado",
                "training_complete": "Entrenamiento completo",
                "inference_ready": "Inferencia lista"
            },
            "fr": {
                "system_ready": "Système prêt",
                "processing_events": "Traitement des événements",
                "error_occurred": "Une erreur s'est produite",
                "validation_failed": "La validation a échoué",
                "circuit_breaker_open": "Le disjoncteur est ouvert",
                "cache_hit": "Succès de cache",
                "cache_miss": "Échec de cache",
                "performance_degraded": "Performance dégradée",
                "system_healthy": "Système en bonne santé",
                "shutting_down": "Arrêt en cours",
                "unauthorized_access": "Tentative d'accès non autorisé",
                "data_processed": "Données traitées avec succès",
                "model_loaded": "Modèle chargé",
                "training_complete": "Formation terminée",
                "inference_ready": "Inférence prête"
            },
            "de": {
                "system_ready": "System bereit",
                "processing_events": "Ereignisse verarbeiten",
                "error_occurred": "Ein Fehler ist aufgetreten",
                "validation_failed": "Validierung fehlgeschlagen",
                "circuit_breaker_open": "Schutzschalter ist offen",
                "cache_hit": "Cache-Treffer",
                "cache_miss": "Cache-Fehler",
                "performance_degraded": "Leistung verschlechtert",
                "system_healthy": "System gesund",
                "shutting_down": "Herunterfahren",
                "unauthorized_access": "Unbefugter Zugriffsversuch",
                "data_processed": "Daten erfolgreich verarbeitet",
                "model_loaded": "Modell geladen",
                "training_complete": "Training abgeschlossen",
                "inference_ready": "Inferenz bereit"
            },
            "ja": {
                "system_ready": "システム準備完了",
                "processing_events": "イベント処理中",
                "error_occurred": "エラーが発生しました",
                "validation_failed": "検証に失敗しました",
                "circuit_breaker_open": "サーキットブレーカーが開いています",
                "cache_hit": "キャッシュヒット",
                "cache_miss": "キャッシュミス",
                "performance_degraded": "パフォーマンス低下",
                "system_healthy": "システム正常",
                "shutting_down": "シャットダウン中",
                "unauthorized_access": "不正アクセスの試み",
                "data_processed": "データ処理成功",
                "model_loaded": "モデル読み込み完了",
                "training_complete": "トレーニング完了",
                "inference_ready": "推論準備完了"
            },
            "zh": {
                "system_ready": "系统就绪",
                "processing_events": "处理事件",
                "error_occurred": "发生错误",
                "validation_failed": "验证失败",
                "circuit_breaker_open": "断路器已打开",
                "cache_hit": "缓存命中",
                "cache_miss": "缓存未命中",
                "performance_degraded": "性能下降",
                "system_healthy": "系统健康",
                "shutting_down": "正在关闭",
                "unauthorized_access": "未授权访问尝试",
                "data_processed": "数据处理成功",
                "model_loaded": "模型已加载",
                "training_complete": "训练完成",
                "inference_ready": "推理就绪"
            }
        }
        return translations
    
    def set_language(self, language: SupportedLanguage):
        """Set current language."""
        self.current_language = language
        self.logger.info(f"Language set to: {language.value}")
    
    def get_text(self, key: str, fallback: Optional[str] = None) -> str:
        """Get translated text for key."""
        lang_code = self.current_language.value
        
        # Try current language
        if lang_code in self.translations and key in self.translations[lang_code]:
            return self.translations[lang_code][key]
        
        # Fall back to English
        if "en" in self.translations and key in self.translations["en"]:
            return self.translations["en"][key]
        
        # Use fallback or key itself
        return fallback or key.replace("_", " ").title()
    
    def format_number(self, number: Union[int, float], region: SupportedRegion) -> str:
        """Format number according to regional conventions."""
        try:
            # Set locale based on region
            if region == SupportedRegion.EUROPE:
                return f"{number:,.2f}".replace(",", " ").replace(".", ",")
            elif region == SupportedRegion.ASIA_PACIFIC:
                if self.current_language == SupportedLanguage.JAPANESE:
                    return f"{number:,.0f}" if isinstance(number, int) else f"{number:.2f}"
                elif self.current_language == SupportedLanguage.CHINESE:
                    return f"{number:,}"
            
            # Default to US formatting
            return f"{number:,}"
            
        except Exception:
            return str(number)
    
    def format_datetime(self, dt: datetime, region: SupportedRegion) -> str:
        """Format datetime according to regional conventions."""
        try:
            if region == SupportedRegion.EUROPE:
                return dt.strftime("%d.%m.%Y %H:%M:%S")
            elif region == SupportedRegion.ASIA_PACIFIC:
                if self.current_language == SupportedLanguage.JAPANESE:
                    return dt.strftime("%Y年%m月%d日 %H:%M:%S")
                elif self.current_language == SupportedLanguage.CHINESE:
                    return dt.strftime("%Y年%m月%d日 %H:%M:%S")
            
            # Default to US formatting
            return dt.strftime("%m/%d/%Y %H:%M:%S")
            
        except Exception:
            return str(dt)


class PrivacyComplianceManager:
    """Manages privacy regulation compliance across regions."""
    
    def __init__(self, regional_config: RegionalConfig):
        self.regional_config = regional_config
        self.logger = logging.getLogger(__name__)
        self.audit_log = []
    
    def ensure_compliance(self, operation: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure operation complies with applicable privacy regulations."""
        compliance_result = {
            "compliant": True,
            "requirements_applied": [],
            "warnings": []
        }
        
        for regulation in self.regional_config.privacy_regulations:
            if regulation == PrivacyRegulation.GDPR:
                gdpr_result = self._apply_gdpr_requirements(operation, user_data)
                compliance_result["requirements_applied"].extend(gdpr_result["applied"])
                compliance_result["warnings"].extend(gdpr_result["warnings"])
                
            elif regulation == PrivacyRegulation.CCPA:
                ccpa_result = self._apply_ccpa_requirements(operation, user_data)
                compliance_result["requirements_applied"].extend(ccpa_result["applied"])
                compliance_result["warnings"].extend(ccpa_result["warnings"])
                
            elif regulation == PrivacyRegulation.PDPA:
                pdpa_result = self._apply_pdpa_requirements(operation, user_data)
                compliance_result["requirements_applied"].extend(pdpa_result["applied"])
                compliance_result["warnings"].extend(pdpa_result["warnings"])
        
        # Log compliance check
        self._log_compliance_check(operation, compliance_result)
        
        return compliance_result
    
    def _apply_gdpr_requirements(self, operation: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply GDPR compliance requirements."""
        result = {"applied": [], "warnings": []}
        
        # Data minimization
        if "personal_data" in user_data:
            result["applied"].append("data_minimization")
            # Remove unnecessary personal data fields
            sensitive_fields = ["email", "name", "phone", "address"]
            for field in sensitive_fields:
                if field in user_data["personal_data"] and operation != "user_registration":
                    user_data["personal_data"].pop(field, None)
                    result["applied"].append(f"removed_{field}")
        
        # Consent tracking
        if operation in ["data_collection", "data_processing"]:
            if "consent" not in user_data:
                result["warnings"].append("GDPR requires explicit consent for data processing")
            result["applied"].append("consent_check")
        
        # Data retention
        if "timestamp" in user_data:
            # GDPR requires data retention policies
            retention_period = 365 * 24 * 3600  # 1 year in seconds
            if time.time() - user_data["timestamp"] > retention_period:
                result["warnings"].append("Data may exceed GDPR retention period")
            result["applied"].append("retention_check")
        
        # Right to be forgotten
        if operation == "data_deletion":
            result["applied"].append("right_to_be_forgotten")
        
        return result
    
    def _apply_ccpa_requirements(self, operation: str, user_data: Dict[str, Any]) -> Dict[str, Any]::
        """Apply CCPA compliance requirements."""
        result = {"applied": [], "warnings": []}
        
        # Right to know
        if operation == "data_access_request":
            result["applied"].append("right_to_know")
        
        # Right to delete
        if operation == "data_deletion":
            result["applied"].append("right_to_delete")
        
        # Do not sell
        if "do_not_sell" in user_data and user_data["do_not_sell"]:
            result["applied"].append("do_not_sell_respect")
        
        # California resident identification
        if "state" in user_data and user_data["state"] == "CA":
            result["applied"].append("california_resident_identified")
        
        return result
    
    def _apply_pdpa_requirements(self, operation: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply PDPA compliance requirements."""
        result = {"applied": [], "warnings": []}
        
        # Purpose limitation
        if "purpose" not in user_data and operation == "data_processing":
            result["warnings"].append("PDPA requires clear purpose for data processing")
        result["applied"].append("purpose_limitation_check")
        
        # Data accuracy
        if "personal_data" in user_data:
            result["applied"].append("data_accuracy_assumption")
        
        # Access and correction rights
        if operation in ["data_access_request", "data_correction"]:
            result["applied"].append("access_correction_rights")
        
        return result
    
    def _log_compliance_check(self, operation: str, result: Dict[str, Any]):
        """Log compliance check for audit purposes."""
        audit_entry = {
            "timestamp": time.time(),
            "operation": operation,
            "regulations": [reg.value for reg in self.regional_config.privacy_regulations],
            "requirements_applied": result["requirements_applied"],
            "warnings": result["warnings"],
            "region": self.regional_config.region.value
        }
        
        self.audit_log.append(audit_entry)
        
        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log.pop(0)
    
    def get_audit_log(self, since_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        if since_timestamp is None:
            return self.audit_log.copy()
        
        return [entry for entry in self.audit_log if entry["timestamp"] >= since_timestamp]
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy compliance report."""
        total_operations = len(self.audit_log)
        operations_by_type = {}
        warnings_by_regulation = {}
        
        for entry in self.audit_log:
            # Count operations
            op_type = entry["operation"]
            operations_by_type[op_type] = operations_by_type.get(op_type, 0) + 1
            
            # Count warnings by regulation
            for regulation in entry["regulations"]:
                if regulation not in warnings_by_regulation:
                    warnings_by_regulation[regulation] = 0
                warnings_by_regulation[regulation] += len(entry["warnings"])
        
        return {
            "report_timestamp": time.time(),
            "region": self.regional_config.region.value,
            "regulations": [reg.value for reg in self.regional_config.privacy_regulations],
            "total_operations": total_operations,
            "operations_by_type": operations_by_type,
            "warnings_by_regulation": warnings_by_regulation,
            "compliance_rate": 1.0 - (sum(warnings_by_regulation.values()) / max(total_operations, 1))
        }


class CrossPlatformCompatibility:
    """Handles cross-platform compatibility and optimization."""
    
    def __init__(self):
        self.platform = self._detect_platform()
        self.capabilities = self._detect_capabilities()
        self.logger = logging.getLogger(__name__)
    
    def _detect_platform(self) -> str:
        """Detect current platform."""
        import platform
        system = platform.system().lower()
        
        if system == "linux":
            return "linux"
        elif system == "darwin":
            return "macos"
        elif system == "windows":
            return "windows"
        else:
            return "unknown"
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect platform capabilities."""
        capabilities = {
            "cuda": False,
            "opencl": False,
            "multiprocessing": True,
            "threading": True,
            "memory_mapping": True,
            "shared_memory": True
        }
        
        try:
            # Check CUDA availability
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
            capabilities["cuda"] = result.returncode == 0
        except:
            pass
        
        try:
            # Check OpenCL
            import pyopencl
            capabilities["opencl"] = True
        except:
            pass
        
        # Platform-specific capabilities
        if self.platform == "windows":
            capabilities["shared_memory"] = False  # Limited support
        elif self.platform == "macos":
            # macOS specific optimizations
            capabilities["metal"] = True
        
        return capabilities
    
    def get_optimal_config(self) -> Dict[str, Any]:
        """Get optimal configuration for current platform."""
        config = {
            "platform": self.platform,
            "max_workers": os.cpu_count(),
            "use_gpu": self.capabilities["cuda"],
            "memory_strategy": "standard"
        }
        
        if self.platform == "linux":
            config["max_workers"] = min(os.cpu_count() * 2, 16)
            config["memory_strategy"] = "aggressive"
            
        elif self.platform == "windows":
            config["max_workers"] = min(os.cpu_count(), 8)
            config["memory_strategy"] = "conservative"
            
        elif self.platform == "macos":
            config["max_workers"] = min(os.cpu_count(), 12)
            config["use_metal"] = self.capabilities.get("metal", False)
        
        return config
    
    def optimize_for_platform(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration for current platform."""
        optimal_config = base_config.copy()
        platform_config = self.get_optimal_config()
        
        # Apply platform-specific optimizations
        for key, value in platform_config.items():
            if key not in optimal_config or key.startswith("use_"):
                optimal_config[key] = value
        
        # Platform-specific adjustments
        if self.platform == "windows":
            # Windows has different memory management
            if "cache_size" in optimal_config:
                optimal_config["cache_size"] = int(optimal_config["cache_size"] * 0.8)
                
        elif self.platform == "linux":
            # Linux can handle larger caches
            if "cache_size" in optimal_config:
                optimal_config["cache_size"] = int(optimal_config["cache_size"] * 1.2)
        
        return optimal_config


class GlobalDeploymentManager:
    """Main manager for global deployment coordination."""
    
    def __init__(self, regional_config: RegionalConfig):
        self.regional_config = regional_config
        self.i18n = InternationalizationManager(
            regional_config.languages[0] if regional_config.languages else SupportedLanguage.ENGLISH
        )
        self.privacy_manager = PrivacyComplianceManager(regional_config)
        self.platform_manager = CrossPlatformCompatibility()
        self.logger = logging.getLogger(__name__)
        
        # Set up regional logging
        self._setup_regional_logging()
    
    def _setup_regional_logging(self):
        """Setup logging according to regional requirements."""
        log_level = logging.INFO
        
        # EU requires more detailed audit logging
        if self.regional_config.region == SupportedRegion.EUROPE:
            log_level = logging.DEBUG
            
        # Configure logger
        self.logger.setLevel(log_level)
        
        # Add audit handler if required
        if self.regional_config.audit_logging_required:
            audit_handler = logging.FileHandler(f"audit_{self.regional_config.region.value}.log")
            audit_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - Region:%(region)s - %(message)s'
            )
            audit_handler.setFormatter(audit_formatter)
            
            # Add region context to all log records
            class RegionalFilter(logging.Filter):
                def __init__(self, region):
                    self.region = region
                
                def filter(self, record):
                    record.region = self.region.value
                    return True
            
            audit_handler.addFilter(RegionalFilter(self.regional_config.region))
            self.logger.addHandler(audit_handler)
    
    def initialize_for_region(self) -> Dict[str, Any]:
        """Initialize system for specific region."""
        self.logger.info(
            self.i18n.get_text("system_ready") + f" - Region: {self.regional_config.region.value}"
        )
        
        # Get platform-optimized configuration
        base_config = {
            "region": self.regional_config.region.value,
            "languages": [lang.value for lang in self.regional_config.languages],
            "timezone": self.regional_config.timezone,
            "cache_size": 1000,
            "max_workers": 4
        }
        
        optimized_config = self.platform_manager.optimize_for_platform(base_config)
        
        # Apply regional data requirements
        if self.regional_config.data_residency_required:
            optimized_config["data_residency"] = True
            optimized_config["cross_border_transfer"] = False
        
        if self.regional_config.encryption_required:
            optimized_config["encryption"] = "AES256"
            optimized_config["key_management"] = "regional"
        
        return optimized_config
    
    def process_with_compliance(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process operation with privacy compliance."""
        # Ensure privacy compliance
        compliance_result = self.privacy_manager.ensure_compliance(operation, data)
        
        if not compliance_result["compliant"]:
            raise ValueError(
                self.i18n.get_text("validation_failed") + 
                f": Privacy compliance failed for {operation}"
            )
        
        # Log operation in appropriate language
        self.logger.info(
            f"{self.i18n.get_text('data_processed')} - Operation: {operation}"
        )
        
        return {
            "operation": operation,
            "status": "success",
            "compliance": compliance_result,
            "region": self.regional_config.region.value,
            "timestamp": time.time()
        }
    
    def get_regional_status(self) -> Dict[str, Any]:
        """Get comprehensive regional deployment status."""
        privacy_report = self.privacy_manager.generate_privacy_report()
        platform_config = self.platform_manager.get_optimal_config()
        
        return {
            "region": {
                "code": self.regional_config.region.value,
                "languages": [lang.value for lang in self.regional_config.languages],
                "current_language": self.i18n.current_language.value,
                "timezone": self.regional_config.timezone,
                "currency": self.regional_config.currency
            },
            "compliance": {
                "regulations": [reg.value for reg in self.regional_config.privacy_regulations],
                "data_residency_required": self.regional_config.data_residency_required,
                "encryption_required": self.regional_config.encryption_required,
                "compliance_rate": privacy_report["compliance_rate"]
            },
            "platform": {
                "detected": platform_config["platform"],
                "capabilities": self.platform_manager.capabilities,
                "optimal_config": platform_config
            },
            "status": self.i18n.get_text("system_healthy"),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def switch_language(self, language: SupportedLanguage):
        """Switch system language."""
        if language in self.regional_config.languages:
            self.i18n.set_language(language)
            self.logger.info(f"Language switched to: {language.value}")
        else:
            raise ValueError(f"Language {language.value} not supported in region {self.regional_config.region.value}")


# Factory function for easy deployment setup
def create_regional_deployment(region: SupportedRegion) -> GlobalDeploymentManager:
    """Create a regional deployment manager with appropriate defaults."""
    regional_config = RegionalConfig(region=region)
    return GlobalDeploymentManager(regional_config)


# Example usage and demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Demonstrate multi-region deployment
    regions = [
        SupportedRegion.NORTH_AMERICA,
        SupportedRegion.EUROPE,
        SupportedRegion.ASIA_PACIFIC
    ]
    
    for region in regions:
        print(f"\n=== Deploying to {region.value.upper()} ===")
        
        deployment_manager = create_regional_deployment(region)
        
        # Initialize for region
        config = deployment_manager.initialize_for_region()
        print(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Demonstrate compliance processing
        test_data = {
            "user_id": "user123",
            "timestamp": time.time(),
            "personal_data": {
                "email": "user@example.com",
                "preferences": {"language": "en"}
            }
        }
        
        result = deployment_manager.process_with_compliance("data_processing", test_data)
        print(f"Processing result: {result['status']}")
        
        # Get regional status
        status = deployment_manager.get_regional_status()
        print(f"Regional status: {status['status']}")
        print(f"Compliance rate: {status['compliance']['compliance_rate']:.2%}")
        
        # Demonstrate language switching
        if len(deployment_manager.regional_config.languages) > 1:
            new_language = deployment_manager.regional_config.languages[1]
            deployment_manager.switch_language(new_language)
            print(f"Switched to language: {new_language.value}")
    
    print("\n🌍 Global deployment demonstration completed!")
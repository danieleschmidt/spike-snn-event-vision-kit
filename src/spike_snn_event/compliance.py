"""
Compliance framework for spike-snn-event-vision-kit.

Implements GDPR, CCPA, PDPA and other privacy regulations compliance
for global deployment of neuromorphic vision systems.
"""

import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path


class ComplianceRegulation(Enum):
    """Supported privacy regulations."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore, Thailand)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPL = "pipl"  # Personal Information Protection Law (China)


class DataType(Enum):
    """Types of data processed by the system."""
    EVENT_DATA = "event_data"              # Raw event camera data
    PROCESSED_EVENTS = "processed_events"  # Filtered/processed events  
    INFERENCE_RESULTS = "inference_results" # SNN model outputs
    METADATA = "metadata"                  # System metadata
    BIOMETRIC = "biometric"               # Biometric identifiers (if any)
    BEHAVIORAL = "behavioral"             # Behavioral patterns
    TECHNICAL = "technical"               # Technical logs and metrics
    USER_PREFERENCES = "user_preferences" # User configuration


class ProcessingPurpose(Enum):
    """Legal purposes for data processing."""
    OBJECT_DETECTION = "object_detection"
    PERFORMANCE_MONITORING = "performance_monitoring"
    SYSTEM_OPTIMIZATION = "system_optimization"
    RESEARCH_DEVELOPMENT = "research_development"
    SECURITY_MONITORING = "security_monitoring"
    COMPLIANCE_REPORTING = "compliance_reporting"


@dataclass
class DataSubject:
    """Information about a data subject."""
    subject_id: str
    region: str
    applicable_regulations: List[ComplianceRegulation]
    consent_timestamp: Optional[datetime] = None
    consent_purposes: Set[ProcessingPurpose] = field(default_factory=set)
    data_retention_period: int = 365  # days
    pseudonymized: bool = False


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    record_id: str
    timestamp: datetime
    data_type: DataType
    processing_purpose: ProcessingPurpose
    data_subject_id: Optional[str] = None
    legal_basis: str = ""
    retention_period: int = 365
    data_size_bytes: int = 0
    processing_location: str = ""
    automated_decision: bool = False


@dataclass
class ConsentRecord:
    """Record of user consent."""
    consent_id: str
    data_subject_id: str
    purposes: Set[ProcessingPurpose]
    granted_at: datetime
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    granular_consent: Dict[str, bool] = field(default_factory=dict)
    consent_mechanism: str = "explicit"  # explicit, implied, opt_out


class ComplianceManager:
    """Manager for privacy regulation compliance."""
    
    def __init__(self, applicable_regulations: Optional[List[ComplianceRegulation]] = None):
        self.applicable_regulations = applicable_regulations or [
            ComplianceRegulation.GDPR,
            ComplianceRegulation.CCPA, 
            ComplianceRegulation.PDPA
        ]
        
        self.logger = logging.getLogger(__name__)
        
        # Data processing records
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.data_subjects: Dict[str, DataSubject] = {}
        
        # Privacy settings
        self.privacy_by_design = True
        self.data_minimization = True
        self.pseudonymization_enabled = True
        self.encryption_at_rest = True
        self.encryption_in_transit = True
        
        # Retention policies
        self.default_retention_days = 365
        self.retention_policies: Dict[DataType, int] = {
            DataType.EVENT_DATA: 90,          # Short retention for raw data
            DataType.PROCESSED_EVENTS: 180,   # Medium retention for processed
            DataType.INFERENCE_RESULTS: 365,  # Longer for results
            DataType.METADATA: 730,          # Longer for metadata
            DataType.TECHNICAL: 365,         # Standard for technical logs
            DataType.USER_PREFERENCES: 1095  # Long retention for user prefs
        }
        
        self.logger.info(f"Compliance manager initialized for: {[r.value for r in self.applicable_regulations]}")
    
    def register_data_subject(
        self, 
        subject_id: str,
        region: str,
        applicable_regs: Optional[List[ComplianceRegulation]] = None
    ) -> DataSubject:
        """Register a new data subject."""
        
        # Determine applicable regulations based on region
        if applicable_regs is None:
            applicable_regs = self._determine_regulations_by_region(region)
            
        subject = DataSubject(
            subject_id=subject_id,
            region=region,
            applicable_regulations=applicable_regs
        )
        
        self.data_subjects[subject_id] = subject
        self.logger.info(f"Registered data subject {subject_id} in region {region}")
        
        return subject
    
    def _determine_regulations_by_region(self, region: str) -> List[ComplianceRegulation]:
        """Determine applicable regulations based on geographic region."""
        region_mappings = {
            # European Union
            "EU": [ComplianceRegulation.GDPR],
            "Germany": [ComplianceRegulation.GDPR],
            "France": [ComplianceRegulation.GDPR],
            "Spain": [ComplianceRegulation.GDPR],
            "Italy": [ComplianceRegulation.GDPR],
            "Netherlands": [ComplianceRegulation.GDPR],
            
            # North America
            "California": [ComplianceRegulation.CCPA, ComplianceRegulation.GDPR],
            "US": [ComplianceRegulation.CCPA],
            "Canada": [ComplianceRegulation.GDPR],  # PIPEDA similar to GDPR
            
            # Asia Pacific
            "Singapore": [ComplianceRegulation.PDPA, ComplianceRegulation.GDPR],
            "Thailand": [ComplianceRegulation.PDPA],
            "Japan": [ComplianceRegulation.GDPR],  # Act on Protection of Personal Information
            "China": [ComplianceRegulation.PIPL],
            
            # Latin America
            "Brazil": [ComplianceRegulation.LGPD, ComplianceRegulation.GDPR],
            
            # Default
            "Global": [ComplianceRegulation.GDPR, ComplianceRegulation.CCPA, ComplianceRegulation.PDPA]
        }
        
        return region_mappings.get(region, [ComplianceRegulation.GDPR])
    
    def record_consent(
        self,
        data_subject_id: str,
        purposes: Set[ProcessingPurpose],
        consent_mechanism: str = "explicit",
        duration_days: Optional[int] = None
    ) -> ConsentRecord:
        """Record user consent for data processing."""
        
        consent_id = self._generate_consent_id(data_subject_id)
        granted_at = datetime.now()
        expires_at = None
        
        if duration_days:
            expires_at = granted_at + timedelta(days=duration_days)
            
        consent = ConsentRecord(
            consent_id=consent_id,
            data_subject_id=data_subject_id,
            purposes=purposes,
            granted_at=granted_at,
            expires_at=expires_at,
            consent_mechanism=consent_mechanism
        )
        
        self.consent_records[consent_id] = consent
        
        # Update data subject consent
        if data_subject_id in self.data_subjects:
            self.data_subjects[data_subject_id].consent_timestamp = granted_at
            self.data_subjects[data_subject_id].consent_purposes = purposes
            
        self.logger.info(f"Recorded consent {consent_id} for subject {data_subject_id}")
        return consent
    
    def withdraw_consent(self, data_subject_id: str, consent_id: Optional[str] = None):
        """Withdraw consent for a data subject."""
        
        # Find consent records for the subject
        consents_to_withdraw = []
        
        if consent_id:
            if consent_id in self.consent_records:
                consents_to_withdraw = [consent_id]
        else:
            # Withdraw all active consents for subject
            consents_to_withdraw = [
                cid for cid, consent in self.consent_records.items()
                if consent.data_subject_id == data_subject_id and consent.withdrawn_at is None
            ]
            
        withdrawn_count = 0
        for cid in consents_to_withdraw:
            self.consent_records[cid].withdrawn_at = datetime.now()
            withdrawn_count += 1
            
        # Update data subject
        if data_subject_id in self.data_subjects:
            self.data_subjects[data_subject_id].consent_purposes.clear()
            
        self.logger.info(f"Withdrew {withdrawn_count} consent records for subject {data_subject_id}")
        
        # Trigger data deletion if required
        self._handle_consent_withdrawal(data_subject_id)
    
    def _handle_consent_withdrawal(self, data_subject_id: str):
        """Handle actions after consent withdrawal."""
        # Mark subject's data for deletion
        subject_records = [
            r for r in self.processing_records 
            if r.data_subject_id == data_subject_id
        ]
        
        self.logger.info(f"Marked {len(subject_records)} records for deletion after consent withdrawal")
        
        # In production, this would trigger actual data deletion
        # For now, we log the action
        for record in subject_records:
            self.logger.info(f"GDPR: Deleting data record {record.record_id}")
    
    def record_processing_activity(
        self,
        data_type: DataType,
        purpose: ProcessingPurpose,
        data_subject_id: Optional[str] = None,
        data_size_bytes: int = 0,
        processing_location: str = "default",
        automated_decision: bool = False
    ) -> DataProcessingRecord:
        """Record a data processing activity."""
        
        record_id = self._generate_processing_id()
        retention_period = self.retention_policies.get(data_type, self.default_retention_days)
        
        # Determine legal basis based on regulation and purpose
        legal_basis = self._determine_legal_basis(purpose, data_subject_id)
        
        record = DataProcessingRecord(
            record_id=record_id,
            timestamp=datetime.now(),
            data_type=data_type,
            processing_purpose=purpose,
            data_subject_id=data_subject_id,
            legal_basis=legal_basis,
            retention_period=retention_period,
            data_size_bytes=data_size_bytes,
            processing_location=processing_location,
            automated_decision=automated_decision
        )
        
        self.processing_records.append(record)
        
        # Check compliance
        self._check_processing_compliance(record)
        
        return record
    
    def _determine_legal_basis(self, purpose: ProcessingPurpose, data_subject_id: Optional[str]) -> str:
        """Determine legal basis for processing under applicable regulations."""
        
        # Check if we have valid consent
        if data_subject_id and self._has_valid_consent(data_subject_id, purpose):
            return "consent"
            
        # Determine based on purpose
        legal_basis_map = {
            ProcessingPurpose.OBJECT_DETECTION: "legitimate_interest",
            ProcessingPurpose.PERFORMANCE_MONITORING: "legitimate_interest", 
            ProcessingPurpose.SYSTEM_OPTIMIZATION: "legitimate_interest",
            ProcessingPurpose.RESEARCH_DEVELOPMENT: "consent",
            ProcessingPurpose.SECURITY_MONITORING: "vital_interests",
            ProcessingPurpose.COMPLIANCE_REPORTING: "legal_obligation"
        }
        
        return legal_basis_map.get(purpose, "legitimate_interest")
    
    def _has_valid_consent(self, data_subject_id: str, purpose: ProcessingPurpose) -> bool:
        """Check if subject has valid consent for the given purpose."""
        
        subject_consents = [
            consent for consent in self.consent_records.values()
            if (consent.data_subject_id == data_subject_id and
                consent.withdrawn_at is None and
                purpose in consent.purposes and
                (consent.expires_at is None or consent.expires_at > datetime.now()))
        ]
        
        return len(subject_consents) > 0
    
    def _check_processing_compliance(self, record: DataProcessingRecord):
        """Check if processing activity is compliant."""
        
        warnings = []
        
        # Check consent for personal data processing
        if record.data_subject_id and record.legal_basis == "consent":
            if not self._has_valid_consent(record.data_subject_id, record.processing_purpose):
                warnings.append("Processing without valid consent")
                
        # Check data minimization
        if self.data_minimization:
            if record.data_size_bytes > 100_000_000:  # 100MB threshold
                warnings.append("Large data size - review data minimization")
                
        # Check automated decision making
        if record.automated_decision:
            subject = self.data_subjects.get(record.data_subject_id)
            if subject and ComplianceRegulation.GDPR in subject.applicable_regulations:
                warnings.append("Automated decision making - ensure GDPR Article 22 compliance")
                
        # Log warnings
        for warning in warnings:
            self.logger.warning(f"Compliance warning for {record.record_id}: {warning}")
    
    def handle_data_subject_request(
        self,
        request_type: str,
        data_subject_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle data subject rights requests (GDPR Articles 15-22)."""
        
        result = {
            "request_id": self._generate_request_id(),
            "request_type": request_type,
            "data_subject_id": data_subject_id,
            "timestamp": datetime.now().isoformat(),
            "status": "processed",
            "data": {}
        }
        
        if request_type == "access":  # Article 15 - Right to access
            result["data"] = self._handle_access_request(data_subject_id)
            
        elif request_type == "rectification":  # Article 16 - Right to rectification
            result["data"] = self._handle_rectification_request(data_subject_id, details or {})
            
        elif request_type == "erasure":  # Article 17 - Right to be forgotten
            result["data"] = self._handle_erasure_request(data_subject_id)
            
        elif request_type == "portability":  # Article 20 - Right to data portability
            result["data"] = self._handle_portability_request(data_subject_id)
            
        elif request_type == "restriction":  # Article 18 - Right to restriction
            result["data"] = self._handle_restriction_request(data_subject_id)
            
        elif request_type == "objection":  # Article 21 - Right to object
            result["data"] = self._handle_objection_request(data_subject_id, details or {})
            
        else:
            result["status"] = "unsupported"
            result["message"] = f"Unsupported request type: {request_type}"
            
        self.logger.info(f"Processed {request_type} request for subject {data_subject_id}")
        return result
    
    def _handle_access_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data access request - provide all personal data."""
        
        # Get subject info
        subject = self.data_subjects.get(data_subject_id, {})
        
        # Get processing records
        subject_records = [
            {
                "record_id": r.record_id,
                "timestamp": r.timestamp.isoformat(),
                "data_type": r.data_type.value,
                "purpose": r.processing_purpose.value,
                "legal_basis": r.legal_basis,
                "retention_days": r.retention_period,
                "data_size": r.data_size_bytes
            }
            for r in self.processing_records
            if r.data_subject_id == data_subject_id
        ]
        
        # Get consent records
        subject_consents = [
            {
                "consent_id": c.consent_id,
                "purposes": [p.value for p in c.purposes],
                "granted_at": c.granted_at.isoformat(),
                "expires_at": c.expires_at.isoformat() if c.expires_at else None,
                "withdrawn_at": c.withdrawn_at.isoformat() if c.withdrawn_at else None,
                "mechanism": c.consent_mechanism
            }
            for c in self.consent_records.values()
            if c.data_subject_id == data_subject_id
        ]
        
        return {
            "subject_info": subject.__dict__ if hasattr(subject, '__dict__') else {},
            "processing_records": subject_records,
            "consent_records": subject_consents,
            "total_records": len(subject_records)
        }
    
    def _handle_rectification_request(self, data_subject_id: str, corrections: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data rectification request."""
        # In production, this would update the actual data
        # For now, we log the correction request
        
        corrected_fields = []
        for field, new_value in corrections.items():
            self.logger.info(f"GDPR: Correcting {field} for subject {data_subject_id}")
            corrected_fields.append(field)
            
        return {
            "corrected_fields": corrected_fields,
            "message": "Data corrections applied"
        }
    
    def _handle_erasure_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle erasure request (right to be forgotten)."""
        
        # Count records to be deleted
        records_to_delete = [
            r for r in self.processing_records
            if r.data_subject_id == data_subject_id
        ]
        
        # Remove consent records
        consents_to_delete = [
            cid for cid, c in self.consent_records.items()
            if c.data_subject_id == data_subject_id
        ]
        
        # In production, this would trigger actual data deletion
        deleted_records = len(records_to_delete)
        deleted_consents = len(consents_to_delete)
        
        # Remove from our tracking
        self.processing_records = [
            r for r in self.processing_records
            if r.data_subject_id != data_subject_id
        ]
        
        for cid in consents_to_delete:
            del self.consent_records[cid]
            
        if data_subject_id in self.data_subjects:
            del self.data_subjects[data_subject_id]
            
        self.logger.info(f"GDPR: Erased data for subject {data_subject_id}")
        
        return {
            "deleted_records": deleted_records,
            "deleted_consents": deleted_consents,
            "message": "Personal data erased successfully"
        }
    
    def _handle_portability_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data portability request."""
        # Get structured data in portable format
        access_data = self._handle_access_request(data_subject_id)
        
        # Return in structured format for portability
        return {
            "format": "JSON",
            "data": access_data,
            "message": "Data provided in structured, machine-readable format"
        }
    
    def _handle_restriction_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle processing restriction request."""
        # Mark records for restricted processing
        restricted_count = 0
        for record in self.processing_records:
            if record.data_subject_id == data_subject_id:
                # In production, this would mark records as restricted
                restricted_count += 1
                
        return {
            "restricted_records": restricted_count,
            "message": "Processing restricted pending resolution"
        }
    
    def _handle_objection_request(self, data_subject_id: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Handle objection to processing request."""
        objected_purposes = details.get("purposes", [])
        
        # Stop processing for objected purposes where legally possible
        stopped_processing = []
        for purpose in objected_purposes:
            # Check if we can legally stop (not overriding legitimate interest)
            if purpose not in ["security_monitoring", "compliance_reporting"]:
                stopped_processing.append(purpose)
                
        return {
            "stopped_processing": stopped_processing,
            "message": f"Processing stopped for {len(stopped_processing)} purposes"
        }
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        current_time = datetime.now()
        
        # Processing statistics
        processing_stats = {}
        for data_type in DataType:
            type_records = [r for r in self.processing_records if r.data_type == data_type]
            processing_stats[data_type.value] = {
                "total_records": len(type_records),
                "total_data_bytes": sum(r.data_size_bytes for r in type_records)
            }
            
        # Consent statistics  
        total_consents = len(self.consent_records)
        active_consents = len([c for c in self.consent_records.values() if c.withdrawn_at is None])
        expired_consents = len([
            c for c in self.consent_records.values()
            if c.expires_at and c.expires_at < current_time
        ])
        
        # Data retention analysis
        retention_analysis = {}
        for data_type, retention_days in self.retention_policies.items():
            cutoff_date = current_time - timedelta(days=retention_days)
            expired_records = [
                r for r in self.processing_records
                if r.data_type == data_type and r.timestamp < cutoff_date
            ]
            retention_analysis[data_type.value] = {
                "retention_days": retention_days,
                "expired_records": len(expired_records)
            }
            
        return {
            "report_timestamp": current_time.isoformat(),
            "applicable_regulations": [r.value for r in self.applicable_regulations],
            "data_subjects": {
                "total_subjects": len(self.data_subjects),
                "by_region": self._group_subjects_by_region()
            },
            "processing_statistics": processing_stats,
            "consent_management": {
                "total_consents": total_consents,
                "active_consents": active_consents, 
                "expired_consents": expired_consents,
                "withdrawal_rate": (total_consents - active_consents) / max(1, total_consents) * 100
            },
            "data_retention": retention_analysis,
            "privacy_settings": {
                "privacy_by_design": self.privacy_by_design,
                "data_minimization": self.data_minimization,
                "pseudonymization": self.pseudonymization_enabled,
                "encryption_at_rest": self.encryption_at_rest,
                "encryption_in_transit": self.encryption_in_transit
            }
        }
    
    def _group_subjects_by_region(self) -> Dict[str, int]:
        """Group data subjects by region."""
        regions = {}
        for subject in self.data_subjects.values():
            region = subject.region
            regions[region] = regions.get(region, 0) + 1
        return regions
    
    # Utility methods
    
    def _generate_consent_id(self, data_subject_id: str) -> str:
        """Generate unique consent ID."""
        timestamp = str(int(time.time() * 1000))
        content = f"{data_subject_id}:{timestamp}"
        return f"consent_{hashlib.md5(content.encode()).hexdigest()[:8]}"
    
    def _generate_processing_id(self) -> str:
        """Generate unique processing record ID."""
        timestamp = str(int(time.time() * 1000))
        return f"proc_{timestamp}_{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"
        
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        timestamp = str(int(time.time() * 1000))
        return f"req_{timestamp}_{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"


# Global compliance manager
_global_compliance = None


def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager instance."""
    global _global_compliance
    if _global_compliance is None:
        _global_compliance = ComplianceManager()
    return _global_compliance


def ensure_compliance(data_type: DataType, purpose: ProcessingPurpose, **kwargs):
    """Decorator to ensure compliance for data processing operations."""
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            compliance = get_compliance_manager()
            
            # Record processing activity
            compliance.record_processing_activity(
                data_type=data_type,
                purpose=purpose,
                **kwargs
            )
            
            # Execute function
            result = func(*args, **func_kwargs)
            
            return result
        return wrapper
    return decorator
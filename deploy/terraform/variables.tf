# Global Variables
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "spike-snn-event"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "spike-snn-cluster"
}

# Multi-Region Configuration
variable "primary_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-west-2"
}

variable "secondary_regions" {
  description = "Secondary regions for multi-region deployment"
  type        = list(string)
  default     = ["us-east-1", "eu-west-1", "ap-southeast-1"]
}

variable "enable_multi_region" {
  description = "Enable multi-region deployment"
  type        = bool
  default     = false
}

# Networking
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones_count" {
  description = "Number of availability zones to use"
  type        = number
  default     = 3
}

# Node Groups
variable "gpu_node_instance_types" {
  description = "Instance types for GPU worker nodes"
  type        = list(string)
  default     = ["g4dn.xlarge", "g4dn.2xlarge", "g4dn.4xlarge"]
}

variable "cpu_node_instance_types" {
  description = "Instance types for CPU worker nodes"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge", "m5.2xlarge"]
}

variable "gpu_nodes_min_size" {
  description = "Minimum number of GPU worker nodes"
  type        = number
  default     = 2
}

variable "gpu_nodes_max_size" {
  description = "Maximum number of GPU worker nodes"
  type        = number
  default     = 50
}

variable "gpu_nodes_desired_size" {
  description = "Desired number of GPU worker nodes"
  type        = number
  default     = 3
}

variable "cpu_nodes_min_size" {
  description = "Minimum number of CPU worker nodes"
  type        = number
  default     = 1
}

variable "cpu_nodes_max_size" {
  description = "Maximum number of CPU worker nodes"
  type        = number
  default     = 20
}

variable "cpu_nodes_desired_size" {
  description = "Desired number of CPU worker nodes"
  type        = number
  default     = 2
}

# Auto-scaling
variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_keda" {
  description = "Enable KEDA for event-driven autoscaling"
  type        = bool
  default     = true
}

variable "enable_vpa" {
  description = "Enable Vertical Pod Autoscaler"
  type        = bool
  default     = true
}

# Storage
variable "enable_efs" {
  description = "Enable EFS for shared storage"
  type        = bool
  default     = true
}

variable "model_storage_class" {
  description = "Storage class for model artifacts"
  type        = string
  default     = "STANDARD_IA"
}

# Monitoring
variable "enable_monitoring" {
  description = "Enable monitoring stack"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable centralized logging"
  type        = bool
  default     = true
}

variable "enable_tracing" {
  description = "Enable distributed tracing"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch logs retention period"
  type        = number
  default     = 30
}

# Security
variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit"
  type        = bool
  default     = true
}

variable "enable_pod_security_standards" {
  description = "Enable Pod Security Standards"
  type        = bool
  default     = true
}

variable "enable_network_policies" {
  description = "Enable network policies"
  type        = bool
  default     = true
}

# Compliance
variable "enable_gdpr_compliance" {
  description = "Enable GDPR compliance features"
  type        = bool
  default     = true
}

variable "enable_hipaa_compliance" {
  description = "Enable HIPAA compliance features"
  type        = bool
  default     = false
}

variable "data_residency_regions" {
  description = "Regions where data must reside for compliance"
  type        = list(string)
  default     = []
}

# Backup and DR
variable "enable_backup" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = true
}

# CDN
variable "enable_cloudfront" {
  description = "Enable CloudFront CDN"
  type        = bool
  default     = true
}

variable "cloudfront_price_class" {
  description = "CloudFront price class"
  type        = string
  default     = "PriceClass_All"
}

# Domain and DNS
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

# Tags
variable "common_tags" {
  description = "Common tags to be applied to all resources"
  type        = map(string)
  default = {
    Project     = "spike-snn-event"
    ManagedBy   = "terraform"
    Application = "neuromorphic-vision"
  }
}

variable "cost_center" {
  description = "Cost center for resource tagging"
  type        = string
  default     = "ai-research"
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
  default     = "platform-team"
}

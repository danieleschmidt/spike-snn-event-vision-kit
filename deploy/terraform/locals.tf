locals {
  # Cluster configuration
  cluster_name    = "${var.project_name}-${var.environment}-cluster"
  cluster_version = "1.28"
  
  # Networking
  azs = slice(data.aws_availability_zones.primary.names, 0, var.availability_zones_count)
  
  private_subnets = [
    for i in range(var.availability_zones_count) :
    cidrsubnet(var.vpc_cidr, 8, i + 1)
  ]
  
  public_subnets = [
    for i in range(var.availability_zones_count) :
    cidrsubnet(var.vpc_cidr, 8, i + 101)
  ]
  
  # Common tags
  common_tags = merge(var.common_tags, {
    Environment   = var.environment
    ClusterName   = local.cluster_name
    Region        = var.primary_region
    CostCenter    = var.cost_center
    Owner         = var.owner
    ManagedBy     = "terraform"
    LastUpdated   = timestamp()
  })
  
  # Kubernetes labels
  cluster_labels = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "environment"                                 = var.environment
    "managed-by"                                  = "terraform"
  }
  
  # Resource naming
  name_prefix = "${var.project_name}-${var.environment}"
  
  # Storage configuration
  model_bucket_name = "${local.name_prefix}-models-${random_id.bucket_suffix.hex}"
  backup_bucket_name = "${local.name_prefix}-backups-${random_id.bucket_suffix.hex}"
  
  # Monitoring configuration
  monitoring_namespace = "monitoring"
  logging_namespace    = "logging"
  
  # Security configuration
  pod_security_standards = var.enable_pod_security_standards ? {
    enforce = "restricted"
    audit   = "restricted"
    warn    = "restricted"
  } : {}
  
  # Multi-region configuration
  regions = var.enable_multi_region ? concat([var.primary_region], var.secondary_regions) : [var.primary_region]
  
  # Compliance settings
  gdpr_regions = ["eu-west-1", "eu-central-1", "eu-north-1"]
  
  # Auto-scaling configuration
  cluster_autoscaler_settings = {
    scale_down_delay_after_add       = "10m"
    scale_down_unneeded_time         = "10m"
    scale_down_utilization_threshold = 0.5
    skip_nodes_with_local_storage    = false
    skip_nodes_with_system_pods      = false
  }
  
  # GPU configuration
  gpu_instance_families = {
    "g4dn" = {
      "accelerator" = "nvidia-tesla-t4"
      "memory"      = "gpu-memory-optimized"
    }
    "p3"   = {
      "accelerator" = "nvidia-tesla-v100"
      "memory"      = "gpu-compute-optimized"
    }
    "p4d"  = {
      "accelerator" = "nvidia-tesla-a100"
      "memory"      = "gpu-hpc-optimized"
    }
  }
  
  # Workload-specific configurations
  neuromorphic_workload_config = {
    cpu_request    = "500m"
    memory_request = "1Gi"
    cpu_limit      = "2000m"
    memory_limit   = "4Gi"
    gpu_limit      = "1"
  }
  
  # Service mesh configuration
  istio_config = {
    version           = "1.19.0"
    enable_mtls       = true
    enable_telemetry  = true
    enable_tracing    = var.enable_tracing
  }
  
  # Backup configuration
  backup_schedule = {
    daily   = "0 2 * * *"    # 2 AM daily
    weekly  = "0 3 * * 0"    # 3 AM Sunday
    monthly = "0 4 1 * *"    # 4 AM 1st of month
  }
}

# Generate random suffix for globally unique names
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Generate random passwords
resource "random_password" "grafana_admin" {
  length  = 16
  special = true
}

resource "random_password" "database_password" {
  length  = 16
  special = false  # Some databases don't like special chars
}

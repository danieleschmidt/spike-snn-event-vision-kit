# =====================================================================================
# OUTPUTS - INFRASTRUCTURE RESOURCE REFERENCES
# =====================================================================================

# Cluster Information
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
  sensitive   = false
}

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_version" {
  description = "Version of the EKS cluster"
  value       = module.eks.cluster_version
}

output "cluster_platform_version" {
  description = "Platform version for the EKS cluster"
  value       = module.eks.cluster_platform_version
}

output "cluster_status" {
  description = "Status of the EKS cluster. One of `CREATING`, `ACTIVE`, `DELETING`, `FAILED`"
  value       = module.eks.cluster_status
}

output "cluster_primary_security_group_id" {
  description = "Cluster security group that was created by Amazon EKS for the cluster"
  value       = module.eks.cluster_primary_security_group_id
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "cluster_arn" {
  description = "The Amazon Resource Name (ARN) of the cluster"
  value       = module.eks.cluster_arn
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = module.eks.cluster_iam_role_name
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = module.eks.cluster_iam_role_arn
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster for the OpenID Connect identity provider"
  value       = module.eks.cluster_oidc_issuer_url
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider if enabled"
  value       = module.eks.oidc_provider_arn
}

# Node Groups
output "eks_managed_node_groups" {
  description = "Map of attribute maps for all EKS managed node groups"
  value       = module.eks.eks_managed_node_groups
  sensitive   = true
}

output "eks_managed_node_groups_autoscaling_group_names" {
  description = "List of the autoscaling group names"
  value       = module.eks.eks_managed_node_groups_autoscaling_group_names
}

# Fargate
output "fargate_profiles" {
  description = "Map of attribute maps for all EKS Fargate profiles"
  value       = module.eks.fargate_profiles
  sensitive   = true
}

# IRSA
output "cluster_autoscaler_iam_role_arn" {
  description = "IAM role ARN for cluster autoscaler"
  value       = module.cluster_autoscaler_irsa_role.iam_role_arn
}

output "load_balancer_controller_iam_role_arn" {
  description = "IAM role ARN for AWS Load Balancer Controller"
  value       = module.load_balancer_controller_irsa_role.iam_role_arn
}

output "ebs_csi_iam_role_arn" {
  description = "IAM role ARN for EBS CSI driver"
  value       = module.ebs_csi_irsa_role.iam_role_arn
}

# VPC
output "vpc_id" {
  description = "ID of the VPC where the cluster and its nodes are deployed"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "The CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "availability_zones" {
  description = "List of availability zones used"
  value       = local.azs
}

# Storage
output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.spike_snn_event.repository_url
}

output "ecr_repository_arn" {
  description = "ARN of the ECR repository"
  value       = aws_ecr_repository.spike_snn_event.arn
}

output "model_storage_bucket_id" {
  description = "ID of the S3 bucket for model storage"
  value       = aws_s3_bucket.model_storage.id
}

output "model_storage_bucket_arn" {
  description = "ARN of the S3 bucket for model storage"
  value       = aws_s3_bucket.model_storage.arn
}

output "model_storage_bucket_domain_name" {
  description = "Domain name of the S3 bucket for model storage"
  value       = aws_s3_bucket.model_storage.bucket_domain_name
}

output "backup_storage_bucket_id" {
  description = "ID of the S3 bucket for backup storage"
  value       = aws_s3_bucket.backups.id
}

output "backup_storage_bucket_arn" {
  description = "ARN of the S3 bucket for backup storage"
  value       = aws_s3_bucket.backups.arn
}

output "efs_id" {
  description = "ID of the EFS file system"
  value       = var.enable_efs ? aws_efs_file_system.shared_storage[0].id : null
}

output "efs_arn" {
  description = "ARN of the EFS file system"
  value       = var.enable_efs ? aws_efs_file_system.shared_storage[0].arn : null
}

output "efs_dns_name" {
  description = "DNS name of the EFS file system"
  value       = var.enable_efs ? aws_efs_file_system.shared_storage[0].dns_name : null
}

# Security
output "kms_key_arns" {
  description = "ARNs of KMS keys created for encryption"
  value = {
    eks = var.enable_encryption_at_rest ? aws_kms_key.eks[0].arn : null
    ebs = var.enable_encryption_at_rest ? aws_kms_key.ebs[0].arn : null
    s3  = var.enable_encryption_at_rest ? aws_kms_key.s3[0].arn : null
    ecr = var.enable_encryption_at_rest ? aws_kms_key.ecr[0].arn : null
    efs = var.enable_efs && var.enable_encryption_at_rest ? aws_kms_key.efs[0].arn : null
  }
  sensitive = true
}

# Configuration
output "region" {
  description = "AWS region"
  value       = var.primary_region
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "project_name" {
  description = "Project name"
  value       = var.project_name
}

# Kubectl Configuration
output "configure_kubectl" {
  description = "Configure kubectl: make sure you're logged in with the correct AWS profile and run the following command to update your kubeconfig"
  value       = "aws eks --region ${var.primary_region} update-kubeconfig --name ${module.eks.cluster_name}"
}

# Monitoring endpoints (will be populated after monitoring stack deployment)
output "monitoring_endpoints" {
  description = "Monitoring service endpoints"
  value = {
    prometheus_endpoint = "http://prometheus.monitoring.svc.cluster.local:9090"
    grafana_endpoint    = "http://grafana.monitoring.svc.cluster.local:3000"
    alertmanager_endpoint = "http://alertmanager.monitoring.svc.cluster.local:9093"
  }
}

# Helm deployment commands
output "helm_deployment_commands" {
  description = "Commands to deploy the neuromorphic vision application"
  value = {
    add_helm_repo = "helm repo add spike-snn-event ./deploy/helm/spike-snn-event"
    install_app   = "helm install neuromorphic-vision spike-snn-event/spike-snn-event --namespace neuromorphic-vision --create-namespace"
    upgrade_app   = "helm upgrade neuromorphic-vision spike-snn-event/spike-snn-event --namespace neuromorphic-vision"
  }
}

# Application URLs (for when ingress is configured)
output "application_urls" {
  description = "Application access URLs"
  value = var.domain_name != "" ? {
    api_endpoint     = "https://api.${var.domain_name}"
    web_interface    = "https://app.${var.domain_name}"
    grafana_dashboard = "https://grafana.${var.domain_name}"
    prometheus_ui    = "https://prometheus.${var.domain_name}"
  } : {
    note = "Configure domain_name variable to get application URLs"
  }
}

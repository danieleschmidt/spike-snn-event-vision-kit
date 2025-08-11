# =====================================================================================
# SPIKE SNN EVENT VISION KIT - PRODUCTION DEPLOYMENT INFRASTRUCTURE
# =====================================================================================
# Comprehensive multi-cloud, multi-region production deployment for neuromorphic
# vision processing with enterprise-grade security, monitoring, and compliance.
# =====================================================================================

# =====================================================================================
# SECURITY - KMS KEYS FOR ENCRYPTION
# =====================================================================================

resource "aws_kms_key" "eks" {
  count = var.enable_encryption_at_rest ? 1 : 0
  
  description             = "EKS Secret Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-eks-key"
  })
}

resource "aws_kms_alias" "eks" {
  count = var.enable_encryption_at_rest ? 1 : 0
  
  name          = "alias/${local.name_prefix}-eks"
  target_key_id = aws_kms_key.eks[0].key_id
}

resource "aws_kms_key" "ebs" {
  count = var.enable_encryption_at_rest ? 1 : 0
  
  description             = "EBS Volume Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-ebs-key"
  })
}

resource "aws_kms_alias" "ebs" {
  count = var.enable_encryption_at_rest ? 1 : 0
  
  name          = "alias/${local.name_prefix}-ebs"
  target_key_id = aws_kms_key.ebs[0].key_id
}

# =====================================================================================
# NETWORKING - VPC AND SUBNETS
# =====================================================================================

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.name_prefix}-vpc"
  cidr = var.vpc_cidr

  azs             = local.azs
  private_subnets = local.private_subnets
  public_subnets  = local.public_subnets

  # NAT Gateway configuration
  enable_nat_gateway     = true
  single_nat_gateway     = var.environment == "dev" ? true : false
  one_nat_gateway_per_az = var.environment == "prod" ? true : false
  
  # DNS configuration
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true
  flow_log_max_aggregation_interval    = 60
  
  # VPC Endpoints for security and cost optimization
  enable_s3_endpoint       = true
  enable_dynamodb_endpoint = true
  
  tags = merge(local.common_tags, local.cluster_labels)

  public_subnet_tags = merge(local.cluster_labels, {
    "kubernetes.io/role/elb" = "1"
    "subnet-type"            = "public"
  })

  private_subnet_tags = merge(local.cluster_labels, {
    "kubernetes.io/role/internal-elb" = "1"
    "subnet-type"                     = "private"
  })

  # Additional VPC endpoints for private connectivity
  vpc_endpoint_tags = local.common_tags
}

# =====================================================================================
# EKS CLUSTER - KUBERNETES CONTROL PLANE
# =====================================================================================

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = local.cluster_name
  cluster_version = local.cluster_version

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  control_plane_subnet_ids       = module.vpc.private_subnets

  # Cluster endpoint configuration
  cluster_endpoint_public_access  = var.environment == "prod" ? false : true
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access_cidrs = var.environment == "prod" ? [] : ["0.0.0.0/0"]

  # Security and encryption
  cluster_encryption_config = var.enable_encryption_at_rest ? [
    {
      provider_key_arn = aws_kms_key.eks[0].arn
      resources        = ["secrets"]
    }
  ] : []
  
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  cloudwatch_log_group_retention_in_days = var.log_retention_days
  
  # IRSA (IAM Roles for Service Accounts)
  enable_irsa = true
  
  # Cluster add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
      configuration_values = jsonencode({
        computeType = "Fargate"
        resources = {
          limits = {
            cpu    = "100m"
            memory = "128Mi"
          }
          requests = {
            cpu    = "100m"
            memory = "128Mi"
          }
        }
      })
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent    = true
      before_compute = true
      configuration_values = jsonencode({
        env = {
          ENABLE_PREFIX_DELEGATION = "true"
          ENABLE_POD_ENI           = "true"
          POD_SECURITY_GROUP_ENFORCING_MODE = "standard"
        }
      })
    }
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa_role.iam_role_arn
    }
  }

  # Enhanced cluster security group rules
  cluster_security_group_additional_rules = {
    ingress_nodes_ephemeral_ports_tcp = {
      description                = "Node groups to cluster API"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "ingress"
      source_node_security_group = true
    }
    ingress_nodes_https = {
      description                = "Nodes to cluster HTTPS"
      protocol                   = "tcp"
      from_port                  = 443
      to_port                    = 443
      type                       = "ingress"
      source_node_security_group = true
    }
  }

  # Enhanced node security group rules
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
    ingress_cluster_all = {
      description                   = "Cluster to node all ports/protocols"
      protocol                      = "-1"
      from_port                     = 0
      to_port                       = 0
      type                          = "ingress"
      source_cluster_security_group = true
    }
    # Allow ingress from ALB
    ingress_alb_http = {
      description = "ALB to nodes HTTP"
      protocol    = "tcp"
      from_port   = 80
      to_port     = 80
      type        = "ingress"
      cidr_blocks = [var.vpc_cidr]
    }
    ingress_alb_https = {
      description = "ALB to nodes HTTPS"
      protocol    = "tcp"
      from_port   = 443
      to_port     = 443
      type        = "ingress"
      cidr_blocks = [var.vpc_cidr]
    }
    # Prometheus and Grafana
    ingress_monitoring = {
      description = "Monitoring access"
      protocol    = "tcp"
      from_port   = 9090
      to_port     = 9100
      type        = "ingress"
      cidr_blocks = [var.vpc_cidr]
    }
  }

  # Enhanced EKS Managed Node Groups
  eks_managed_node_groups = {
    # GPU-optimized nodes for neuromorphic processing
    gpu_nodes = {
      name = "${local.name_prefix}-gpu-nodes"

      instance_types = var.gpu_node_instance_types
      capacity_type  = var.environment == "prod" ? "ON_DEMAND" : "SPOT"

      min_size     = var.gpu_nodes_min_size
      max_size     = var.gpu_nodes_max_size
      desired_size = var.gpu_nodes_desired_size

      ami_type                   = "AL2_x86_64_GPU"
      use_custom_launch_template = true

      # Enhanced launch template configuration
      launch_template_name    = "${local.cluster_name}-gpu-lt"
      launch_template_version = "$Latest"
      
      # Block device mappings for GPU workloads
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 100
            volume_type           = "gp3"
            iops                  = 3000
            throughput            = 150
            encrypted             = var.enable_encryption_at_rest
            kms_key_id           = var.enable_encryption_at_rest ? aws_kms_key.ebs[0].arn : null
            delete_on_termination = true
          }
        }
      }

      # Network configuration
      subnet_ids = module.vpc.private_subnets
      
      # Update strategy
      update_config = {
        max_unavailable_percentage = 25
      }
      
      # Enhanced labels and taints
      labels = {
        role                    = "gpu-worker"
        "node-type"            = "gpu"
        "workload-type"        = "neuromorphic"
        "nvidia.com/gpu.present" = "true"
      }

      taints = {
        gpu = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
        neuromorphic = {
          key    = "workload.neuromorphic/dedicated"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }

      # Enhanced metadata options
      metadata_options = {
        http_endpoint = "enabled"
        http_tokens   = "required"
        http_put_response_hop_limit = 2
        instance_metadata_tags      = "enabled"
      }

      tags = merge(local.common_tags, {
        NodeGroup = "gpu-nodes"
        Workload  = "neuromorphic-vision"
      })
    }

    # CPU nodes for general workloads
    cpu_nodes = {
      name = "${local.name_prefix}-cpu-nodes"

      instance_types = var.cpu_node_instance_types
      capacity_type  = var.environment == "prod" ? "ON_DEMAND" : "SPOT"

      min_size     = var.cpu_nodes_min_size
      max_size     = var.cpu_nodes_max_size
      desired_size = var.cpu_nodes_desired_size

      ami_type = "AL2_x86_64"
      use_custom_launch_template = true

      # Block device mappings
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 50
            volume_type           = "gp3"
            iops                  = 3000
            throughput            = 125
            encrypted             = var.enable_encryption_at_rest
            kms_key_id           = var.enable_encryption_at_rest ? aws_kms_key.ebs[0].arn : null
            delete_on_termination = true
          }
        }
      }

      # Network configuration
      subnet_ids = module.vpc.private_subnets
      
      # Update strategy
      update_config = {
        max_unavailable_percentage = 33
      }
      
      labels = {
        role         = "cpu-worker"
        "node-type" = "cpu"
        "workload-type" = "general"
      }

      # Enhanced metadata options
      metadata_options = {
        http_endpoint = "enabled"
        http_tokens   = "required"
        http_put_response_hop_limit = 2
        instance_metadata_tags      = "enabled"
      }

      tags = merge(local.common_tags, {
        NodeGroup = "cpu-nodes"
        Workload  = "general"
      })
    }

    # Monitoring and system nodes
    system_nodes = {
      name = "${local.name_prefix}-system-nodes"

      instance_types = ["m5.large"]
      capacity_type  = "ON_DEMAND"

      min_size     = 1
      max_size     = 3
      desired_size = 1

      ami_type = "AL2_x86_64"
      
      labels = {
        role         = "system-worker"
        "node-type" = "system"
        "workload-type" = "monitoring"
      }

      taints = {
        system = {
          key    = "node-role.kubernetes.io/system"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }

      tags = merge(local.common_tags, {
        NodeGroup = "system-nodes"
        Workload  = "monitoring"
      })
    }
  }

  # Fargate profiles for serverless workloads
  fargate_profiles = {
    karpenter = {
      name = "karpenter"
      selectors = [
        {
          namespace = "karpenter"
        }
      ]
      
      subnet_ids = module.vpc.private_subnets
      
      tags = merge(local.common_tags, {
        Profile = "karpenter"
      })
    }
    
    monitoring = {
      name = "monitoring"
      selectors = [
        {
          namespace = "monitoring"
        }
      ]
      
      subnet_ids = module.vpc.private_subnets
      
      tags = merge(local.common_tags, {
        Profile = "monitoring"
      })
    }
  }

  # Enhanced cluster access management
  access_entries = {
    cluster_admin = {
      kubernetes_groups = ["system:masters"]
      principal_arn     = data.aws_caller_identity.current.arn

      policy_associations = {
        admin = {
          policy_arn = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
          access_scope = {
            type = "cluster"
          }
        }
      }
    }
  }

  # Comprehensive cluster tags
  tags = local.common_tags
}

# =====================================================================================
# IAM ROLES FOR SERVICE ACCOUNTS (IRSA)
# =====================================================================================

# EBS CSI Driver IRSA
module "ebs_csi_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${local.name_prefix}-ebs-csi"
  attach_ebs_csi_policy = true

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }

  tags = local.common_tags
}

# AWS Load Balancer Controller IRSA
module "load_balancer_controller_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name                              = "${local.name_prefix}-load-balancer-controller"
  attach_load_balancer_controller_policy = true

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }

  tags = local.common_tags
}

# Cluster Autoscaler IRSA
module "cluster_autoscaler_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name                        = "${local.name_prefix}-cluster-autoscaler"
  attach_cluster_autoscaler_policy = true
  cluster_autoscaler_cluster_ids   = [module.eks.cluster_name]

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:cluster-autoscaler"]
    }
  }

  tags = local.common_tags
}

# =====================================================================================
# STORAGE - S3 BUCKETS AND EFS
# =====================================================================================

# S3 bucket for model storage
resource "aws_s3_bucket" "model_storage" {
  bucket = local.model_bucket_name
  
  tags = merge(local.common_tags, {
    Name        = "Model Storage"
    StorageType = "Models"
  })
}

resource "aws_s3_bucket_versioning" "model_storage" {
  bucket = aws_s3_bucket.model_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_storage" {
  bucket = aws_s3_bucket.model_storage.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.enable_encryption_at_rest ? "aws:kms" : "AES256"
      kms_master_key_id = var.enable_encryption_at_rest ? aws_kms_key.s3[0].arn : null
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "model_storage" {
  bucket = aws_s3_bucket.model_storage.id

  rule {
    id     = "transition_to_ia"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}

# S3 bucket for backups
resource "aws_s3_bucket" "backups" {
  bucket = local.backup_bucket_name
  
  tags = merge(local.common_tags, {
    Name        = "Backup Storage"
    StorageType = "Backups"
  })
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.enable_encryption_at_rest ? "aws:kms" : "AES256"
      kms_master_key_id = var.enable_encryption_at_rest ? aws_kms_key.s3[0].arn : null
    }
  }
}

# KMS key for S3 encryption
resource "aws_kms_key" "s3" {
  count = var.enable_encryption_at_rest ? 1 : 0
  
  description             = "S3 Bucket Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-s3-key"
  })
}

resource "aws_kms_alias" "s3" {
  count = var.enable_encryption_at_rest ? 1 : 0
  
  name          = "alias/${local.name_prefix}-s3"
  target_key_id = aws_kms_key.s3[0].key_id
}

# EFS for shared storage
resource "aws_efs_file_system" "shared_storage" {
  count = var.enable_efs ? 1 : 0
  
  creation_token   = "${local.name_prefix}-efs"
  performance_mode = "generalPurpose"
  throughput_mode  = "provisioned"
  provisioned_throughput_in_mibps = 1024
  
  encrypted  = var.enable_encryption_at_rest
  kms_key_id = var.enable_encryption_at_rest ? aws_kms_key.efs[0].arn : null

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-efs"
  })
}

resource "aws_kms_key" "efs" {
  count = var.enable_efs && var.enable_encryption_at_rest ? 1 : 0
  
  description             = "EFS Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-efs-key"
  })
}

resource "aws_efs_mount_target" "shared_storage" {
  count = var.enable_efs ? length(module.vpc.private_subnets) : 0
  
  file_system_id  = aws_efs_file_system.shared_storage[0].id
  subnet_id       = module.vpc.private_subnets[count.index]
  security_groups = [aws_security_group.efs[0].id]
}

resource "aws_security_group" "efs" {
  count = var.enable_efs ? 1 : 0
  
  name_prefix = "${local.name_prefix}-efs-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port = 2049
    to_port   = 2049
    protocol  = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-efs-sg"
  })
}

# =====================================================================================
# CONTAINER REGISTRY - ECR
# =====================================================================================

resource "aws_ecr_repository" "spike_snn_event" {
  name                 = "${local.name_prefix}/neuromorphic-vision"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = var.enable_encryption_at_rest ? "KMS" : "AES256"
    kms_key        = var.enable_encryption_at_rest ? aws_kms_key.ecr[0].arn : null
  }

  tags = local.common_tags
}

resource "aws_kms_key" "ecr" {
  count = var.enable_encryption_at_rest ? 1 : 0
  
  description             = "ECR Repository Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-ecr-key"
  })
}

resource "aws_ecr_lifecycle_policy" "spike_snn_event" {
  repository = aws_ecr_repository.spike_snn_event.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 50 production images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v", "prod"]
          countType     = "imageCountMoreThan"
          countNumber   = 50
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Keep last 10 development images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["dev", "staging"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 3
        description  = "Delete untagged images older than 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# ECR Repository policy for cross-account access
resource "aws_ecr_repository_policy" "spike_snn_event" {
  repository = aws_ecr_repository.spike_snn_event.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowPull"
        Effect = "Allow"
        Principal = {
          AWS = [
            "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
          ]
        }
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
      }
    ]
  })
}
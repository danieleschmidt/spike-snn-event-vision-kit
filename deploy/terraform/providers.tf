# Primary AWS Provider
provider "aws" {
  alias  = "primary"
  region = var.primary_region
  
  default_tags {
    tags = merge(var.common_tags, {
      Environment = var.environment
      Region      = var.primary_region
      CostCenter  = var.cost_center
      Owner       = var.owner
    })
  }
}

# Secondary AWS Providers for Multi-Region
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
  count  = var.enable_multi_region && contains(var.secondary_regions, "us-east-1") ? 1 : 0
  
  default_tags {
    tags = merge(var.common_tags, {
      Environment = var.environment
      Region      = "us-east-1"
      CostCenter  = var.cost_center
      Owner       = var.owner
    })
  }
}

provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"
  count  = var.enable_multi_region && contains(var.secondary_regions, "eu-west-1") ? 1 : 0
  
  default_tags {
    tags = merge(var.common_tags, {
      Environment = var.environment
      Region      = "eu-west-1"
      CostCenter  = var.cost_center
      Owner       = var.owner
    })
  }
}

provider "aws" {
  alias  = "ap_southeast_1"
  region = "ap-southeast-1"
  count  = var.enable_multi_region && contains(var.secondary_regions, "ap-southeast-1") ? 1 : 0
  
  default_tags {
    tags = merge(var.common_tags, {
      Environment = var.environment
      Region      = "ap-southeast-1"
      CostCenter  = var.cost_center
      Owner       = var.owner
    })
  }
}

# Kubernetes Provider
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args = [
      "eks", "get-token",
      "--cluster-name", module.eks.cluster_name,
      "--region", var.primary_region
    ]
  }
}

# Helm Provider
provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args = [
        "eks", "get-token",
        "--cluster-name", module.eks.cluster_name,
        "--region", var.primary_region
      ]
    }
  }
}

# Kubectl Provider
provider "kubectl" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args = [
      "eks", "get-token",
      "--cluster-name", module.eks.cluster_name,
      "--region", var.primary_region
    ]
  }
}

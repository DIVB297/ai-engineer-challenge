# AWS Deployment Migration Guide

## Overview

This guide provides comprehensive instructions for deploying the RAG system to AWS, comparing different deployment architectures and their tradeoffs. The system consists of three main components:

1. **Embedding Service** (Python FastAPI)
2. **Orchestrator** (Node.js Express)
3. **Database** (MongoDB Atlas Vector Search)
4. **React Demo UI**

## Architecture Options

### 1. AWS Lambda + API Gateway (Serverless)

**Best for**: Cost-effective, variable workloads, automatic scaling

#### Components Mapping:
- **Embedding Service**: AWS Lambda + API Gateway
- **Orchestrator**: AWS Lambda + API Gateway
- **Database**: MongoDB Atlas (external) or Amazon DocumentDB
- **Frontend**: S3 + CloudFront

#### Pros:
- ✅ Pay-per-request pricing
- ✅ Automatic scaling from zero
- ✅ No server management
- ✅ Built-in monitoring with CloudWatch

#### Cons:
- ❌ Cold start latency (especially for ML models)
- ❌ 15-minute execution limit
- ❌ Memory limit (10GB max)
- ❌ Complex local development

#### Migration Steps:

```bash
# 1. Install Serverless Framework
npm install -g serverless
npm install serverless-python-requirements

# 2. Create serverless.yml for embedding service
```

**Embedding Service Lambda Configuration:**
```yaml
# serverless-embedding.yml
service: rag-embedding-service

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  memorySize: 3008  # For ML models
  timeout: 900      # 15 minutes max
  environment:
    OPENAI_API_KEY: ${env:OPENAI_API_KEY}
    MONGODB_URI: ${env:MONGODB_URI}
    MONGODB_DATABASE: ${env:MONGODB_DATABASE}
    MONGODB_COLLECTION: ${env:MONGODB_COLLECTION}

functions:
  embedding:
    handler: lambda_handler.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
          cors: true

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
    layer: true
```

**Lambda Handler (lambda_handler.py):**
```python
import os
from mangum import Mangum
from main import app

# Ensure environment variables are set
os.environ.setdefault("EMBEDDING_SERVICE_PORT", "8000")

handler = Mangum(app, lifespan="off")
```

**Orchestrator Lambda Configuration:**
```yaml
# serverless-orchestrator.yml
service: rag-orchestrator

provider:
  name: aws
  runtime: nodejs18.x
  region: us-east-1
  memorySize: 1024
  timeout: 300
  environment:
    OPENAI_API_KEY: ${env:OPENAI_API_KEY}
    EMBEDDING_SERVICE_URL: ${env:EMBEDDING_SERVICE_URL}

functions:
  orchestrator:
    handler: lambda_handler.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
          cors: true
```

**Node.js Lambda Handler (lambda_handler.js):**
```javascript
const serverless = require('serverless-http');
const app = require('./src/app');

module.exports.handler = serverless(app);
```

**Deployment Commands:**
```bash
# Deploy embedding service
cd services/embedding_service
serverless deploy -c serverless-embedding.yml

# Deploy orchestrator
cd services/orchestrator
serverless deploy -c serverless-orchestrator.yml

# Deploy frontend to S3
aws s3 sync demo-ui/build s3://your-rag-frontend-bucket
```

---

### 2. Amazon ECS with Fargate (Containerized)

**Best for**: Consistent performance, complex applications, Docker-familiar teams

#### Components Mapping:
- **Services**: ECS Tasks on Fargate
- **Load Balancing**: Application Load Balancer
- **Database**: MongoDB Atlas or DocumentDB
- **Frontend**: S3 + CloudFront or ECS service

#### Pros:
- ✅ No cold starts
- ✅ Consistent performance
- ✅ Easy migration from Docker Compose
- ✅ Full control over runtime environment

#### Cons:
- ❌ Always-on costs
- ❌ More complex setup
- ❌ Need to manage scaling

#### Migration Steps:

**ECS Task Definition (ecs-task-definition.json):**
```json
{
  "family": "rag-system",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "embedding-service",
      "image": "your-account.dkr.ecr.region.amazonaws.com/rag-embedding:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAI_API_KEY",
          "value": "your-openai-key"
        },
        {
          "name": "MONGODB_URI",
          "value": "your-mongodb-uri"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-embedding",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    },
    {
      "name": "orchestrator",
      "image": "your-account.dkr.ecr.region.amazonaws.com/rag-orchestrator:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAI_API_KEY",
          "value": "your-openai-key"
        },
        {
          "name": "EMBEDDING_SERVICE_URL",
          "value": "http://localhost:8000"
        }
      ],
      "dependsOn": [
        {
          "targetContainerName": "embedding-service",
          "condition": "START"
        }
      ]
    }
  ]
}
```

**Deployment Script:**
```bash
#!/bin/bash
# deploy-ecs.sh

# Build and push images
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com

# Build images
docker build -t rag-embedding services/embedding_service/
docker build -t rag-orchestrator services/orchestrator/

# Tag and push
docker tag rag-embedding:latest your-account.dkr.ecr.us-east-1.amazonaws.com/rag-embedding:latest
docker tag rag-orchestrator:latest your-account.dkr.ecr.us-east-1.amazonaws.com/rag-orchestrator:latest

docker push your-account.dkr.ecr.us-east-1.amazonaws.com/rag-embedding:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/rag-orchestrator:latest

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create or update service
aws ecs create-service \
  --cluster rag-cluster \
  --service-name rag-service \
  --task-definition rag-system:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

---

### 3. Amazon EC2 (Virtual Machines)

**Best for**: Full control, high-performance requirements, existing VM expertise

#### Components Mapping:
- **Services**: EC2 instances with Docker or direct installation
- **Load Balancing**: Application Load Balancer or NGINX
- **Database**: MongoDB Atlas, self-hosted MongoDB, or DocumentDB
- **Frontend**: Same EC2 or separate S3+CloudFront

#### Pros:
- ✅ Full control and customization
- ✅ Predictable performance
- ✅ Can use GPU instances for ML
- ✅ Easy debugging and monitoring

#### Cons:
- ❌ Server management overhead
- ❌ Manual scaling
- ❌ Higher operational complexity

#### Migration Steps:

**User Data Script (ec2-user-data.sh):**
```bash
#!/bin/bash
yum update -y
yum install -y docker git

# Start Docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone repository
cd /home/ec2-user
git clone https://github.com/your-repo/ai-engineer-challenge.git
cd ai-engineer-challenge

# Set environment variables
echo "OPENAI_API_KEY=your-key" >> .env
echo "MONGODB_URI=your-mongodb-uri" >> .env

# Start services
docker-compose up -d

# Setup log rotation
cat > /etc/logrotate.d/rag-system << EOF
/home/ec2-user/ai-engineer-challenge/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 ec2-user ec2-user
}
EOF
```

**CloudFormation Template (ec2-deployment.yaml):**
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'RAG System EC2 Deployment'

Parameters:
  InstanceType:
    Type: String
    Default: t3.large
    AllowedValues: [t3.medium, t3.large, t3.xlarge, c5.large, c5.xlarge]
  
  KeyPairName:
    Type: AWS::EC2::KeyPair::KeyName
    Description: EC2 Key Pair for SSH access

Resources:
  RAGSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for RAG system
      SecurityGroupIngresses:
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0
          Description: SSH access
        - IpProtocol: tcp
          FromPort: 5000
          ToPort: 5000
          CidrIp: 0.0.0.0/0
          Description: Orchestrator API
        - IpProtocol: tcp
          FromPort: 8000
          ToPort: 8000
          CidrIp: 0.0.0.0/0
          Description: Embedding Service API

  RAGInstance:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: !Ref InstanceType
      KeyName: !Ref KeyPairName
      ImageId: ami-0abcdef1234567890  # Amazon Linux 2
      SecurityGroupIds:
        - !Ref RAGSecurityGroup
      UserData:
        Fn::Base64: !Sub |
          ${ec2-user-data.sh content here}
      Tags:
        - Key: Name
          Value: RAG-System

Outputs:
  InstancePublicDNS:
    Description: Public DNS of the EC2 instance
    Value: !GetAtt RAGInstance.PublicDnsName
  
  OrchestratorURL:
    Description: Orchestrator API URL
    Value: !Sub 'http://${RAGInstance.PublicDnsName}:5000'
```

---

### 4. Amazon SageMaker (ML-Optimized)

**Best for**: ML-heavy workloads, model experimentation, Jupyter notebooks

#### Components Mapping:
- **Embedding Service**: SageMaker Endpoint
- **Orchestrator**: Lambda or ECS
- **Database**: MongoDB Atlas or DocumentDB
- **Model Training**: SageMaker Training Jobs

#### Pros:
- ✅ Optimized for ML workloads
- ✅ Built-in model versioning
- ✅ Auto-scaling inference endpoints
- ✅ Integrated with ML workflow tools

#### Cons:
- ❌ More complex setup
- ❌ Higher costs for simple models
- ❌ Learning curve for SageMaker specifics

#### Migration Steps:

**SageMaker Model Deployment (sagemaker-deploy.py):**
```python
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create HuggingFace Model
huggingface_model = HuggingFaceModel(
    transformers_version="4.21",
    pytorch_version="1.12",
    py_version="py39",
    role=role,
    model_data="s3://your-bucket/model.tar.gz",  # Your trained LoRA model
    entry_point="inference.py"
)

# Deploy model to endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="rag-embedding-endpoint"
)
```

---

## Database Options Comparison

### MongoDB Atlas (Recommended)
- ✅ Fully managed vector search
- ✅ Global clusters
- ✅ Built-in security and backups
- ❌ External dependency
- ❌ Data transfer costs

### Amazon DocumentDB with MongoDB Compatibility
- ✅ AWS-native
- ✅ VPC isolation
- ✅ Managed backups
- ❌ Limited MongoDB feature support
- ❌ No native vector search (requires custom implementation)

### Amazon OpenSearch with Vector Support
- ✅ Native AWS service
- ✅ Excellent vector search capabilities
- ✅ Built-in analytics
- ❌ Different query syntax
- ❌ Requires data migration

---

## Cost Analysis

### Monthly Cost Estimates (us-east-1, moderate usage)

| Architecture | Compute | Database | Storage | Total/Month |
|--------------|---------|----------|---------|-------------|
| **Lambda** | $50-200 | $50-100 | $10-20 | $110-320 |
| **ECS Fargate** | $150-300 | $50-100 | $20-40 | $220-440 |
| **EC2** | $100-400 | $50-100 | $20-50 | $170-550 |
| **SageMaker** | $200-800 | $50-100 | $30-50 | $280-950 |

*Costs vary significantly based on usage patterns, instance sizes, and data volume.*

---

## Security Considerations

### Secrets Management
```bash
# Store secrets in AWS Secrets Manager
aws secretsmanager create-secret \
  --name "rag-system/openai-key" \
  --description "OpenAI API Key for RAG System" \
  --secret-string "your-openai-api-key"

aws secretsmanager create-secret \
  --name "rag-system/mongodb-uri" \
  --description "MongoDB URI for RAG System" \
  --secret-string "your-mongodb-connection-string"
```

### IAM Roles
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": [
        "arn:aws:secretsmanager:*:*:secret:rag-system/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

### Network Security
- Use VPC for internal communication
- Implement Security Groups with least privilege
- Use ALB with SSL/TLS termination
- Enable VPC Flow Logs for monitoring

---

## Monitoring and Observability

### CloudWatch Setup
```bash
# Create custom metrics
aws logs create-log-group --log-group-name /aws/rag-system/embedding-service
aws logs create-log-group --log-group-name /aws/rag-system/orchestrator

# Set up alarms
aws cloudwatch put-metric-alarm \
  --alarm-name "RAG-HighErrorRate" \
  --alarm-description "High error rate in RAG system" \
  --metric-name "ErrorRate" \
  --namespace "RAG/System" \
  --statistic "Average" \
  --period 300 \
  --threshold 5.0 \
  --comparison-operator "GreaterThanThreshold" \
  --evaluation-periods 2
```

### X-Ray Tracing (for Lambda/ECS)
```python
# Add to your Python code
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

# Patch libraries
patch_all()

@xray_recorder.capture('generate_embedding')
async def get_embedding(self, text: str):
    # Your embedding logic here
    pass
```

---

## Migration Checklist

### Pre-Migration
- [ ] Set up AWS CLI and credentials
- [ ] Create ECR repositories for Docker images
- [ ] Configure MongoDB Atlas cluster
- [ ] Set up domain and SSL certificates
- [ ] Create IAM roles and policies

### Migration Steps
- [ ] Choose deployment architecture
- [ ] Update configuration for cloud environment
- [ ] Set up secrets management
- [ ] Deploy infrastructure (CloudFormation/CDK)
- [ ] Deploy applications
- [ ] Configure monitoring and alerting
- [ ] Set up CI/CD pipeline
- [ ] Test end-to-end functionality

### Post-Migration
- [ ] Monitor performance and costs
- [ ] Set up automated backups
- [ ] Configure log retention policies
- [ ] Document runbook procedures
- [ ] Train team on new environment

---

## Troubleshooting Guide

### Common Issues

1. **Cold Start Performance (Lambda)**
   - Use provisioned concurrency for critical functions
   - Implement connection pooling
   - Consider ECS for consistently high traffic

2. **Memory Issues**
   - Monitor CloudWatch metrics
   - Optimize model loading
   - Use appropriate instance sizes

3. **Network Connectivity**
   - Check Security Group rules
   - Verify VPC configuration
   - Test DNS resolution

4. **Database Connection Issues**
   - Verify connection strings
   - Check network access
   - Monitor connection pool sizes

### Performance Optimization
- Enable CloudFront for static assets
- Use ElastiCache for frequently accessed data
- Implement proper indexing strategies
- Monitor and optimize database queries

---

## Conclusion

Choose your deployment architecture based on:

- **Lambda**: Variable workloads, cost optimization, rapid development
- **ECS**: Consistent performance, containerized applications, moderate complexity
- **EC2**: Full control, high performance, existing infrastructure expertise
- **SageMaker**: ML-focused deployments, model experimentation, advanced ML workflows

Each architecture has its strengths and is suitable for different use cases and organizational requirements.

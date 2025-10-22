# DynamoDB Integration Guide

## Overview

The RAG application now uses **AWS DynamoDB** for persistent storage of:
- ✅ User query history
- ✅ Query answers and metadata
- ✅ User feedback (thumbs up/down)
- ✅ Validated Q&A pairs (for caching)
- ✅ Analytics and metrics

**Cost:** ~$1-5/month for typical usage (8 users, ~100 queries/day)

---

## 🚀 Quick Start (Local Development)

### 1. Start Local DynamoDB

```bash
# Start DynamoDB Local in Docker
docker-compose -f docker-compose.dynamodb.yml up -d

# Verify it's running
curl http://localhost:8000/
```

### 2. Create Tables

```bash
# Install dependencies if needed
pip install boto3

# Create tables (local mode)
python setup_dynamodb.py

# Verify tables were created
python setup_dynamodb.py --list
```

### 3. Migrate Existing Data (Optional)

```bash
# Migrate from logs/saved_answers.json to DynamoDB
python migrate_json_to_dynamodb.py
```

### 4. Start Application

```bash
./start.sh
```

The application will automatically detect and use local DynamoDB!

---

## 🗄️ Database Schema

### Table 1: QueryHistory
Stores all user queries and answers.

**Keys:**
- `PK` (Hash Key): `USER#{username}`
- `SK` (Range Key): `QUERY#{timestamp}`

**Attributes:**
```json
{
  "query_id": "john_2025-01-20T10:30:00",
  "query_text": "How to fix error?",
  "answer_text": "To fix this error...",
  "intent_type": "troubleshooting",
  "intent_confidence": 0.95,
  "sources": ["doc1.pdf", "doc2.pdf"],
  "confidence": 0.92,
  "response_time_ms": 450,
  "session_id": "abc123",
  "timestamp": "2025-01-20T10:30:00"
}
```

**Global Secondary Index (GSI):**
- `DateIndex`: Query by date range
  - `GSI1PK`: `DATE#{YYYY-MM-DD}`
  - `GSI1SK`: `timestamp`

---

### Table 2: Feedback
Stores user feedback on answers.

**Keys:**
- `PK` (Hash Key): `QUERY#{query_id}`
- `SK` (Range Key): `FEEDBACK#{timestamp}`

**Attributes:**
```json
{
  "user": "john",
  "is_helpful": true,
  "feedback_text": "Very helpful!",
  "timestamp": "2025-01-20T10:31:00"
}
```

---

### Table 3: ValidatedQnA
Stores validated Q&A pairs for fast cache lookup.

**Keys:**
- `query_hash` (Hash Key): MD5 hash of query text

**Attributes:**
```json
{
  "query_text": "How to fix error?",
  "answer_text": "To fix this error...",
  "sources": ["doc1.pdf"],
  "helpful_count": 5,
  "unhelpful_count": 0,
  "last_used": "2025-01-20T10:30:00",
  "is_active": true
}
```

---

## 💻 Using the Database Manager

### Python API

```python
from utils.dynamodb_manager import DynamoDBManager

# Initialize (auto-detects local vs AWS)
db = DynamoDBManager()

# Save a query
query_id = db.save_query(
    user="john",
    query_text="How to fix error?",
    answer_text="To fix this error...",
    intent_type="troubleshooting",
    intent_confidence=0.95,
    sources=["doc1.pdf"],
    confidence=0.92,
    response_time_ms=450,
    session_id="session123"
)

# Get user's query history
history = db.get_user_query_history(user="john", limit=20)

# Save feedback
db.save_feedback(
    query_id=query_id,
    user="john",
    is_helpful=True,
    feedback_text="Very helpful!"
)

# Get analytics
stats = db.get_feedback_stats()
print(f"Helpful rate: {stats['helpful_percentage']:.1f}%")

metrics = db.get_average_metrics(days=30)
print(f"Avg confidence: {metrics['avg_confidence']:.2f}")

intent_dist = db.get_intent_distribution(days=30)
print(f"Intent distribution: {intent_dist}")
```

---

## 🌐 AWS Deployment

### Prerequisites
1. AWS account
2. AWS CLI configured
3. IAM permissions for DynamoDB

### Setup on AWS

```bash
# Set AWS mode
export AWS_REGION=us-east-1

# Create tables on AWS
python setup_dynamodb.py --aws

# Verify
python setup_dynamodb.py --aws --list
```

### IAM Permissions Required

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:Scan",
        "dynamodb:UpdateItem",
        "dynamodb:DescribeTable"
      ],
      "Resource": [
        "arn:aws:dynamodb:*:*:table/RAG_*"
      ]
    }
  ]
}
```

### Environment Variables

```bash
# For AWS deployment
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# Or use IAM roles (recommended)
# No env vars needed if using EC2/ECS with IAM roles
```

---

## 📊 Cost Breakdown

### Local Development
- **Cost:** $0 (DynamoDB Local is free)

### AWS Production (8 users, ~100 queries/day)

**DynamoDB Costs:**
- Write requests: ~3,000/month × $1.25 per million = **$0.004**
- Read requests: ~10,000/month × $0.25 per million = **$0.003**
- Storage: 1GB × $0.25/GB = **$0.25**
- **Total: ~$0.26/month** (under free tier!)

**With Free Tier:**
- 25GB storage free
- 25 write capacity units free
- 25 read capacity units free
- **Effectively $0/month for your usage!**

**Even at 10x volume (1,000 queries/day):**
- **Cost: ~$2-3/month**

---

## 🔍 Querying and Analytics

### Common Queries

```python
from utils.dynamodb_manager import DynamoDBManager
from datetime import datetime, timedelta

db = DynamoDBManager()

# Get recent queries for a user
history = db.get_user_query_history(
    user="john",
    limit=20,
    start_date=datetime.utcnow() - timedelta(days=7)
)

# Get all feedback for a query
feedback = db.get_query_feedback(query_id="john_2025-01-20T10:30:00")

# Get validated answer (cache lookup)
cached = db.get_validated_answer("How to fix error?")
if cached:
    print("Using cached answer!")

# Analytics
stats = db.get_feedback_stats()
metrics = db.get_average_metrics(days=30)
intent_dist = db.get_intent_distribution(days=30)
```

### Export to S3 + Athena (Future)

For complex analytics:
1. Enable DynamoDB Point-in-Time backup
2. Export to S3
3. Query with Amazon Athena (SQL)

---

## 🛠️ Troubleshooting

### Local DynamoDB Issues

**Problem:** "Cannot connect to http://localhost:8000"

**Solution:**
```bash
# Check if container is running
docker ps | grep dynamodb

# Start container
docker-compose -f docker-compose.dynamodb.yml up -d

# Check logs
docker logs rag-dynamodb-local

# Restart
docker-compose -f docker-compose.dynamodb.yml restart
```

---

**Problem:** "ResourceNotFoundException: Table does not exist"

**Solution:**
```bash
# Create tables
python setup_dynamodb.py

# Verify
python setup_dynamodb.py --list
```

---

### AWS Issues

**Problem:** "UnrecognizedClientException"

**Solution:**
```bash
# Check AWS credentials
aws configure list

# Verify permissions
aws dynamodb list-tables
```

---

**Problem:** "ResourceInUseException: Table already exists"

**Solution:**
- Tables already exist, this is fine!
- Check: `python setup_dynamodb.py --aws --list`

---

## 📈 Monitoring

### Local Development
```python
# Get stats
from utils.dynamodb_manager import DynamoDBManager
db = DynamoDBManager()

print("Feedback stats:", db.get_feedback_stats())
print("Metrics:", db.get_average_metrics())
print("Intent distribution:", db.get_intent_distribution())
```

### AWS Production
- AWS Console → DynamoDB → Tables → Metrics
- CloudWatch for detailed monitoring
- Set up alarms for high read/write usage

---

## 🔄 Migration Path

### From JSON Files
```bash
# Automatic migration from logs/saved_answers.json
python migrate_json_to_dynamodb.py
```

### From PostgreSQL (Future)
```python
# TODO: Create migration script if needed
```

---

## 🚀 Performance Tips

1. **Use consistent naming** for PK/SK to enable efficient queries
2. **Enable caching** in your application layer
3. **Use GSI sparingly** (they cost extra)
4. **Batch operations** when possible
5. **Monitor costs** via AWS Cost Explorer

---

## 📚 Additional Resources

- [DynamoDB Best Practices](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/best-practices.html)
- [DynamoDB Local](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/DynamoDBLocal.html)
- [Boto3 DynamoDB](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html)

---

## ✅ Next Steps

1. ✅ Tables created locally
2. ⏭️ Integrate with your application
3. ⏭️ Test all operations
4. ⏭️ Deploy to AWS when ready
5. ⏭️ Set up monitoring/alerts

---

**Questions?** Check logs at `utils/dynamodb_manager.py` for debugging.


# Auto-TPM Risk Radar

AI-Native Project Intelligence Platform for Technical Program Management.

Predicts sprint slippage at the ticket level using XGBoost and generates executive status reports using Amazon Bedrock (Claude), by fusing structured Jira data with unstructured Slack sentiment signals.

## Architecture

```
┌──────────────┐     ┌──────────────┐
│   Jira API   │     │  Slack API   │
│  (OAuth 1.0a)│     │ (Bot Token)  │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────────────────────────────┐
│     MLA Domain 1: Data Ingestion     │
│  ┌────────────┐  ┌────────────────┐  │
│  │Jira Crawler│  │ Slack Crawler  │  │
│  │+ Pagination│  │+ VADER Sentimnt│  │
│  └─────┬──────┘  └───────┬────────┘  │
│        │  ┌───────────┐  │           │
│        └──► PII Redact◄──┘           │
│           │(Comprehend)│             │
│           └─────┬─────┘             │
└─────────────────┼────────────────────┘
                  ▼
┌──────────────────────────────────────┐
│        S3 Data Lake (Medallion)      │
│  Bronze (raw) → Silver (clean) →    │
│  Gold (features)  [Parquet/Snappy]   │
│        + Glue Data Catalog           │
└─────────────────┬────────────────────┘
                  ▼
┌──────────────────────────────────────┐
│   MLA Domain 2: Risk Modeling        │
│  ┌──────────────────────────────┐    │
│  │  Feature Engineering         │    │
│  │  complexity_per_person,      │    │
│  │  sentiment_velocity,         │    │
│  │  priority_interaction, etc.  │    │
│  └──────────┬───────────────────┘    │
│             ▼                        │
│  ┌──────────────────────────────┐    │
│  │  XGBoost Binary Classifier   │    │
│  │  Walk-Forward CV             │    │
│  │  Optimized for Recall        │    │
│  └──────────┬───────────────────┘    │
└─────────────┼────────────────────────┘
              ▼
┌──────────────────────────────────────┐
│   AIF: GenAI Reporting (Bedrock)     │
│  ┌──────────────────────────────┐    │
│  │  RAG + Prompt Engineering    │    │
│  │  Claude Sonnet (reports)     │    │
│  │  Claude Haiku (classify)     │    │
│  │  Temperature: 0.1            │    │
│  └──────────┬───────────────────┘    │
└─────────────┼────────────────────────┘
              ▼
┌──────────────────────────────────────┐
│   API Layer (Lambda + API Gateway)   │
│   /risk, /report, /radar, /feedback  │
└─────────────┬────────────────────────┘
              ▼
┌──────────────────────────────────────┐
│   Dashboard (Flask + Canvas)         │
│  Risk Radar Scatter Plot             │
│  Auto-Draft Split-Screen Reports     │
│  Human-in-the-Loop Feedback (RLHF)  │
└──────────────────────────────────────┘
```

## Project Structure

```
├── src/
│   ├── config/          # Pydantic settings management
│   ├── ingestion/       # Jira crawler, Slack crawler, PII redaction
│   ├── datalake/        # S3 Medallion storage, Glue catalog
│   ├── features/        # Feature engineering (21 features)
│   ├── models/          # XGBoost risk model, drift monitoring
│   ├── reporting/       # Bedrock GenAI, prompt templates, RAG
│   └── api/             # Flask API / Lambda handler
├── dashboard/           # Web UI (templates, CSS, JS)
├── scripts/             # Ingestion and training entry points
├── infrastructure/      # CloudFormation / SAM templates
└── tests/               # 71 tests across all domains
```

## Key Features

### MLA Domain 1: Data Ingestion
- **Jira**: OAuth 1.0a, incremental JQL extraction, dynamic custom field resolution, sprint history parsing
- **Slack**: Tiered sentiment analysis (VADER Tier 1 + LLM Tier 2 on anomaly), engineering-specific lexicon
- **PII Redaction**: Amazon Comprehend entity detection + regex-based secret scanning (AWS keys, DB URIs, private keys)
- **Data Lake**: Bronze/Silver/Gold Medallion architecture on S3 with Parquet/Snappy compression

### MLA Domain 2: Risk Modeling
- **21 Engineered Features**: Including `complexity_per_person`, `sentiment_velocity`, `priority_interaction`, `days_in_status`, `description_ambiguity`, `ticket_churn`
- **XGBoost Classifier**: Binary prediction of sprint slippage with native missing-value handling
- **Walk-Forward CV**: Time-series cross-validation to prevent future data leakage
- **Explainability**: Feature importance (gain-based) with per-ticket risk factor attribution

### AIF: GenAI Reporting
- **Amazon Bedrock**: Claude 3 Sonnet for executive reports, Haiku for ticket classification
- **RAG Architecture**: Context-grounded generation with citation support
- **Prompt Engineering**: XML-tagged data isolation, Chain of Thought, structured JSON output
- **Hallucination Guardrails**: Temperature 0.1, citation requirements

### MLA Domain 3: Deployment
- **Serverless**: SageMaker Serverless Inference, Lambda, API Gateway
- **Security**: KMS encryption, IAM least privilege, VPC isolation, PII redaction
- **CloudFormation**: Full IaC for S3, DynamoDB, Glue, Lambda, API Gateway, SNS

### MLA Domain 4: MLOps
- **Data Quality**: Deequ-inspired constraint validation (completeness, uniqueness, value ranges)
- **Drift Detection**: Feature distribution monitoring with z-score based anomaly detection
- **Automated Retraining**: Pipeline with quality gates and model registry
- **Alerting**: SNS notifications for quality violations and drift events

### Dashboard
- **Risk Radar**: Interactive scatter plot (Complexity x Sentiment x Story Points x Risk Level)
- **Auto-Draft Reports**: Split-screen with AI draft (editable) + Evidence Locker (citations)
- **RLHF Feedback Loop**: Human edits captured for future model fine-tuning

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/projects/{key}/risk` | Ticket-level risk predictions |
| GET | `/api/v1/projects/{key}/radar` | Risk Radar visualization data |
| POST | `/api/v1/projects/{key}/report` | Generate executive status report |
| GET | `/api/v1/tickets/{key}/risk` | Single ticket risk analysis |
| POST | `/api/v1/reports/{key}/feedback` | Submit report feedback (RLHF) |

## Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Start dashboard locally
python dashboard/app.py

# Run Jira ingestion
python scripts/ingest_jira.py PROJ

# Train model
python scripts/train_model.py PROJ
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `JIRA_BASE_URL` | Jira instance base URL |
| `JIRA_CONSUMER_KEY` | OAuth consumer key |
| `JIRA_ACCESS_TOKEN` | OAuth access token |
| `JIRA_PRIVATE_KEY_SSM_PARAM` | SSM parameter name for RSA private key |
| `SLACK_BOT_TOKEN` | Slack bot OAuth token |
| `SLACK_CHANNELS` | Comma-separated channel IDs |
| `S3_BUCKET` | S3 data lake bucket name |
| `DYNAMODB_TABLE` | DynamoDB ingestion state table |
| `BEDROCK_MODEL_ID` | Bedrock model ID for report generation |
| `AWS_REGION` | AWS region (default: us-east-1) |

## Deploy

```bash
# Deploy infrastructure with SAM
sam build --template infrastructure/cloudformation/template.yaml
sam deploy --guided
```

## Testing

71 tests covering all domains:
- **Ingestion**: Jira parsing, sprint history, sentiment analysis, PII redaction
- **Features**: Feature engineering, team aggregation, sentiment merging
- **Models**: XGBoost training, prediction, save/load, threshold validation
- **MLOps**: Data quality constraints, drift detection
- **Reporting**: Prompt template formatting
- **API**: Health, feedback endpoints

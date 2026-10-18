#!/usr/bin/env bash
# Deploy the Split Step Counter API to AWS Lambda with a public Function URL.
#
# Prereqs: AWS CLI v2 configured (`aws configure`), Docker running.
# Usage (from the repo root, in Git Bash / WSL / macOS / Linux):
#
#   ALLOWED_ORIGINS="https://www.your-site.com" ./deploy/deploy_lambda.sh
#
# Safe to re-run: existing resources are reused and the code is updated.
#
# Creates:
#   - S3 bucket for uploads (uploads/ auto-deleted after 1 day, browser CORS)
#   - ECR repository + the Lambda container image
#   - IAM role: logs + read/write only under s3://<bucket>/uploads/
#   - Lambda function (3 GB RAM, 5 min timeout, capped concurrency)
#   - Public Function URL
set -euo pipefail

NAME="${NAME:-split-step-api}"
REGION="${AWS_REGION:-$(aws configure get region || true)}"
REGION="${REGION:-us-east-1}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${UPLOAD_BUCKET:-${NAME}-uploads-${ACCOUNT}-${REGION}}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:?Set ALLOWED_ORIGINS to your website origin, e.g. https://www.example.com}"
MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-200}"
MAX_VIDEO_SECONDS="${MAX_VIDEO_SECONDS:-600}"
RATE_LIMIT_PER_HOUR="${RATE_LIMIT_PER_HOUR:-20}"
# Hard cap on simultaneous executions = hard cap on cost from abuse.
CONCURRENCY="${CONCURRENCY:-3}"

REPO_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${NAME}"
ROLE="${NAME}-role"
export AWS_REGION="$REGION" AWS_PAGER=""

echo "==> Account ${ACCOUNT}, region ${REGION}, bucket ${BUCKET}"

# ── S3 upload bucket ──────────────────────────────────────────────────────────
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET"
  else
    aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION"
  fi
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration '{
  "Rules": [{"ID": "expire-uploads", "Status": "Enabled", "Filter": {"Prefix": "uploads/"},
             "Expiration": {"Days": 1}}]}'
ORIGINS_JSON="$(printf '%s' "$ALLOWED_ORIGINS" | sed 's/ //g; s/,/","/g')"
aws s3api put-bucket-cors --bucket "$BUCKET" --cors-configuration "{
  \"CORSRules\": [{\"AllowedOrigins\": [\"${ORIGINS_JSON}\"], \"AllowedMethods\": [\"POST\"],
                   \"AllowedHeaders\": [\"*\"], \"MaxAgeSeconds\": 3600}]}"

# ── IAM role ──────────────────────────────────────────────────────────────────
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                   "Action": "sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "    waiting for the new role to propagate…"; sleep 15
fi
# PutObject lets the presigned POSTs this role signs succeed; GetObject is for counting.
aws iam put-role-policy --role-name "$ROLE" --policy-name uploads-access --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [{\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\", \"s3:PutObject\"],
                   \"Resource\": \"arn:aws:s3:::${BUCKET}/uploads/*\"}]}"
ROLE_ARN="$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)"

# ── Image ─────────────────────────────────────────────────────────────────────
aws ecr describe-repositories --repository-names "$NAME" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$NAME" >/dev/null
aws ecr get-login-password | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
docker build --platform linux/amd64 --provenance=false -f Dockerfile.lambda -t "${REPO_URI}:latest" .
docker push "${REPO_URI}:latest"

# ── Function ──────────────────────────────────────────────────────────────────
# JSON form, because the shorthand syntax can't hold commas inside ALLOWED_ORIGINS
ENV_VARS="{\"Variables\": {\"UPLOAD_BUCKET\": \"${BUCKET}\", \"ALLOWED_ORIGINS\": \"${ALLOWED_ORIGINS}\", \"MAX_UPLOAD_MB\": \"${MAX_UPLOAD_MB}\", \"MAX_VIDEO_SECONDS\": \"${MAX_VIDEO_SECONDS}\", \"RATE_LIMIT_PER_HOUR\": \"${RATE_LIMIT_PER_HOUR}\"}}"
if aws lambda get-function --function-name "$NAME" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$NAME" --image-uri "${REPO_URI}:latest" >/dev/null
  aws lambda wait function-updated --function-name "$NAME"
  aws lambda update-function-configuration --function-name "$NAME" --environment "$ENV_VARS" >/dev/null
else
  aws lambda create-function --function-name "$NAME" --package-type Image \
    --code ImageUri="${REPO_URI}:latest" --role "$ROLE_ARN" \
    --memory-size 3008 --timeout 300 --ephemeral-storage Size=2048 \
    --environment "$ENV_VARS" >/dev/null
fi
aws lambda wait function-updated --function-name "$NAME"
# New accounts often have a total limit of 10, which leaves nothing to reserve;
# in that case the account limit itself already caps the cost.
aws lambda put-function-concurrency --function-name "$NAME" \
  --reserved-concurrent-executions "$CONCURRENCY" >/dev/null \
  || echo "    (could not reserve concurrency; account limit applies instead)"

# ── Public Function URL (CORS is handled by the app, so it's left off here) ───
if ! aws lambda get-function-url-config --function-name "$NAME" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$NAME" --auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$NAME" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null
fi
URL="$(aws lambda get-function-url-config --function-name "$NAME" --query FunctionUrl --output text)"

echo
echo "==> Deployed: ${URL}"
echo "    Test:  curl ${URL}health"
echo "    Web app: ${URL}app/"
echo "    To embed on your own site, put ${URL%/} into API_URL in web/index.html."

#!/usr/bin/env bash
# Tear down anything left behind by `test-windows-gpu.sh` — finds resources
# tagged `zeusd-test=*` and deletes them in the right order.
#
# Use cases:
#   - A previous test-windows-gpu.sh run crashed before its cleanup trap ran
#     (SIGKILL, OOM, credentials expired mid-cleanup, etc.).
#   - You started a run with `--keep` and now want to tear it down.
#   - You want to confirm nothing zeusd-test is still billing.
#
# Usage:
#   ./cleanup-test-aws.sh                # list + delete everything tagged zeusd-test=*
#   ./cleanup-test-aws.sh --dry-run      # list only, don't delete
#   ./cleanup-test-aws.sh --tag-value X  # only resources whose tag value is exactly X
#
# Prereqs: aws CLI v2 with valid credentials, jq.
set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"
DRY_RUN=0
TAG_VALUE_FILTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)    REGION="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --tag-value) TAG_VALUE_FILTER="$2"; shift 2 ;;
    -h|--help)   sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    *)           echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

command -v aws >/dev/null || { echo "aws CLI v2 required"; exit 1; }
command -v jq  >/dev/null || { echo "jq required"; exit 1; }
aws sts get-caller-identity --output text --query Arn >/dev/null || {
  echo "AWS credentials missing or expired. Log in and re-run." >&2
  exit 1
}

# `aws ec2 describe-*` filters use Name=tag:KEY,Values=...
# `aws iam list-*` does not natively filter on tags; we list, then jq-filter
# by tag client-side.
EC2_FILTER=("Name=tag-key,Values=zeusd-test")
if [[ -n "$TAG_VALUE_FILTER" ]]; then
  EC2_FILTER=("Name=tag:zeusd-test,Values=$TAG_VALUE_FILTER")
fi

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  DRY-RUN: $*"
  else
    "$@" 2>&1 | sed 's/^/    /' || true
  fi
}

iam_has_tag() {
  # arg1: JSON tag list. Echoes "yes" if tag matches the filter; "" otherwise.
  local tags="$1"
  if [[ -n "$TAG_VALUE_FILTER" ]]; then
    echo "$tags" | jq -er --arg v "$TAG_VALUE_FILTER" \
      'map(select(.Key=="zeusd-test" and .Value==$v)) | length>0' >/dev/null 2>&1 && echo yes
  else
    echo "$tags" | jq -er 'map(select(.Key=="zeusd-test")) | length>0' >/dev/null 2>&1 && echo yes
  fi
}

echo "== EC2 instances (tag zeusd-test${TAG_VALUE_FILTER:+ = $TAG_VALUE_FILTER}) =="
mapfile -t INSTANCES < <(aws ec2 describe-instances --region "$REGION" \
  --filters "${EC2_FILTER[@]}" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null | tr '\t' '\n' | sed '/^$/d')
if [[ ${#INSTANCES[@]} -gt 0 ]]; then
  echo "  Found: ${INSTANCES[*]}"
  run aws ec2 terminate-instances --region "$REGION" --instance-ids "${INSTANCES[@]}"
  if [[ $DRY_RUN -eq 0 ]]; then
    echo "  Waiting for termination..."
    aws ec2 wait instance-terminated --region "$REGION" --instance-ids "${INSTANCES[@]}" 2>/dev/null || true
  fi
else
  echo "  none"
fi

echo "== Security groups =="
mapfile -t SGS < <(aws ec2 describe-security-groups --region "$REGION" \
  --filters "${EC2_FILTER[@]}" --query 'SecurityGroups[].GroupId' --output text 2>/dev/null | tr '\t' '\n' | sed '/^$/d')
for sg in "${SGS[@]}"; do
  echo "  $sg"
  run aws ec2 delete-security-group --region "$REGION" --group-id "$sg"
done
[[ ${#SGS[@]} -eq 0 ]] && echo "  none"

echo "== Key pairs =="
mapfile -t KEYS < <(aws ec2 describe-key-pairs --region "$REGION" \
  --filters "${EC2_FILTER[@]}" --query 'KeyPairs[].KeyName' --output text 2>/dev/null | tr '\t' '\n' | sed '/^$/d')
for k in "${KEYS[@]}"; do
  echo "  $k"
  run aws ec2 delete-key-pair --region "$REGION" --key-name "$k"
done
[[ ${#KEYS[@]} -eq 0 ]] && echo "  none"

echo "== IAM instance profiles (tag zeusd-test${TAG_VALUE_FILTER:+ = $TAG_VALUE_FILTER}) =="
PROFILES_JSON=$(aws iam list-instance-profiles --query 'InstanceProfiles[]' --output json)
PROFILE_NAMES=()
while IFS= read -r row; do
  name=$(echo "$row" | jq -r '.InstanceProfileName')
  tags=$(aws iam list-instance-profile-tags --instance-profile-name "$name" --query Tags --output json 2>/dev/null || echo '[]')
  if [[ -n "$(iam_has_tag "$tags")" ]]; then
    PROFILE_NAMES+=("$name")
  fi
done < <(echo "$PROFILES_JSON" | jq -c '.[]')
for p in "${PROFILE_NAMES[@]}"; do
  echo "  $p"
  # Detach any roles first
  while IFS= read -r r; do
    [[ -z "$r" ]] && continue
    run aws iam remove-role-from-instance-profile --instance-profile-name "$p" --role-name "$r"
  done < <(aws iam get-instance-profile --instance-profile-name "$p" --query 'InstanceProfile.Roles[].RoleName' --output text 2>/dev/null | tr '\t' '\n')
  run aws iam delete-instance-profile --instance-profile-name "$p"
done
[[ ${#PROFILE_NAMES[@]} -eq 0 ]] && echo "  none"

echo "== IAM roles (tag zeusd-test${TAG_VALUE_FILTER:+ = $TAG_VALUE_FILTER}) =="
ROLE_NAMES=()
while IFS= read -r row; do
  name=$(echo "$row" | jq -r '.RoleName')
  tags=$(aws iam list-role-tags --role-name "$name" --query Tags --output json 2>/dev/null || echo '[]')
  if [[ -n "$(iam_has_tag "$tags")" ]]; then
    ROLE_NAMES+=("$name")
  fi
done < <(aws iam list-roles --query 'Roles[]' --output json | jq -c '.[]')
for r in "${ROLE_NAMES[@]}"; do
  echo "  $r"
  # Detach any managed policies
  while IFS= read -r arn; do
    [[ -z "$arn" ]] && continue
    run aws iam detach-role-policy --role-name "$r" --policy-arn "$arn"
  done < <(aws iam list-attached-role-policies --role-name "$r" --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null | tr '\t' '\n')
  run aws iam delete-role --role-name "$r"
done
[[ ${#ROLE_NAMES[@]} -eq 0 ]] && echo "  none"

echo "Done."

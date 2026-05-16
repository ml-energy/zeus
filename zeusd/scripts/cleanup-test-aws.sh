#!/usr/bin/env bash
# List or delete resources left behind by `test-windows-gpu.sh`.
#
# All resources are tagged with key `zeusd-test` and a unique per-run value
# (timestamp + PID + random suffix). The default action is LIST ONLY, which
# is safe to run on a shared AWS account where multiple devs may have test
# resources in flight. Deletion requires an explicit scope so a stale-orphan
# cleanup cannot accidentally take down someone else's running test.
#
# Use cases:
#   - A previous test-windows-gpu.sh run crashed before its cleanup trap ran
#     (SIGKILL, OOM, credentials expired mid-cleanup, etc.).
#   - You started a run with `--keep` and now want to tear it down.
#   - Account hygiene: find stale orphan resources without harming in-flight runs.
#
# Usage:
#   List (safe default; never deletes):
#     ./cleanup-test-aws.sh                          # everything tagged zeusd-test
#     ./cleanup-test-aws.sh --tag-value V            # one specific run
#     ./cleanup-test-aws.sh --older-than 4h          # resources older than 4 hours
#
#   Delete (must be explicit):
#     ./cleanup-test-aws.sh --delete --tag-value V   # one specific run
#     ./cleanup-test-aws.sh --delete --older-than 4h # stale orphans only
#     ./cleanup-test-aws.sh --delete --all           # everything (prompts unless --yes)
#     ./cleanup-test-aws.sh --delete --all --yes     # everything (no prompt)
#
# `--older-than` accepts Ns, Nm, Nh, Nd (seconds, minutes, hours, days).
# It filters EC2 instances and IAM resources by their LaunchTime/CreateDate.
# Security groups and key pairs lack a creation timestamp in the AWS API,
# so they are deleted only when their tag value matches a deleted instance
# or IAM resource (i.e., they ride along with their owning run, never on
# their own under --older-than).
#
# Prereqs: aws CLI v2 with valid credentials, jq.
set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"
ACTION="list"
TAG_VALUE_FILTER=""
OLDER_THAN=""
ALL_OK=0
ASSUME_YES=0

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \?//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)      REGION="$2"; shift 2 ;;
    --tag-value)   TAG_VALUE_FILTER="$2"; shift 2 ;;
    --older-than)  OLDER_THAN="$2"; shift 2 ;;
    --delete)      ACTION="delete"; shift ;;
    --all)         ALL_OK=1; shift ;;
    --yes|-y)      ASSUME_YES=1; shift ;;
    --dry-run)     ACTION="list"; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$ACTION" == "delete" && -z "$TAG_VALUE_FILTER" && -z "$OLDER_THAN" && $ALL_OK -ne 1 ]]; then
  cat >&2 <<EOF
Refusing to delete without a scope. Choose one:
  --tag-value <V>      delete exactly one run's resources
  --older-than <dur>   delete stale orphans (e.g. 4h, 2d)
  --all                delete every zeusd-test resource (prompts unless --yes)
EOF
  exit 2
fi

command -v aws >/dev/null || { echo "aws CLI v2 required"; exit 1; }
command -v jq  >/dev/null || { echo "jq required"; exit 1; }
aws sts get-caller-identity --output text --query Arn >/dev/null || {
  echo "AWS credentials missing or expired. Log in and re-run." >&2
  exit 1
}

parse_duration() {
  local s="$1"
  case "$s" in
    *s) echo "${s%s}" ;;
    *m) echo "$((${s%m} * 60))" ;;
    *h) echo "$((${s%h} * 3600))" ;;
    *d) echo "$((${s%d} * 86400))" ;;
    *)  echo "invalid duration (use Ns/Nm/Nh/Nd): $s" >&2; return 1 ;;
  esac
}

CUTOFF_EPOCH=""
if [[ -n "$OLDER_THAN" ]]; then
  CUTOFF_EPOCH=$(( $(date +%s) - $(parse_duration "$OLDER_THAN") ))
fi

iso_to_epoch() {
  # Try GNU date, then BSD date. Strip fractional seconds and trailing Z for BSD.
  date -u -d "$1" +%s 2>/dev/null && return 0
  local clean="${1%%.*}"; clean="${clean%Z}"
  date -u -j -f "%Y-%m-%dT%H:%M:%S" "$clean" +%s 2>/dev/null
}

is_too_new() {
  # 0 (true) if the resource is newer than the cutoff, i.e. should be skipped.
  [[ -z "$CUTOFF_EPOCH" ]] && return 1
  local launch="$1"
  [[ -z "$launch" || "$launch" == "None" ]] && return 1
  local epoch
  epoch=$(iso_to_epoch "$launch") || return 1
  [[ "$epoch" -ge "$CUTOFF_EPOCH" ]]
}

EC2_FILTER=("Name=tag-key,Values=zeusd-test")
[[ -n "$TAG_VALUE_FILTER" ]] && EC2_FILTER=("Name=tag:zeusd-test,Values=$TAG_VALUE_FILTER")

run() {
  if [[ "$ACTION" == "list" ]]; then
    echo "    WOULD: $*"
  else
    "$@" 2>&1 | sed 's/^/      /' || true
  fi
}

# Confirmation gate for blanket delete (no scope filter, `--all` passed).
if [[ "$ACTION" == "delete" && $ALL_OK -eq 1 && -z "$TAG_VALUE_FILTER" && -z "$OLDER_THAN" && $ASSUME_YES -ne 1 ]]; then
  echo "About to DELETE every resource tagged 'zeusd-test' in $REGION."
  echo "This will affect other Zeus devs sharing this account if they have in-flight tests."
  read -r -p "Type 'yes' to confirm: " confirm
  [[ "$confirm" == "yes" ]] || { echo "Aborted."; exit 1; }
fi

# Track tag values of resources we actually targeted, so SGs and key pairs
# (which lack a creation timestamp) can ride along under --older-than mode.
declare -A TARGETED_TAG_VALUES

mark_targeted() {
  local tv="$1"
  [[ -n "$tv" && "$tv" != "None" ]] && TARGETED_TAG_VALUES["$tv"]=1
}

# Returns 0 (true) if the resource is eligible for deletion under the current
# filters. For SGs and key pairs (no age info), eligibility under --older-than
# is "tag value matches a resource we already targeted."
ride_along_eligible() {
  local tv="$1"
  [[ -z "$OLDER_THAN" ]] && return 0
  [[ -n "${TARGETED_TAG_VALUES[$tv]:-}" ]]
}

echo "== EC2 instances =="
EC2_TARGETS=()
while IFS=$'\t' read -r iid tv launch state; do
  [[ -z "$iid" ]] && continue
  if is_too_new "$launch"; then
    echo "  $iid  tag=$tv  launch=$launch  state=$state  [skip: newer than $OLDER_THAN]"
    continue
  fi
  echo "  $iid  tag=$tv  launch=$launch  state=$state"
  EC2_TARGETS+=("$iid")
  mark_targeted "$tv"
done < <(aws ec2 describe-instances --region "$REGION" \
  --filters "${EC2_FILTER[@]}" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].[InstanceId, (Tags[?Key==`zeusd-test`]|[0].Value), LaunchTime, State.Name]' \
  --output text 2>/dev/null | sed '/^$/d')
if [[ ${#EC2_TARGETS[@]} -eq 0 ]]; then
  echo "  none to act on"
else
  run aws ec2 terminate-instances --region "$REGION" --instance-ids "${EC2_TARGETS[@]}"
  if [[ "$ACTION" == "delete" ]]; then
    echo "  Waiting for termination..."
    aws ec2 wait instance-terminated --region "$REGION" --instance-ids "${EC2_TARGETS[@]}" 2>/dev/null || true
  fi
fi

echo "== Security groups =="
SG_TARGETS=()
while IFS=$'\t' read -r sgid tv; do
  [[ -z "$sgid" ]] && continue
  if ! ride_along_eligible "$tv"; then
    echo "  $sgid  tag=$tv  [skip: --older-than gates SGs to tag values that had an aged instance/role]"
    continue
  fi
  echo "  $sgid  tag=$tv"
  SG_TARGETS+=("$sgid")
done < <(aws ec2 describe-security-groups --region "$REGION" \
  --filters "${EC2_FILTER[@]}" \
  --query 'SecurityGroups[].[GroupId, (Tags[?Key==`zeusd-test`]|[0].Value)]' \
  --output text 2>/dev/null | sed '/^$/d')
if [[ ${#SG_TARGETS[@]} -eq 0 ]]; then
  echo "  none to act on"
else
  for sg in "${SG_TARGETS[@]}"; do
    run aws ec2 delete-security-group --region "$REGION" --group-id "$sg"
  done
fi

echo "== Key pairs =="
KEY_TARGETS=()
while IFS=$'\t' read -r kname tv; do
  [[ -z "$kname" ]] && continue
  if ! ride_along_eligible "$tv"; then
    echo "  $kname  tag=$tv  [skip: --older-than gates keys to tag values that had an aged instance/role]"
    continue
  fi
  echo "  $kname  tag=$tv"
  KEY_TARGETS+=("$kname")
done < <(aws ec2 describe-key-pairs --region "$REGION" \
  --filters "${EC2_FILTER[@]}" \
  --query 'KeyPairs[].[KeyName, (Tags[?Key==`zeusd-test`]|[0].Value)]' \
  --output text 2>/dev/null | sed '/^$/d')
if [[ ${#KEY_TARGETS[@]} -eq 0 ]]; then
  echo "  none to act on"
else
  for k in "${KEY_TARGETS[@]}"; do
    run aws ec2 delete-key-pair --region "$REGION" --key-name "$k"
  done
fi

# IAM resources need client-side tag filtering (no native --filters support).
iam_tag_value() {
  local tags="$1"
  echo "$tags" | jq -r --arg v "$TAG_VALUE_FILTER" \
    '(map(select(.Key=="zeusd-test"))[0] // null) as $t
     | if $t == null then empty
       elif ($v == "" or $t.Value == $v) then $t.Value
       else empty end' 2>/dev/null
}

echo "== IAM instance profiles =="
PROFILE_TARGETS=()
while IFS= read -r row; do
  name=$(echo "$row" | jq -r '.InstanceProfileName')
  created=$(echo "$row" | jq -r '.CreateDate // empty')
  tags=$(aws iam list-instance-profile-tags --instance-profile-name "$name" --query Tags --output json 2>/dev/null || echo '[]')
  tv=$(iam_tag_value "$tags")
  [[ -z "$tv" ]] && continue
  if is_too_new "$created"; then
    echo "  $name  tag=$tv  created=$created  [skip: newer than $OLDER_THAN]"
    continue
  fi
  echo "  $name  tag=$tv  created=$created"
  PROFILE_TARGETS+=("$name")
  mark_targeted "$tv"
done < <(aws iam list-instance-profiles --query 'InstanceProfiles[]' --output json | jq -c '.[]')
if [[ ${#PROFILE_TARGETS[@]} -eq 0 ]]; then
  echo "  none to act on"
else
  for p in "${PROFILE_TARGETS[@]}"; do
    while IFS= read -r r; do
      [[ -z "$r" ]] && continue
      run aws iam remove-role-from-instance-profile --instance-profile-name "$p" --role-name "$r"
    done < <(aws iam get-instance-profile --instance-profile-name "$p" --query 'InstanceProfile.Roles[].RoleName' --output text 2>/dev/null | tr '\t' '\n')
    run aws iam delete-instance-profile --instance-profile-name "$p"
  done
fi

echo "== IAM roles =="
ROLE_TARGETS=()
while IFS= read -r row; do
  name=$(echo "$row" | jq -r '.RoleName')
  created=$(echo "$row" | jq -r '.CreateDate // empty')
  tags=$(aws iam list-role-tags --role-name "$name" --query Tags --output json 2>/dev/null || echo '[]')
  tv=$(iam_tag_value "$tags")
  [[ -z "$tv" ]] && continue
  if is_too_new "$created"; then
    echo "  $name  tag=$tv  created=$created  [skip: newer than $OLDER_THAN]"
    continue
  fi
  echo "  $name  tag=$tv  created=$created"
  ROLE_TARGETS+=("$name")
  mark_targeted "$tv"
done < <(aws iam list-roles --query 'Roles[]' --output json | jq -c '.[]')
if [[ ${#ROLE_TARGETS[@]} -eq 0 ]]; then
  echo "  none to act on"
else
  for r in "${ROLE_TARGETS[@]}"; do
    while IFS= read -r arn; do
      [[ -z "$arn" ]] && continue
      run aws iam detach-role-policy --role-name "$r" --policy-arn "$arn"
    done < <(aws iam list-attached-role-policies --role-name "$r" --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null | tr '\t' '\n')
    run aws iam delete-role --role-name "$r"
  done
fi

if [[ "$ACTION" == "list" ]]; then
  echo
  echo "(Listing only. Pass --delete with --tag-value, --older-than, or --all to actually remove.)"
fi
echo "Done."

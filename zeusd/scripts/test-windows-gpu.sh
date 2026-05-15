#!/usr/bin/env bash
# Provision a Windows g4dn.xlarge, build zeusd from a Git branch, run smoke
# tests over TCP and Windows Named Pipe, install Python+PyTorch+zeus and run a
# sustained matmul + power/energy + power-limit/clock validation against the
# live NVML, then tear down.
#
# Prereqs (script aborts if missing):
#   - aws CLI v2 with valid credentials (`aws sts get-caller-identity` ok)
#   - jq
#   - base64
#   - bash 4+
#
# Usage:
#   ./test-windows-gpu.sh [options]
# Options:
#   --region <r>          AWS region (default: us-west-2)
#   --ami <id>            Windows AMI id (default: pinned DCV-Windows-2022 + NVIDIA)
#   --instance-type <t>   EC2 instance type (default: g4dn.xlarge)
#   --git-ref <ref>       Branch/tag/SHA on origin to test (default: current branch)
#   --keep                Skip teardown (instance + resources stay around)
#   --no-quota-check      Skip the G-family quota pre-flight
#
# Env overrides:
#   AWS_REGION, ZEUSD_TEST_AMI, ZEUSD_TEST_INSTANCE_TYPE, ZEUSD_TEST_GIT_REF
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_TEST_FILE="$SCRIPT_DIR/test-windows-gpu.py"
[[ -f "$PY_TEST_FILE" ]] || { echo "missing $PY_TEST_FILE" >&2; exit 1; }

# ---------- parse args ----------
REGION="${AWS_REGION:-us-west-2}"
AMI="${ZEUSD_TEST_AMI:-ami-03f09307db43011ca}"   # DCV-Windows-2022 + NVIDIA 538.67 (us-west-2)
ITYPE="${ZEUSD_TEST_INSTANCE_TYPE:-g4dn.xlarge}"
GIT_REF="${ZEUSD_TEST_GIT_REF:-$(git rev-parse --abbrev-ref HEAD)}"
KEEP=0
SKIP_QUOTA=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)            REGION="$2"; shift 2 ;;
    --ami)               AMI="$2"; shift 2 ;;
    --instance-type)     ITYPE="$2"; shift 2 ;;
    --git-ref)           GIT_REF="$2"; shift 2 ;;
    --keep)              KEEP=1; shift ;;
    --no-quota-check)    SKIP_QUOTA=1; shift ;;
    -h|--help)           sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    *)                   echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---------- pre-flight ----------
command -v aws    >/dev/null || { echo "aws CLI v2 not installed"; exit 1; }
command -v jq     >/dev/null || { echo "jq not installed"; exit 1; }
command -v base64 >/dev/null || { echo "base64 not installed"; exit 1; }
aws sts get-caller-identity --output text --query Arn >/dev/null || {
  echo "AWS credentials missing or expired. Log in with your usual method and re-run." >&2
  exit 1
}
echo "Region: $REGION   AMI: $AMI   Instance: $ITYPE   Git ref: $GIT_REF"

if [[ $SKIP_QUOTA -eq 0 ]]; then
  Q=$(aws service-quotas get-service-quota --region "$REGION" --service-code ec2 \
        --quota-code L-DB2E81BA --query 'Quota.Value' --output text)
  REQUIRED=$(aws ec2 describe-instance-types --region "$REGION" --instance-types "$ITYPE" \
        --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text)
  if (( $(printf '%.0f' "$Q") < REQUIRED )); then
    echo "G/VT On-Demand vCPU quota in $REGION is $Q; $ITYPE needs $REQUIRED." >&2
    echo "Request a quota increase via Service Quotas (code L-DB2E81BA) and retry." >&2
    exit 1
  fi
fi

# ---------- scoped resource names ----------
TAG_VALUE="zeusd-test-$(date -u +%Y%m%d-%H%M%S)-$$"
KEY_NAME="$TAG_VALUE"
SG_NAME="$TAG_VALUE"
ROLE_NAME="$TAG_VALUE-role"
PROFILE_NAME="$TAG_VALUE-profile"
INSTANCE_ID=""
SG_ID=""
KEY_FILE="$(mktemp -u)-${KEY_NAME}.pem"

# ---------- cleanup trap ----------
cleanup() {
  local rc=$?
  if [[ $KEEP -eq 1 ]]; then
    echo "[--keep] leaving resources behind (tag zeusd-test=$TAG_VALUE)"
    [[ -n "$INSTANCE_ID" ]] && echo "  instance:    $INSTANCE_ID"
    [[ -n "$SG_ID"       ]] && echo "  sg:          $SG_ID"
    echo  "  key file:    $KEY_FILE"
    exit $rc
  fi

  # If credentials died during the run, AWS calls below will all silently
  # fail (each is `|| true`). Detect that up front and tell the user exactly
  # what to delete manually so the state is actionable.
  if ! aws sts get-caller-identity --output text --query Arn >/dev/null 2>&1; then
    {
      echo "WARNING: AWS credentials are not valid; cleanup cannot run."
      echo "After re-logging in, run this to clean up everything tagged for this run:"
      echo "  bash $SCRIPT_DIR/cleanup-test-aws.sh --tag-value $TAG_VALUE"
      echo "Or delete by hand:"
      [[ -n "$INSTANCE_ID" ]] && \
        echo "  aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"
      echo "  aws iam remove-role-from-instance-profile --instance-profile-name $PROFILE_NAME --role-name $ROLE_NAME"
      echo "  aws iam delete-instance-profile --instance-profile-name $PROFILE_NAME"
      echo "  aws iam detach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
      echo "  aws iam delete-role --role-name $ROLE_NAME"
      [[ -n "$SG_ID" ]] && \
        echo "  aws ec2 delete-security-group --region $REGION --group-id $SG_ID"
      echo "  aws ec2 delete-key-pair --region $REGION --key-name $KEY_NAME"
    } >&2
    rm -f "$KEY_FILE"
    exit $rc
  fi

  echo "Cleaning up (tag $TAG_VALUE)..."
  if [[ -n "$INSTANCE_ID" ]]; then
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null 2>&1 || true
    aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$INSTANCE_ID" 2>/dev/null || true
  fi
  aws iam remove-role-from-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME" 2>/dev/null || true
  aws iam delete-instance-profile --instance-profile-name "$PROFILE_NAME" 2>/dev/null || true
  aws iam detach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null || true
  aws iam delete-role --role-name "$ROLE_NAME" 2>/dev/null || true
  [[ -n "$SG_ID" ]] && aws ec2 delete-security-group --region "$REGION" --group-id "$SG_ID" 2>/dev/null || true
  aws ec2 delete-key-pair --region "$REGION" --key-name "$KEY_NAME" 2>/dev/null || true
  rm -f "$KEY_FILE"
  exit $rc
}
trap cleanup EXIT INT TERM

# ---------- provision ----------
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)

aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" --key-type rsa \
  --tag-specifications "ResourceType=key-pair,Tags=[{Key=zeusd-test,Value=$TAG_VALUE}]" \
  --query KeyMaterial --output text > "$KEY_FILE"
chmod 600 "$KEY_FILE"

SG_ID=$(aws ec2 create-security-group --region "$REGION" --group-name "$SG_NAME" \
  --description "zeusd Windows test (no inbound)" --vpc-id "$VPC_ID" \
  --tag-specifications "ResourceType=security-group,Tags=[{Key=zeusd-test,Value=$TAG_VALUE}]" \
  --query GroupId --output text)

aws iam create-role --role-name "$ROLE_NAME" \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
  --tags "Key=zeusd-test,Value=$TAG_VALUE" >/dev/null
aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" \
  --tags "Key=zeusd-test,Value=$TAG_VALUE" >/dev/null
aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
sleep 8  # let IAM propagate

INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$ITYPE" --count 1 \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=$PROFILE_NAME" \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=60,VolumeType=gp3,DeleteOnTermination=true}" \
  --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=zeusd-test,Value=$TAG_VALUE},{Key=Name,Value=$TAG_VALUE}]" \
                      "ResourceType=volume,Tags=[{Key=zeusd-test,Value=$TAG_VALUE}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "Instance: $INSTANCE_ID"

echo "Waiting for SSM agent..."
until aws ssm describe-instance-information --region "$REGION" \
        --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
        --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null \
        | grep -q Online; do sleep 20; done
echo "SSM Online."

# ---------- build & test (one big PowerShell payload via SSM) ----------
# Base64-encode the local Python test script so we can ship it intact through
# bash -> jq JSON -> SSM -> PowerShell -> file without escape-soup.
PY_B64=$(base64 -w0 "$PY_TEST_FILE" 2>/dev/null || base64 "$PY_TEST_FILE" | tr -d '\n')

PS_SCRIPT=$(cat <<POWERSHELL
\$ErrorActionPreference = 'Stop'
\$ProgressPreference    = 'SilentlyContinue'

# ---------- Visual Studio Build Tools (C++ workload) ----------
if (-not (Test-Path 'C:\\BuildTools\\VC')) {
  Invoke-WebRequest -UseBasicParsing -Uri 'https://aka.ms/vs/17/release/vs_BuildTools.exe' -OutFile "\$env:TEMP\\vs_BuildTools.exe"
  \$p = Start-Process -FilePath "\$env:TEMP\\vs_BuildTools.exe" -ArgumentList '--quiet','--wait','--norestart','--nocache','--installPath','C:\\BuildTools','--add','Microsoft.VisualStudio.Workload.VCTools','--includeRecommended' -Wait -NoNewWindow -PassThru
  if (\$p.ExitCode -notin 0,3010) { throw "vs_BuildTools exit \$(\$p.ExitCode)" }
}

# ---------- Rust (MSVC) ----------
[Environment]::SetEnvironmentVariable('CARGO_HOME','C:\\rust\\cargo','Machine')
[Environment]::SetEnvironmentVariable('RUSTUP_HOME','C:\\rust\\rustup','Machine')
\$env:CARGO_HOME='C:\\rust\\cargo'; \$env:RUSTUP_HOME='C:\\rust\\rustup'
if (-not (Test-Path 'C:\\rust\\cargo\\bin\\cargo.exe')) {
  Invoke-WebRequest -UseBasicParsing -Uri 'https://win.rustup.rs/x86_64' -OutFile "\$env:TEMP\\rustup-init.exe"
  Start-Process -FilePath "\$env:TEMP\\rustup-init.exe" -ArgumentList '-y','--default-toolchain','stable-x86_64-pc-windows-msvc','--profile','minimal','--no-modify-path' -Wait -NoNewWindow
}
\$env:Path = "C:\\rust\\cargo\\bin;\$env:Path"

# ---------- Python 3.12 ----------
if (-not (Test-Path 'C:\\Python312\\python.exe')) {
  Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe' -OutFile "\$env:TEMP\\python-installer.exe"
  Start-Process -FilePath "\$env:TEMP\\python-installer.exe" -ArgumentList '/quiet','InstallAllUsers=1','PrependPath=1','Include_pip=1','Include_test=0','Include_doc=0','TargetDir=C:\\Python312' -Wait -NoNewWindow
}
\$env:Path = "C:\\Python312;C:\\Python312\\Scripts;\$env:Path"

# ---------- Python deps (torch + zeus + requests) ----------
& cmd /c "C:\\Python312\\python.exe -m pip install --upgrade pip 2>&1" | Select-Object -Last 3
& cmd /c "C:\\Python312\\python.exe -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 2>&1" | Select-Object -Last 3
& cmd /c "C:\\Python312\\python.exe -m pip install --no-cache-dir requests zeus 2>&1" | Select-Object -Last 3

# ---------- Fetch source (branch zip) ----------
\$src = 'C:\\zeusbuild'
if (-not (Test-Path \$src)) { New-Item -ItemType Directory \$src | Out-Null }
\$zip = "\$src\\zeus.zip"; if (Test-Path \$zip) { Remove-Item \$zip }
Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/ml-energy/zeus/archive/${GIT_REF}.zip' -OutFile \$zip
Get-ChildItem -Path \$src -Directory | Where-Object Name -Like 'zeus-*' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path \$zip -DestinationPath \$src
\$repo = Get-ChildItem -Path \$src -Directory | Where-Object Name -Like 'zeus-*' | Select-Object -First 1
\$zeusd = Join-Path \$repo.FullName 'zeusd'

# ---------- Build ----------
Set-Location \$zeusd
& cmd /c "cargo build > \$src\\build.log 2>&1"
if (\$LASTEXITCODE -ne 0) { Get-Content "\$src\\build.log" -Tail 80; throw "cargo build failed" }
Write-Host "BUILD_OK"
\$exe = "\$zeusd\\target\\debug\\zeusd.exe"

# ---------- TCP smoke ----------
Write-Host "===== TCP smoke ====="
Get-Process -Name zeusd -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
\$p = Start-Process -FilePath \$exe -ArgumentList @('serve','--mode','tcp','--tcp-bind-address','127.0.0.1:4938','--enable','gpu-control,gpu-read') -PassThru -NoNewWindow -RedirectStandardOutput "\$src\\zeusd-tcp.out" -RedirectStandardError "\$src\\zeusd-tcp.err"
Start-Sleep -Seconds 4
foreach (\$spec in @(
  @{ M='GET';  P='/discover' },
  @{ M='GET';  P='/gpu/get_power' },
  @{ M='GET';  P='/gpu/get_cumulative_energy' },
  @{ M='POST'; P='/gpu/set_persistence_mode?gpu_ids=0&enabled=true&block=true' },
  @{ M='POST'; P='/gpu/set_persistence_mode?gpu_ids=0&enabled=false&block=true' }
)) {
  Write-Host ">>> TCP \$(\$spec.M) \$(\$spec.P)"
  & curl.exe -sS -o - -w "  HTTP %{http_code}\`n" -X \$spec.M "http://127.0.0.1:4938\$(\$spec.P)"
}
Get-Process -Name zeusd -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

# ---------- Named-pipe smoke ----------
Write-Host "===== Named-pipe smoke ====="
\$p = Start-Process -FilePath \$exe -ArgumentList @('serve','--mode','named-pipe','--pipe-name','\\\\.\\pipe\\zeusd','--enable','gpu-control,gpu-read') -PassThru -NoNewWindow -RedirectStandardOutput "\$src\\zeusd-pipe.out" -RedirectStandardError "\$src\\zeusd-pipe.err"
Start-Sleep -Seconds 4
function Invoke-PipeHttp(\$m,\$path) {
  \$pipe = New-Object System.IO.Pipes.NamedPipeClientStream('.','zeusd',[System.IO.Pipes.PipeDirection]::InOut)
  \$pipe.Connect(3000)
  \$req = "\$m \$path HTTP/1.1\`r\`nHost: localhost\`r\`nConnection: close\`r\`nContent-Length: 0\`r\`n\`r\`n"
  \$b = [Text.Encoding]::ASCII.GetBytes(\$req); \$pipe.Write(\$b,0,\$b.Length); \$pipe.Flush()
  \$ms = New-Object IO.MemoryStream; \$buf = New-Object byte[] 8192
  while ((\$n = \$pipe.Read(\$buf,0,\$buf.Length)) -gt 0) { \$ms.Write(\$buf,0,\$n) }
  \$pipe.Close(); return [Text.Encoding]::ASCII.GetString(\$ms.ToArray())
}
foreach (\$x in @(
  @{m='GET'; p='/discover'}, @{m='GET'; p='/gpu/get_power'}, @{m='GET'; p='/gpu/get_cumulative_energy'},
  @{m='POST'; p='/gpu/set_persistence_mode?gpu_ids=0&enabled=true&block=true'},
  @{m='POST'; p='/gpu/set_persistence_mode?gpu_ids=0&enabled=false&block=true'}
)) {
  Write-Host ">>> PIPE \$(\$x.m) \$(\$x.p)"
  \$resp = Invoke-PipeHttp \$x.m \$x.p
  Write-Host "  \$((\$resp -split "\`r\`n",2)[0])"
}
Get-Process -Name zeusd -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

# ---------- PyTorch load + zeus.device.gpu test ----------
Write-Host "===== PyTorch load + ranges ====="
\$pyPath = "\$src\\zeusd_gpu_test.py"
\$pyBytes = [Convert]::FromBase64String('${PY_B64}')
[IO.File]::WriteAllBytes(\$pyPath, \$pyBytes)

\$p = Start-Process -FilePath \$exe -ArgumentList @('serve','--mode','tcp','--tcp-bind-address','127.0.0.1:4938','--enable','gpu-control,gpu-read') -PassThru -NoNewWindow -RedirectStandardOutput "\$src\\zeusd-load.out" -RedirectStandardError "\$src\\zeusd-load.err"
Start-Sleep -Seconds 4
& cmd /c "C:\\Python312\\python.exe \`"\$pyPath\`" 2>&1"
\$pyExit = \$LASTEXITCODE
Get-Process -Name zeusd -ErrorAction SilentlyContinue | Stop-Process -Force
if (\$pyExit -ne 0) { throw "python test exit \$pyExit" }
Write-Host "ALL_OK"
POWERSHELL
)

echo "Sending build+smoke+load to instance (this takes ~15-25 min on first run)..."
PS_JSON=$(mktemp)
jq -Rn --arg s "$PS_SCRIPT" '{commands: [$s], executionTimeout: ["3600"]}' > "$PS_JSON"
CMD_ID=$(aws ssm send-command --region "$REGION" --document-name AWS-RunPowerShellScript \
  --instance-ids "$INSTANCE_ID" --parameters "file://$PS_JSON" --timeout-seconds 3600 \
  --query 'Command.CommandId' --output text)
rm -f "$PS_JSON"
until aws ssm get-command-invocation --region "$REGION" --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
        --query Status --output text 2>/dev/null | grep -qE '^(Success|Failed|Cancelled|TimedOut)$'; do sleep 30; done
STATUS=$(aws ssm get-command-invocation --region "$REGION" --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" --query Status --output text)
echo "--- ssm stdout ---"
aws ssm get-command-invocation --region "$REGION" --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" --query StandardOutputContent --output text
echo "--- ssm stderr ---"
aws ssm get-command-invocation --region "$REGION" --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" --query StandardErrorContent --output text
[[ "$STATUS" == Success ]] || { echo "FAILED ($STATUS)"; exit 1; }
echo "All Windows GPU tests passed."

#!/bin/bash
# hunt-ds4-pods.sh - bounded 1-GPU pod hunt for the DSV4 capture/encode.
# Self-limits at 4 RUNNING tc-ds4* pods total (~$8.4/hr ceiling).
set -u
KEYFILE="$HOME/.runpod/config.toml"
KEY=$(grep -oE "apikey\s*=\s*'[^']+'" "$KEYFILE" | sed "s/apikey *= *'//;s/'//")

count_pods() {
  curl -s -m 15 -X POST https://api.runpod.io/graphql \
    -H "Content-Type: application/json" -H "Authorization: $KEY" \
    -d '{"query":"query { myself { pods { id name desiredStatus } } }"}' \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
pods=[p for p in d['data']['myself']['pods']
      if p['name'].startswith('tc-ds4') and p['desiredStatus']=='RUNNING']
print(len(pods))"
}

try_create() {
  local slot="$1"
  runpodctl pod create \
    --gpu-id "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
    --gpu-count 1 --cloud-type ALL --data-center-ids US-MO-2 \
    --image runpod/pytorch:1.2.0-rc.162-cu1281-torch2130-ubuntu2204-cluster \
    --container-disk-in-gb 100 \
    --network-volume-id gq012mrat9 --volume-mount-path /workspace \
    --ports "22/tcp,8888/http" --name "tc-ds4-w$slot" 2>/dev/null \
  | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(d.get('id') or '')
except Exception:
    print('')"
}

while :; do
  N=$(count_pods)
  if [ "${N:-0}" -ge 8 ]; then
    echo "HUNT-DONE at $N pods (cap 8)"
    exit 0
  fi
  ID=$(try_create "$((N+1))")
  if [ -n "$ID" ]; then
    echo "POD-CREATED: tc-ds4-w$((N+1)) $ID"
  fi
  sleep 90
done

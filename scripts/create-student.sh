#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="$SCRIPT_DIR/../templates"

if [ -z "$1" ]; then
  echo "Usage: $0 <username>"
  echo "Example: $0 alice"
  exit 1
fi

USERNAME="$1"

echo "=== Creating namespace for student: $USERNAME ==="
cat "$TEMPLATES_DIR/student-namespace.yaml" | sed "s/\${USERNAME}/$USERNAME/g" | kubectl apply -f -

echo ""
echo "Created resources for student-$USERNAME:"
kubectl get ns student-$USERNAME
kubectl get pvc -n student-$USERNAME
kubectl get resourcequota -n student-$USERNAME

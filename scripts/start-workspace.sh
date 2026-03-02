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
NAMESPACE="student-$USERNAME"

# Check if namespace exists
if ! kubectl get ns "$NAMESPACE" &>/dev/null; then
  echo "Error: Namespace $NAMESPACE does not exist."
  echo "Run: ./create-student.sh $USERNAME"
  exit 1
fi

# Check if workspace already exists
if kubectl get pod workspace -n "$NAMESPACE" &>/dev/null; then
  echo "Workspace already exists for $USERNAME"
  kubectl get pod workspace -n "$NAMESPACE"
  exit 0
fi

echo "=== Starting workspace for $USERNAME ==="
cat "$TEMPLATES_DIR/workspace-pod.yaml" | sed "s/\${USERNAME}/$USERNAME/g" | kubectl apply -f -

echo "=== Waiting for workspace to be ready ==="
kubectl wait --for=condition=Ready pod/workspace -n "$NAMESPACE" --timeout=120s

echo ""
echo "Workspace started for $USERNAME!"
echo "Access with: kubectl exec -it -n $NAMESPACE workspace -- bash"

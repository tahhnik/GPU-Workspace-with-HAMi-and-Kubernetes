#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <username>"
  echo "Example: $0 alice"
  exit 1
fi

USERNAME="$1"
NAMESPACE="student-$USERNAME"

echo "=== Stopping workspace for $USERNAME ==="
kubectl delete pod workspace -n "$NAMESPACE" --ignore-not-found

echo "Workspace stopped. PVC data is preserved."

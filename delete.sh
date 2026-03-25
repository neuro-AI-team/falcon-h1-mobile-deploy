#!/bin/bash
# ===========================================================================
# Falcon H1 Mobile Models — Deletion Script
#
# Removes all Falcon H1 Mobile resources from the target GKE cluster.
#
# Usage:
#   ./delete.sh                                      # delete from default namespace (ai)
#   ./delete.sh --namespace my-ns                    # delete from custom namespace
#   ./delete.sh --context gke_proj_zone_cluster      # delete from specific cluster
#   ./delete.sh --delete-namespace                   # also delete the namespace itself
# ===========================================================================
set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

print_status()  { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ── Defaults ───────────────────────────────────────────────────────────────
NAMESPACE="ai"
CONTEXT=""
DELETE_NAMESPACE=false
FORCE=false
KUBECTL="kubectl"

# ── Argument parsing ─────────────────────────────────────────────────────
show_usage() {
    cat <<EOF
${BOLD}Falcon H1 Mobile Models — Deletion Script${NC}

Removes all Falcon H1 Mobile deployment resources from the cluster.

${BOLD}Usage:${NC}  $0 [OPTIONS]

${BOLD}Options:${NC}
  --context <ctx>        kubectl context              (default: current-context)
  --namespace <ns>       Kubernetes namespace          (default: ai)
  --delete-namespace     Also delete the namespace     (default: no)
  --force                Skip confirmation prompt
  --help                 Show this help

${BOLD}Resources that will be deleted:${NC}
  - RayService:  falcon-h1-mobile
  - Service:     falcon-h1-mobile-service
  - ConfigMap:   falcon-h1-mobile-serve-script
  - SA:          ray-sa (if --delete-namespace, removed with namespace)

${BOLD}Examples:${NC}
  $0                                          # delete from namespace 'ai'
  $0 --namespace staging --force              # delete from 'staging', no prompt
  $0 --context gke_proj_zone_cluster --force  # target specific cluster
  $0 --delete-namespace --force               # remove namespace too
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --context)          CONTEXT="$2";          shift 2 ;;
        --namespace)        NAMESPACE="$2";        shift 2 ;;
        --delete-namespace) DELETE_NAMESPACE=true;  shift ;;
        --force)            FORCE=true;            shift ;;
        --help)             show_usage; exit 0     ;;
        *) print_error "Unknown option: $1"; show_usage; exit 1 ;;
    esac
done

if [[ -n "$CONTEXT" ]]; then
    KUBECTL="kubectl --context=$CONTEXT"
fi

kctl() { $KUBECTL "$@"; }

# ── Pre-checks ────────────────────────────────────────────────────────────
if ! command -v kubectl &>/dev/null; then
    print_error "kubectl not found in PATH"
    exit 1
fi

if ! kctl cluster-info &>/dev/null; then
    print_error "Cannot reach cluster. Check your kubeconfig / --context."
    exit 1
fi

# ── Show what will be deleted ─────────────────────────────────────────────
echo ""
echo -e "${BOLD}=========================================${NC}"
echo -e "${BOLD}  Falcon H1 Mobile — Delete Resources${NC}"
echo -e "${BOLD}=========================================${NC}"
echo "  Namespace:  ${NAMESPACE}"
if [[ -n "$CONTEXT" ]]; then
    echo "  Context:    ${CONTEXT}"
fi
echo -e "${BOLD}=========================================${NC}"
echo ""

print_status "Checking existing resources in namespace '${NAMESPACE}' ..."
echo ""

FOUND=false

if kctl get rayservice falcon-h1-mobile -n "$NAMESPACE" &>/dev/null; then
    echo "  - RayService/falcon-h1-mobile"
    FOUND=true
fi
if kctl get svc falcon-h1-mobile-service -n "$NAMESPACE" &>/dev/null; then
    echo "  - Service/falcon-h1-mobile-service"
    FOUND=true
fi
if kctl get configmap falcon-h1-mobile-serve-script -n "$NAMESPACE" &>/dev/null; then
    echo "  - ConfigMap/falcon-h1-mobile-serve-script"
    FOUND=true
fi
if kctl get sa ray-sa -n "$NAMESPACE" &>/dev/null; then
    echo "  - ServiceAccount/ray-sa"
    FOUND=true
fi

# Check for Ray-created serve services
SERVE_SVCS=$(kctl get svc -n "$NAMESPACE" -o name 2>/dev/null | grep "falcon-h1-mobile" | grep -v "falcon-h1-mobile-service" || true)
if [[ -n "$SERVE_SVCS" ]]; then
    while IFS= read -r svc; do
        echo "  - ${svc}"
    done <<< "$SERVE_SVCS"
    FOUND=true
fi

if [[ "$DELETE_NAMESPACE" == true ]]; then
    echo "  - Namespace/${NAMESPACE} (and all remaining resources in it)"
fi

if [[ "$FOUND" == false && "$DELETE_NAMESPACE" == false ]]; then
    echo ""
    print_warning "No Falcon H1 Mobile resources found in namespace '${NAMESPACE}'"
    exit 0
fi

echo ""

# ── Confirmation ──────────────────────────────────────────────────────────
if [[ "$FORCE" != true ]]; then
    echo -e "${YELLOW}Are you sure you want to delete these resources? [y/N]${NC} "
    read -r confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        print_status "Aborted."
        exit 0
    fi
    echo ""
fi

# ── Delete resources ──────────────────────────────────────────────────────

# 1. RayService (this triggers pod cleanup, so do it first)
if kctl get rayservice falcon-h1-mobile -n "$NAMESPACE" &>/dev/null; then
    print_status "Deleting RayService falcon-h1-mobile ..."
    kctl delete rayservice falcon-h1-mobile -n "$NAMESPACE" --wait=true --timeout=120s
    print_success "RayService deleted"
else
    print_warning "RayService falcon-h1-mobile not found — skipping"
fi

# 2. Wait for pods to terminate
print_status "Waiting for pods to terminate ..."
WAIT_ELAPSED=0
while [[ $WAIT_ELAPSED -lt 60 ]]; do
    REMAINING=$(kctl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | grep -c "falcon-h1-mobile" || true)
    if [[ "$REMAINING" -eq 0 ]]; then
        break
    fi
    sleep 5
    WAIT_ELAPSED=$((WAIT_ELAPSED + 5))
done

REMAINING=$(kctl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | grep -c "falcon-h1-mobile" || true)
if [[ "$REMAINING" -gt 0 ]]; then
    print_warning "${REMAINING} pods still terminating — continuing cleanup"
else
    print_success "All pods terminated"
fi

# 3. Service
if kctl get svc falcon-h1-mobile-service -n "$NAMESPACE" &>/dev/null; then
    print_status "Deleting Service falcon-h1-mobile-service ..."
    kctl delete svc falcon-h1-mobile-service -n "$NAMESPACE" --ignore-not-found=true
    print_success "Service deleted"
fi

# 4. ConfigMap
if kctl get configmap falcon-h1-mobile-serve-script -n "$NAMESPACE" &>/dev/null; then
    print_status "Deleting ConfigMap falcon-h1-mobile-serve-script ..."
    kctl delete configmap falcon-h1-mobile-serve-script -n "$NAMESPACE" --ignore-not-found=true
    print_success "ConfigMap deleted"
fi

# 5. ServiceAccount
if kctl get sa ray-sa -n "$NAMESPACE" &>/dev/null; then
    print_status "Deleting ServiceAccount ray-sa ..."
    kctl delete sa ray-sa -n "$NAMESPACE" --ignore-not-found=true
    print_success "ServiceAccount deleted"
fi

# 6. Namespace (optional)
if [[ "$DELETE_NAMESPACE" == true ]]; then
    print_status "Deleting namespace '${NAMESPACE}' ..."
    kctl delete namespace "$NAMESPACE" --wait=true --timeout=120s
    print_success "Namespace '${NAMESPACE}' deleted"
fi

echo ""
print_success "All Falcon H1 Mobile resources have been removed."
echo ""

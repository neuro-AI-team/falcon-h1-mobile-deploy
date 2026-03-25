#!/bin/bash
# ===========================================================================
# Falcon H1 Mobile Models — Standalone Deployment Script
#
# Deploys 15 Falcon H1 GGUF models via llama-cpp-python + Ray Serve
# onto any GKE cluster with sufficient GPU resources.
#
# Usage:
#   ./deploy.sh --wait                                     # deploy to current cluster
#   ./deploy.sh --context gke_proj_zone_cluster --wait     # deploy to specific cluster
#   ./deploy.sh --gpu-pool my-gpu-pool --head-pool default # custom node pools
#   ./deploy.sh --preflight-only                           # just check GPU availability
#   ./deploy.sh --status                                   # show deployment status
#   ./deploy.sh --delete                                   # remove everything
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="ai"
HEAD_POOL="app-pool"
GPU_POOL="l4-pool"
GPU_TOLERATION="l4"
REQUIRED_GPUS=1         # number of GPU workers (1 GPU holds all 15 models)
MIN_VRAM_GB=12          # minimum VRAM per GPU in GiB (all 15 models ≈ 10 GiB + headroom)
CPU_FALLBACK=false      # fall back to CPU inference when no GPU meets VRAM requirement
CONTEXT=""
WAIT=false
TIMEOUT=900
DRY_RUN=false

# ── Computed ───────────────────────────────────────────────────────────────
KUBECTL="kubectl"   # will be overridden if --context is set

kctl() { $KUBECTL "$@"; }

# ── Argument parsing ──────────────────────────────────────────────────────
ACTION="deploy"

show_usage() {
    cat <<EOF
${BOLD}Falcon H1 Mobile Models — Standalone Deployment${NC}

Deploys 15 Falcon H1 GGUF models on Ray Serve + llama-cpp-python.
Requires 1 GPU worker with ≥${MIN_VRAM_GB} GiB VRAM (all 15 models fit on a single GPU).
Falls back to CPU inference if no qualifying GPU is found and --cpu-fallback is set.

${BOLD}Usage:${NC}  $0 [ACTION] [OPTIONS]

${BOLD}Actions (pick one):${NC}
  deploy (default)     Deploy the full stack
  --preflight-only     Run VRAM check only, don't deploy
  --status             Show current deployment status
  --delete             Remove all deployed resources
  --dry-run            Validate manifests without applying

${BOLD}Options:${NC}
  --context <ctx>      kubectl context              (default: current-context)
  --namespace <ns>     Kubernetes namespace          (default: ai)
  --head-pool <name>   Node pool for head pod        (default: app-pool)
  --gpu-pool <name>    Node pool for GPU workers     (default: l4-pool)
  --gpu-toleration <v> GPU taint toleration value    (default: l4)
  --required-gpus <n>  Number of GPU workers         (default: 1)
  --min-vram-gb <n>    Min VRAM per GPU in GiB       (default: 12)
  --cpu-fallback       Deploy on CPU if VRAM check fails (slower inference)
  --skip-preflight     Skip GPU/VRAM availability check
  --wait               Wait for service to be ready
  --timeout <sec>      Wait timeout                  (default: 900)
  --help               Show this help

${BOLD}VRAM requirements:${NC}
  All 15 models loaded simultaneously:  ~10 GiB
  Recommended minimum (with headroom):  12 GiB  (e.g. T4 16 GiB, L4 24 GiB)
  3× L4 setup is ~6× more than needed — 1× L4 or 1× T4 is sufficient.

${BOLD}Examples:${NC}
  $0 --wait
  $0 --context gke_myproj_us-central1_mycluster --gpu-pool gpu-pool --head-pool default-pool --wait
  $0 --preflight-only
  $0 --cpu-fallback --wait          # CPU inference, no GPU required
  $0 --delete --namespace ai
EOF
}

SKIP_PREFLIGHT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --preflight-only) ACTION="preflight"; shift ;;
        --status)         ACTION="status";    shift ;;
        --delete)         ACTION="delete";    shift ;;
        --dry-run)        DRY_RUN=true;       shift ;;
        --context)        CONTEXT="$2";       shift 2 ;;
        --namespace)      NAMESPACE="$2";     shift 2 ;;
        --head-pool)      HEAD_POOL="$2";     shift 2 ;;
        --gpu-pool)       GPU_POOL="$2";      shift 2 ;;
        --gpu-toleration) GPU_TOLERATION="$2"; shift 2 ;;
        --required-gpus)  REQUIRED_GPUS="$2"; shift 2 ;;
        --min-vram-gb)    MIN_VRAM_GB="$2";   shift 2 ;;
        --cpu-fallback)   CPU_FALLBACK=true;  shift ;;
        --skip-preflight) SKIP_PREFLIGHT=true; shift ;;
        --wait)           WAIT=true;          shift ;;
        --timeout)        TIMEOUT="$2";       shift 2 ;;
        --help)           show_usage; exit 0  ;;
        *) print_error "Unknown option: $1"; show_usage; exit 1 ;;
    esac
done

if [[ -n "$CONTEXT" ]]; then
    KUBECTL="kubectl --context=$CONTEXT"
fi

# ── Helpers ────────────────────────────────────────────────────────────────

apply_manifest() {
    local file="$1"

    # GPU vs CPU mode template variables
    if [[ "$CPU_FALLBACK" == true ]]; then
        local gpu_limit_expr="/^__GPU_LIMIT__$/d"
        local gpu_request_expr="/^__GPU_REQUEST__$/d"
        local ray_num_gpus="0"
        local falcon_use_gpu="0"
        local worker_node_pool="$HEAD_POOL"
    else
        local gpu_limit_expr="s|^__GPU_LIMIT__$|                nvidia.com/gpu: 1|"
        local gpu_request_expr="s|^__GPU_REQUEST__$|                nvidia.com/gpu: 1|"
        local ray_num_gpus="1"
        local falcon_use_gpu="1"
        local worker_node_pool="$GPU_POOL"
    fi

    sed \
        -e "s|__NAMESPACE__|${NAMESPACE}|g" \
        -e "s|__HEAD_NODE_POOL__|${HEAD_POOL}|g" \
        -e "s|__GPU_NODE_POOL__|${GPU_POOL}|g" \
        -e "s|__WORKER_NODE_POOL__|${worker_node_pool}|g" \
        -e "s|__GPU_TOLERATION__|${GPU_TOLERATION}|g" \
        -e "s|__NUM_GPU_WORKERS__|${REQUIRED_GPUS}|g" \
        -e "s|__RAY_NUM_GPUS__|${ray_num_gpus}|g" \
        -e "s|__FALCON_USE_GPU__|${falcon_use_gpu}|g" \
        -e "$gpu_limit_expr" \
        -e "$gpu_request_expr" \
        "$file" \
    | if [[ "$DRY_RUN" == true ]]; then
        kctl apply --dry-run=server -f -
    else
        kctl apply -f -
    fi
}

# ── Pre-flight checks ────────────────────────────────────────────────────

check_prerequisites() {
    if ! command -v kubectl &>/dev/null; then
        print_error "kubectl not found in PATH"
        exit 1
    fi

    if ! kctl cluster-info &>/dev/null; then
        print_error "Cannot reach cluster. Check your kubeconfig / --context."
        exit 1
    fi

    if ! kctl get crd rayservices.ray.io &>/dev/null; then
        print_error "KubeRay operator not installed (CRD rayservices.ray.io missing)."
        echo "  Install it first:  helm install kuberay-operator kuberay/kuberay-operator --version 1.3.0"
        exit 1
    fi

    print_success "Prerequisites OK (kubectl, cluster reachable, KubeRay CRD present)"
}

gpu_preflight_check() {
    local min_vram_mib=$(( MIN_VRAM_GB * 1024 ))
    print_status "Running GPU VRAM pre-flight check (need ≥${MIN_VRAM_GB} GiB VRAM on at least one node) ..."

    # Query GPU nodes: resolve VRAM via NVIDIA GPU Feature Discovery labels, then GKE
    # accelerator labels, then fall back to counting GPUs without VRAM info.
    local result
    result=$(kctl get nodes -o json | python3 -c "
import sys, json

# Known VRAM (MiB) by GKE accelerator label value
KNOWN_VRAM = {
    'nvidia-l4':            24576,
    'nvidia-tesla-l4':      24576,
    'nvidia-tesla-t4':      16384,
    'nvidia-tesla-v100':    16384,
    'nvidia-tesla-p100':    16384,
    'nvidia-tesla-p4':       7680,
    'nvidia-a100-80gb':     81920,
    'nvidia-a100':          40960,
    'nvidia-a10':           24576,
    'nvidia-a10g':          24576,
    'nvidia-h100-80gb':     81920,
    'nvidia-h100':          80960,
    'nvidia-h200':         141312,
}

data = json.load(sys.stdin)
for node in data['items']:
    name  = node['metadata']['name']
    labels = node['metadata'].get('labels', {})
    alloc  = int(node['status'].get('allocatable', {}).get('nvidia.com/gpu', '0'))
    if alloc == 0:
        continue

    # 1) NVIDIA GPU Feature Discovery label (most accurate)
    vram_mib = int(labels.get('nvidia.com/gpu.memory', '0'))

    # 2) GKE accelerator label -> lookup table
    if vram_mib == 0:
        accel = labels.get('cloud.google.com/gke-accelerator', '').lower()
        vram_mib = KNOWN_VRAM.get(accel, 0)

    print(f'{name}|{alloc}|{vram_mib}')
" 2>/dev/null) || true

    if [[ -z "$result" ]]; then
        print_error "No GPU nodes found on this cluster."
        return 1
    fi

    printf "\n  ${BOLD}%-50s %5s %10s %8s${NC}\n" "NODE" "GPUs" "VRAM(GiB)" "STATUS"
    printf "  %-50s %5s %10s %8s\n"               "----" "----" "---------" "------"

    local found_qualified=false

    while IFS='|' read -r node_name node_alloc node_vram_mib; do
        local status_str vram_display
        if [[ "$node_vram_mib" -eq 0 ]]; then
            vram_display="unknown"
            status_str="${YELLOW}UNKNOWN${NC}"
            # Can't verify VRAM — assume it qualifies and warn
            found_qualified=true
        else
            local vram_gib=$(( node_vram_mib / 1024 ))
            vram_display="${vram_gib} GiB"
            if [[ "$node_vram_mib" -ge "$min_vram_mib" ]]; then
                status_str="${GREEN}OK${NC}"
                found_qualified=true
            else
                status_str="${RED}LOW${NC}"
            fi
        fi
        printf "  %-50s %5d %10s " "$node_name" "$node_alloc" "$vram_display"
        echo -e "${status_str}"
    done <<< "$result"

    printf "  %-50s %5s %10s %8s\n\n" "----" "----" "---------" "------"

    if [[ "$found_qualified" == true ]]; then
        print_success "VRAM check passed: at least one GPU node has ≥${MIN_VRAM_GB} GiB VRAM"
        return 0
    else
        print_error "VRAM check failed: no GPU node has ≥${MIN_VRAM_GB} GiB VRAM"
        print_status "All 15 models require ~10 GiB VRAM loaded simultaneously."
        print_status "Use --min-vram-gb to adjust the threshold, or --cpu-fallback for CPU inference."
        return 1
    fi
}

# ── Deploy ────────────────────────────────────────────────────────────────

do_deploy() {
    print_status "Deploying to namespace '${NAMESPACE}' ..."
    echo ""

    # 1. Namespace
    print_status "Applying namespace ..."
    apply_manifest "${SCRIPT_DIR}/manifests/namespace.yaml"

    # 2. ServiceAccount
    print_status "Applying ServiceAccount ..."
    apply_manifest "${SCRIPT_DIR}/manifests/service-account.yaml"

    # 3. ConfigMap from serve script
    print_status "Creating serve-script ConfigMap ..."
    if [[ "$DRY_RUN" == true ]]; then
        kctl create configmap falcon-h1-mobile-serve-script \
            --from-file=mobile_llama_app.py="${SCRIPT_DIR}/serve/mobile_llama_app.py" \
            -n "$NAMESPACE" --dry-run=client -o yaml \
        | kctl apply --dry-run=server -f -
    else
        kctl create configmap falcon-h1-mobile-serve-script \
            --from-file=mobile_llama_app.py="${SCRIPT_DIR}/serve/mobile_llama_app.py" \
            -n "$NAMESPACE" --dry-run=client -o yaml \
        | kctl apply -f -
    fi

    # 4. Service
    print_status "Applying Service ..."
    apply_manifest "${SCRIPT_DIR}/manifests/falcon-h1-mobile-service.yaml"

    # 5. RayService (delete existing first to avoid stuck updates)
    if kctl get rayservice falcon-h1-mobile -n "$NAMESPACE" &>/dev/null; then
        if [[ "$DRY_RUN" != true ]]; then
            print_warning "RayService 'falcon-h1-mobile' exists. Deleting first ..."
            kctl delete rayservice falcon-h1-mobile -n "$NAMESPACE" --wait=true
            sleep 10
        fi
    fi

    print_status "Applying RayService ..."
    apply_manifest "${SCRIPT_DIR}/manifests/falcon-h1-mobile-rayservice.yaml"

    if [[ "$DRY_RUN" == true ]]; then
        echo ""
        print_success "Dry-run complete — all manifests validated."
        return
    fi

    print_success "All resources applied."

    if [[ "$WAIT" == true ]]; then
        wait_for_ready
    fi

    echo ""
    print_success "Deployment complete!"
    echo ""
    print_status "OpenWebUI connection URL:"
    echo "  http://falcon-h1-mobile-serve-svc.${NAMESPACE}.svc.cluster.local:8000/v1"
    echo ""
    print_status "Run '$0 --status' to check status"
    print_status "Run '$0 --delete' to tear down"
}

# ── Wait ──────────────────────────────────────────────────────────────────

wait_for_ready() {
    print_status "Waiting for RayService to become RUNNING (timeout: ${TIMEOUT}s) ..."
    local elapsed=0
    while [[ $elapsed -lt $TIMEOUT ]]; do
        local app_status
        app_status=$(kctl get rayservice falcon-h1-mobile -n "$NAMESPACE" \
            -o jsonpath='{.status.activeServiceStatus.applicationStatuses.falcon-h1-mobile-app.status}' 2>/dev/null || echo "")

        if [[ "$app_status" == "RUNNING" ]]; then
            print_success "RayService is RUNNING!"
            return 0
        fi

        if (( elapsed % 30 == 0 )); then
            print_status "Status: ${app_status:-pending} (${elapsed}s elapsed)"
        fi

        sleep 5
        elapsed=$((elapsed + 5))
    done

    print_error "Timed out after ${TIMEOUT}s"
    print_status "Debug with:  $0 --status"
    return 1
}

# ── Status ────────────────────────────────────────────────────────────────

do_status() {
    echo ""
    print_status "=== RayService ==="
    kctl get rayservice falcon-h1-mobile -n "$NAMESPACE" -o wide 2>/dev/null || echo "  Not found"

    echo ""
    print_status "=== Pods ==="
    kctl get pods -n "$NAMESPACE" 2>/dev/null | grep "falcon-h1-mobile" || echo "  No pods"

    echo ""
    print_status "=== Services ==="
    kctl get svc -n "$NAMESPACE" 2>/dev/null | grep "falcon-h1-mobile" || echo "  No services"

    echo ""
    print_status "=== Model Deployment Health ==="
    local statuses
    statuses=$(kctl get rayservice falcon-h1-mobile -n "$NAMESPACE" \
        -o json 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for section in ['activeServiceStatus', 'pendingServiceStatus']:
        app = d.get('status', {}).get(section, {}).get('applicationStatuses', {}).get('falcon-h1-mobile-app', {})
        if app:
            print(f'App status: {app.get(\"status\", \"?\")}'  )
            for name, s in sorted(app.get('serveDeploymentStatuses', {}).items()):
                print(f'  {name}: {s[\"status\"]}')
            break
except: pass
" 2>/dev/null) || true
    if [[ -n "$statuses" ]]; then
        echo "$statuses"
    else
        echo "  No deployment status available"
    fi

    # Test commands
    local serve_svc
    serve_svc=$(kctl get svc -n "$NAMESPACE" -o name 2>/dev/null \
        | grep "falcon-h1-mobile.*serve-svc" | head -1 | cut -d'/' -f2) || true
    if [[ -n "$serve_svc" ]]; then
        echo ""
        print_status "=== Test Commands ==="
        echo "  # Port-forward:"
        echo "  kubectl port-forward svc/${serve_svc} 8000:8000 -n ${NAMESPACE}"
        echo ""
        echo "  # List models:"
        echo "  curl -s http://localhost:8000/v1/models | python3 -m json.tool"
        echo ""
        echo "  # Chat completion:"
        echo '  curl -s http://localhost:8000/v1/chat/completions \'
        echo '    -H "Content-Type: application/json" \'
        echo '    -d '"'"'{"model":"h1-1.5b-q4","messages":[{"role":"user","content":"Hello!"}]}'"'"
        echo ""
        echo "  # OpenWebUI connection URL:"
        echo "  http://falcon-h1-mobile-serve-svc.${NAMESPACE}.svc.cluster.local:8000/v1"
    fi
}

# ── Delete ────────────────────────────────────────────────────────────────

do_delete() {
    print_status "Deleting Falcon H1 Mobile resources from namespace '${NAMESPACE}' ..."

    if kctl get rayservice falcon-h1-mobile -n "$NAMESPACE" &>/dev/null; then
        kctl delete rayservice falcon-h1-mobile -n "$NAMESPACE" --wait=true
        print_success "RayService deleted"
    else
        print_warning "RayService not found"
    fi

    sleep 5

    kctl delete svc falcon-h1-mobile-service -n "$NAMESPACE" --ignore-not-found=true
    kctl delete configmap falcon-h1-mobile-serve-script -n "$NAMESPACE" --ignore-not-found=true

    print_success "Cleanup complete (namespace '${NAMESPACE}' preserved)"
}

# ── Main ──────────────────────────────────────────────────────────────────

main() {
    echo ""
    echo -e "${BOLD}=========================================${NC}"
    echo -e "${BOLD}  Falcon H1 Mobile Model Deployment${NC}"
    echo -e "${BOLD}=========================================${NC}"
    echo "  Namespace:     ${NAMESPACE}"
    echo "  Head pool:     ${HEAD_POOL}"
    echo "  GPU pool:      ${GPU_POOL}"
    echo "  GPU workers:   ${REQUIRED_GPUS}"
    echo "  Min VRAM:      ${MIN_VRAM_GB} GiB  (~10 GiB for all 15 models)"
    if [[ "$CPU_FALLBACK" == true ]]; then
    echo "  Mode:          CPU fallback (no GPU required)"
    else
    echo "  Mode:          GPU  (FALCON_USE_GPU=1)"
    fi
    echo "  Models:        15 (Falcon H1 GGUF)"
    if [[ -n "$CONTEXT" ]]; then
        echo "  Context:       ${CONTEXT}"
    fi
    echo -e "${BOLD}=========================================${NC}"
    echo ""

    case "$ACTION" in
        status)
            do_status
            ;;
        delete)
            check_prerequisites
            do_delete
            ;;
        preflight)
            check_prerequisites
            if [[ "$CPU_FALLBACK" == true ]]; then
                print_warning "CPU fallback mode set — VRAM check is informational only."
            fi
            gpu_preflight_check
            ;;
        deploy)
            check_prerequisites

            if [[ "$CPU_FALLBACK" == true ]]; then
                print_warning "CPU fallback mode enabled — skipping VRAM check, deploying CPU-only workers."
                echo ""
            elif [[ "$SKIP_PREFLIGHT" != true && "$DRY_RUN" != true ]]; then
                if ! gpu_preflight_check; then
                    echo ""
                    print_error "Aborting deployment — no GPU node meets the ${MIN_VRAM_GB} GiB VRAM requirement."
                    print_status "Options:"
                    print_status "  --cpu-fallback          deploy with CPU inference (slower, no GPU required)"
                    print_status "  --min-vram-gb <n>       lower the VRAM threshold"
                    print_status "  --skip-preflight        skip the check entirely"
                    exit 1
                fi
                echo ""
            fi

            do_deploy
            ;;
    esac
}

main

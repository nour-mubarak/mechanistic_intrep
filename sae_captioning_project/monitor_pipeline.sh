#!/bin/bash

# Monitor Pipeline Progress
# Run with: bash monitor_pipeline.sh

echo "=== PIPELINE MONITORING DASHBOARD ==="
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "=== SAE Captioning Pipeline Status ==="
    echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check if pipeline is running
    if ps aux | grep -q "[p]ython scripts/run_full_pipeline.py"; then
        echo "✓ Pipeline is RUNNING"
    else
        echo "✗ Pipeline is NOT running"
    fi
    
    echo ""
    echo "=== Recent Log Lines ==="
    tail -15 pipeline_full_run.log
    
    echo ""
    echo "=== Pipeline Stages ==="
    grep -E "STAGE [0-9]:|completed in" pipeline_full_run.log | tail -10
    
    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "GPU monitoring not available"
    
    sleep 10
done

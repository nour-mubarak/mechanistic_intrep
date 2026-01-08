#!/bin/bash
# Pipeline Monitor with Email Notifications
# ==========================================
# Monitors job status and sends email notifications

EMAIL="jmsk62@durham.ac.uk"
PROJECT_DIR="/home2/jmsk62/mechanistic_intrep/mech_intrep/mechanistic_intrep/mechanistic_intrep/mechanistic_intrep/sae_captioning_project"
LOG_FILE="$PROJECT_DIR/logs/pipeline_monitor.log"
STATUS_FILE="$PROJECT_DIR/pipeline_status/monitor_state.txt"

# Initialize
mkdir -p "$PROJECT_DIR/pipeline_status"
touch "$STATUS_FILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

send_notification() {
    local subject="$1"
    local message="$2"
    echo "$message" | mail -s "$subject" "$EMAIL" 2>/dev/null || true
    log "Notification sent: $subject"
}

check_stage() {
    local stage_name="$1"
    local job_pattern="$2"
    
    # Get job states
    local running=$(squeue -u jmsk62 2>/dev/null | grep "$job_pattern" | grep " R " | wc -l)
    local pending=$(squeue -u jmsk62 2>/dev/null | grep "$job_pattern" | grep " PD " | wc -l)
    local total=$((running + pending))
    
    # Check for completed/failed jobs using sacct
    local completed=$(sacct -u jmsk62 --starttime=today -n 2>/dev/null | grep "$job_pattern" | grep "COMPLETED" | wc -l)
    local failed=$(sacct -u jmsk62 --starttime=today -n 2>/dev/null | grep "$job_pattern" | grep -E "FAILED|CANCELLED|OOM" | wc -l)
    
    echo "$stage_name:running=$running,pending=$pending,completed=$completed,failed=$failed"
}

main_loop() {
    log "Starting pipeline monitor..."
    
    local prev_state=""
    
    while true; do
        # Get current state
        local extract_state=$(check_stage "Extraction" "02_extra")
        local train_state=$(check_stage "SAE_Training" "03_train")
        local analysis_state=$(check_stage "Analysis" "09_compr\|11_feat\|13_feat\|14_visu\|15_feat\|16_cros\|17_qual")
        
        local current_state="$extract_state|$train_state|$analysis_state"
        
        # Check for state changes
        if [[ "$current_state" != "$prev_state" ]]; then
            log "State changed: $current_state"
            
            # Parse extraction state
            local ext_running=$(echo "$extract_state" | grep -oP 'running=\K\d+')
            local ext_pending=$(echo "$extract_state" | grep -oP 'pending=\K\d+')
            local ext_completed=$(echo "$extract_state" | grep -oP 'completed=\K\d+')
            local ext_failed=$(echo "$extract_state" | grep -oP 'failed=\K\d+')
            
            # Parse training state
            local train_running=$(echo "$train_state" | grep -oP 'running=\K\d+')
            local train_pending=$(echo "$train_state" | grep -oP 'pending=\K\d+')
            local train_completed=$(echo "$train_state" | grep -oP 'completed=\K\d+')
            local train_failed=$(echo "$train_state" | grep -oP 'failed=\K\d+')
            
            # Check for failures
            if [[ "$ext_failed" -gt 0 ]]; then
                send_notification "[PIPELINE ALERT] Extraction Jobs Failed" \
                    "WARNING: $ext_failed extraction job(s) have failed!\n\nCheck logs: tail -50 $PROJECT_DIR/logs/step2_*.err\n\nCurrent status:\n- Running: $ext_running\n- Pending: $ext_pending\n- Completed: $ext_completed\n- Failed: $ext_failed"
            fi
            
            if [[ "$train_failed" -gt 0 ]]; then
                send_notification "[PIPELINE ALERT] SAE Training Jobs Failed" \
                    "WARNING: $train_failed SAE training job(s) have failed!\n\nCheck logs: tail -50 $PROJECT_DIR/logs/step3_*.err"
            fi
            
            # Check for stage completion
            if [[ "$ext_running" -eq 0 && "$ext_pending" -eq 0 && "$ext_completed" -gt 0 ]]; then
                if ! grep -q "extraction_complete" "$STATUS_FILE" 2>/dev/null; then
                    echo "extraction_complete" >> "$STATUS_FILE"
                    send_notification "[PIPELINE] Extraction Stage Complete" \
                        "All extraction jobs have completed!\n\nCompleted: $ext_completed layers\nFailed: $ext_failed\n\nSAE Training is starting...\n\nWandB Dashboard: https://wandb.ai/nourmubarak/sae-captioning-bias"
                fi
            fi
            
            if [[ "$train_running" -eq 0 && "$train_pending" -eq 0 && "$train_completed" -gt 0 ]]; then
                if ! grep -q "training_complete" "$STATUS_FILE" 2>/dev/null; then
                    echo "training_complete" >> "$STATUS_FILE"
                    send_notification "[PIPELINE] SAE Training Complete" \
                        "All SAE training jobs have completed!\n\nCompleted: $train_completed\nFailed: $train_failed\n\nAnalysis stages are starting...\n\nWandB Dashboard: https://wandb.ai/nourmubarak/sae-captioning-bias"
                fi
            fi
            
            prev_state="$current_state"
        fi
        
        # Log current status
        log "Extraction: R=$ext_running P=$ext_pending C=$ext_completed F=$ext_failed | Training: R=$train_running P=$train_pending C=$train_completed F=$train_failed"
        
        # Check if all done
        local total_jobs=$(squeue -u jmsk62 2>/dev/null | grep -v JOBID | wc -l)
        if [[ "$total_jobs" -eq 0 ]]; then
            if ! grep -q "pipeline_complete" "$STATUS_FILE" 2>/dev/null; then
                echo "pipeline_complete" >> "$STATUS_FILE"
                send_notification "[PIPELINE COMPLETE] All Jobs Finished" \
                    "The full mechanistic interpretability pipeline has completed!\n\nCheck results:\n- Checkpoints: $PROJECT_DIR/checkpoints/\n- Visualizations: $PROJECT_DIR/visualizations/\n- Logs: $PROJECT_DIR/logs/\n\nWandB Dashboard: https://wandb.ai/nourmubarak/sae-captioning-bias"
                log "Pipeline complete! Exiting monitor."
                exit 0
            fi
        fi
        
        sleep 60  # Check every minute
    done
}

# Run in background
main_loop

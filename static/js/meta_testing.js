/**
 * Meta Testing Interface - ENHANCED with Log Polling
 */

let selectedMode = 'balanced';
let metaTestInterval = null;
let autoScrollMeta = true;

// Mode Selection
function selectTestMode(mode) {
    selectedMode = mode;
    
    // Update UI
    document.querySelectorAll('.test-mode-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    document.getElementById(`mode-${mode}`).classList.add('selected');
    document.getElementById('selected-mode').value = mode;
}

// Form Submission
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('meta-test-form');
    
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(form);
            const data = {
                mode: formData.get('mode'),
                max_experiments: formData.get('max_experiments') || null
            };
            
            try {
                const response = await fetch(form.action, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Show progress section
                    document.getElementById('progress-section').style.display = 'block';
                    document.getElementById('start-btn').disabled = true;
                    document.getElementById('start-btn').textContent = '⏳ Tests laufen...';
                    
                    // Start polling
                    startMetaTestPolling();
                    
                    // Scroll to progress
                    document.getElementById('progress-section').scrollIntoView({ 
                        behavior: 'smooth' 
                    });
                } else {
                    alert('Fehler: ' + (result.error || 'Unbekannter Fehler'));
                }
                
            } catch (error) {
                console.error('Submit error:', error);
                alert('Fehler beim Starten der Tests!');
            }
        });
    }
    
    // Load existing results on page load
    loadMetaTestResults();
    
    // Auto-scroll checkbox
    const checkbox = document.getElementById('meta-auto-scroll');
    if (checkbox) {
        checkbox.addEventListener('change', (e) => {
            autoScrollMeta = e.target.checked;
        });
    }
});

// Start polling for progress
function startMetaTestPolling() {
    metaTestInterval = setInterval(updateMetaTestProgress, 3000);
}

// ✅ Update progress MIT LOG
async function updateMetaTestProgress() {
    try {
        const response = await fetch('/api/meta_test_status');
        const status = await response.json();
        
        // Update progress bar
        const progress = Math.round(status.progress);
        document.getElementById('meta-progress-bar').style.width = `${progress}%`;
        document.getElementById('meta-progress-bar').textContent = `${progress}%`;
        document.getElementById('progress-percentage').textContent = `${progress}%`;
        
        // Update stats
        document.getElementById('stat-completed').textContent = status.completed || 0;
        document.getElementById('stat-running').textContent = status.running ? 1 : 0;
        document.getElementById('stat-failed').textContent = status.failed || 0;
        document.getElementById('stat-total').textContent = status.total || 0;
        
        // Update current test
        if (status.current_test) {
            document.getElementById('current-test').textContent = 
                `Aktuell: ${status.current_test}`;
        }
        
        // ✅ Update Log Output
        updateMetaLogOutput(status.log_output);
        
        // Check if completed
        if (status.status === 'completed') {
            clearInterval(metaTestInterval);
            
            document.getElementById('progress-title').textContent = '✅ Tests abgeschlossen!';
            document.getElementById('start-btn').disabled = false;
            document.getElementById('start-btn').textContent = '🔬 Meta-Testing starten';
            
            // Reload results
            loadMetaTestResults();
            
            showNotification('✅ Alle Tests erfolgreich abgeschlossen!', 'success');
        }
        
    } catch (error) {
        console.error('Progress error:', error);
    }
}

// ✅ Update Log Output
function updateMetaLogOutput(logText) {
    const logContent = document.getElementById('meta-log-content');
    const logContainer = document.getElementById('log-container');
    
    if (logText && logText.trim() !== "") {
        // Show log container
        logContainer.style.display = 'block';
        
        // Clean log (remove ANSI codes)
        const cleanLog = logText.replace(/\x1b\[[0-9;]*m/g, "");
        
        logContent.textContent = cleanLog;
        
        // Auto-scroll
        if (autoScrollMeta) {
            const logOutput = logContent.parentElement;
            logOutput.scrollTop = logOutput.scrollHeight;
        }
    } else {
        // ✅ FALLBACK: Try to read log file directly
        fetchMetaTestLog();
    }
}

// ✅ Fallback: Read log file directly via separate endpoint
async function fetchMetaTestLog() {
    try {
        const response = await fetch('/api/meta_test_log');
        if (!response.ok) return;
        
        const data = await response.json();
        
        if (data.log) {
            const logContent = document.getElementById('meta-log-content');
            const logContainer = document.getElementById('log-container');
            
            logContainer.style.display = 'block';
            
            // Clean log
            const cleanLog = data.log.replace(/\x1b\[[0-9;]*m/g, "");
            logContent.textContent = cleanLog;
            
            // Auto-scroll
            if (autoScrollMeta) {
                const logOutput = logContent.parentElement;
                logOutput.scrollTop = logOutput.scrollHeight;
            }
        }
    } catch (error) {
        console.error('Log fetch error:', error);
    }
}

// Load test results
async function loadMetaTestResults() {
    try {
        const response = await fetch('/api/meta_test_results');
        const results = await response.json();
        
        const resultsGrid = document.getElementById('results-grid');
        const resultsInfo = document.getElementById('results-info');
        
        if (results.runs && results.runs.length > 0) {
            resultsInfo.textContent = `${results.runs.length} Test-Runs gefunden`;
            resultsGrid.innerHTML = '';
            
            results.runs.forEach(run => {
                const card = createResultCard(run);
                resultsGrid.appendChild(card);
            });
        } else {
            resultsInfo.textContent = 'Keine Test-Ergebnisse vorhanden.';
            resultsGrid.innerHTML = '';
        }
        
    } catch (error) {
        console.error('Load results error:', error);
    }
}

// Create result card
function createResultCard(run) {
    const card = document.createElement('div');
    card.className = 'result-card';
    
    const statusClass = run.status === 'completed' ? 'status-completed' : 
                       run.status === 'running' ? 'status-running' : 'status-failed';
    
    const statusText = run.status === 'completed' ? '✅ Fertig' : 
                      run.status === 'running' ? '⏳ Läuft' : '❌ Fehler';
    
    const roi = run.metrics?.roi || 0;
    const winrate = run.metrics?.winrate || 0;
    
    card.innerHTML = `
        <div class="result-header">
            <div class="result-id">${run.run_id}</div>
            <div class="result-status ${statusClass}">${statusText}</div>
        </div>
        
        <div>
            <strong>Modus:</strong> ${run.mode}<br>
            <strong>Season:</strong> ${run.season}<br>
            <strong>Tests:</strong> ${run.experiments || 0}<br>
            <strong>Gestartet:</strong> ${formatDate(run.started_at)}
        </div>
        
        ${run.status === 'completed' ? `
            <div class="result-metrics">
                <div class="metric">
                    <div class="metric-label">ROI</div>
                    <div class="metric-value ${roi >= 0 ? 'positive' : 'negative'}">
                        ${(roi * 100).toFixed(2)}%
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Winrate</div>
                    <div class="metric-value">
                        ${(winrate * 100).toFixed(2)}%
                    </div>
                </div>
            </div>
            
            <div class="button-group" style="margin-top: 15px;">
                <a href="/meta_test_details/${run.run_id}" class="btn-primary btn-small" style="width: 100%;">
                    📊 Details ansehen
                </a>
            </div>
        ` : ''}
    `;
    
    return card;
}

// Helper: Format date
function formatDate(dateStr) {
    if (!dateStr) return 'N/A';
    const date = new Date(dateStr);
    return date.toLocaleString('de-DE', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Show notification
function showNotification(message, type = 'info') {
    alert(message);
}
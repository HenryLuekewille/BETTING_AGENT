/**
 * Status Page - Live Training Updates
 */

let statusInterval = null;
let isTrainingComplete = false;
let autoScroll = true;

// Start polling on page load
window.addEventListener("load", () => {
  startStatusPolling();
  
  // Auto-scroll checkbox
  const checkbox = document.getElementById("auto-scroll");
  if (checkbox) {
    checkbox.addEventListener("change", (e) => {
      autoScroll = e.target.checked;
    });
  }
});

function startStatusPolling() {
  // Initial update
  updateStatus();

  // Poll every 2 seconds
  statusInterval = setInterval(updateStatus, 2000);
}

async function updateStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Update UI
    updateProgressBar(status.progress);
    updateStatusMessage(status.message, status.phase);
    updateStatsGrid(status);
    updateLogOutput(status.log_output);  // ✅ NEU

    // Check if completed
    if (!status.running && status.phase === "completed") {
      handleTrainingComplete();
    } else if (!status.running && status.phase === "error") {
      handleTrainingError(status.message);
    }
  } catch (error) {
    console.error("Error fetching status:", error);
  }
}

function updateProgressBar(progress) {
  const progressBar = document.getElementById("progress-bar");
  progressBar.style.width = `${progress}%`;
  progressBar.textContent = `${progress}%`;
}

function updateStatusMessage(message, phase) {
  const statusMessage = document.getElementById("status-message");
  statusMessage.textContent = message;

  // Update header
  const statusPhase = document.getElementById("status-phase");
  const phaseEmojis = {
    idle: "💤",
    starting: "🚀",
    global_training: "🎓",
    fine_tuning: "🎯",
    evaluation: "📊",
    completed: "✅",
    error: "❌",
  };

  const emoji = phaseEmojis[phase] || "⏳";
  const phaseName = phase.replace(/_/g, " ").toUpperCase();
  statusPhase.textContent = `${emoji} ${phaseName}`;
}

function updateStatsGrid(status) {
  const statsGrid = document.getElementById("stats-grid");

  if (status.running || status.phase === "completed") {
    statsGrid.style.display = "grid";

    document.getElementById("stat-phase").textContent = status.phase.replace(
      /_/g,
      " "
    );
    document.getElementById("stat-progress").textContent = `${status.progress}%`;

    if (status.current_run) {
      const runName = status.current_run.split("/").pop();
      document.getElementById("stat-run").textContent = runName;
    }
  }
}

// ✅ NEU: Update Log Output
function updateLogOutput(logText) {
  const logContent = document.getElementById("log-content");
  const logContainer = logContent.parentElement;
  
  if (logText && logText.trim() !== "") {
    // Formatiere Log (entferne ANSI-Codes falls vorhanden)
    const cleanLog = logText.replace(/\x1b\[[0-9;]*m/g, "");
    
    logContent.textContent = cleanLog;
    
    // Auto-scroll zum Ende
    if (autoScroll) {
      logContainer.scrollTop = logContainer.scrollHeight;
    }
  } else {
    logContent.textContent = "Warte auf Log-Ausgabe...";
  }
}

function handleTrainingComplete() {
  if (isTrainingComplete) return;

  isTrainingComplete = true;

  // Stop polling
  if (statusInterval) {
    clearInterval(statusInterval);
  }

  // Show success message
  const statusMessage = document.getElementById("status-message");
  statusMessage.innerHTML = `
        <strong>✅ Training erfolgreich abgeschlossen!</strong><br>
        Die Ergebnisse können jetzt eingesehen werden.
    `;

  // Show results button
  document.getElementById("results-btn").style.display = "inline-block";

  // Confetti effect
  showConfetti();
}

function handleTrainingError(message) {
  // Stop polling
  if (statusInterval) {
    clearInterval(statusInterval);
  }

  // Show error
  const statusMessage = document.getElementById("status-message");
  statusMessage.innerHTML = `
        <strong>❌ Training fehlgeschlagen</strong><br>
        ${message}
    `;
  statusMessage.style.color = "var(--danger)";
}

function showConfetti() {
  // Simple confetti effect
  const duration = 3000;
  const animationEnd = Date.now() + duration;

  const colors = ["#3b82f6", "#10b981", "#f59e0b"];

  (function frame() {
    const timeLeft = animationEnd - Date.now();

    if (timeLeft <= 0) return;

    const particleCount = 2;

    for (let i = 0; i < particleCount; i++) {
      const particle = document.createElement("div");
      particle.style.position = "fixed";
      particle.style.width = "10px";
      particle.style.height = "10px";
      particle.style.background =
        colors[Math.floor(Math.random() * colors.length)];
      particle.style.left = Math.random() * window.innerWidth + "px";
      particle.style.top = "-10px";
      particle.style.opacity = "1";
      particle.style.borderRadius = "50%";
      particle.style.pointerEvents = "none";
      particle.style.zIndex = "9999";

      document.body.appendChild(particle);

      const fallDuration = 2000 + Math.random() * 1000;
      const horizontalMove = (Math.random() - 0.5) * 200;

      particle.animate(
        [
          { transform: "translateY(0px) translateX(0px)", opacity: 1 },
          {
            transform: `translateY(${window.innerHeight}px) translateX(${horizontalMove}px)`,
            opacity: 0,
          },
        ],
        {
          duration: fallDuration,
          easing: "cubic-bezier(0.25, 0.46, 0.45, 0.94)",
        }
      ).onfinish = () => particle.remove();
    }

    requestAnimationFrame(frame);
  })();
}
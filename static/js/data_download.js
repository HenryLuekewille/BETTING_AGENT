/**
 * Data Download & Preprocessing
 */

let downloadInterval = null;

function selectAllSeasons() {
  const checkboxes = document.querySelectorAll(".season-checkbox");
  const allChecked = Array.from(checkboxes).every((cb) => cb.checked);

  checkboxes.forEach((cb) => {
    cb.checked = !allChecked;
  });

  document.getElementById("btn-select-all").textContent = allChecked
    ? "✅ Alle auswählen"
    : "❌ Alle abwählen";
}

async function startDownload() {
  const checkboxes = document.querySelectorAll(".season-checkbox:checked");
  const seasons = Array.from(checkboxes).map((cb) => cb.value);

  if (seasons.length === 0) {
    alert("Bitte mindestens eine Saison auswählen!");
    return;
  }

  // Disable button
  const btn = document.getElementById("btn-download");
  btn.disabled = true;
  btn.textContent = "⏳ Download läuft...";

  // Show progress
  const progressDiv = document.getElementById("download-progress");
  progressDiv.style.display = "block";

  try {
    const response = await fetch("/api/download_data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seasons }),
    });

    const data = await response.json();

    if (data.error) {
      alert(data.error);
      resetDownloadUI();
      return;
    }

    // Start polling
    startDownloadPolling();
  } catch (error) {
    console.error("Download error:", error);
    alert("Fehler beim Starten des Downloads!");
    resetDownloadUI();
  }
}

function startDownloadPolling() {
  downloadInterval = setInterval(updateDownloadStatus, 1000);
}

async function updateDownloadStatus() {
  try {
    const response = await fetch("/api/download_status");
    const status = await response.json();

    // Update progress bar
    const progressBar = document.getElementById("download-progress-bar");
    progressBar.style.width = `${status.progress}%`;
    progressBar.textContent = `${status.progress}%`;

    // Update message
    const message = document.getElementById("download-message");
    message.textContent = status.message;

    // Check if complete
    if (!status.running) {
      clearInterval(downloadInterval);

      if (status.progress >= 50) {
        message.innerHTML =
          status.message +
          '<br><strong>→ Jetzt Preprocessing durchführen!</strong>';
        document.getElementById("btn-preprocess").disabled = false;
      }

      // Re-enable download button
      setTimeout(() => {
        resetDownloadUI();
        location.reload(); // Reload to show downloaded files
      }, 2000);
    }
  } catch (error) {
    console.error("Status error:", error);
  }
}

function resetDownloadUI() {
  const btn = document.getElementById("btn-download");
  btn.disabled = false;
  btn.textContent = "⬇️ Download starten";
}

async function startPreprocessing() {
  const btn = document.getElementById("btn-preprocess");
  btn.disabled = true;
  btn.textContent = "⏳ Preprocessing läuft...";

  // Show progress
  const progressDiv = document.getElementById("preprocess-progress");
  progressDiv.style.display = "block";

  try {
    const response = await fetch("/api/preprocess_data", { method: "POST" });
    const data = await response.json();

    if (data.error) {
      alert(data.error);
      resetPreprocessUI();
      return;
    }

    // Start polling
    startPreprocessPolling();
  } catch (error) {
    console.error("Preprocessing error:", error);
    alert("Fehler beim Starten des Preprocessing!");
    resetPreprocessUI();
  }
}

function startPreprocessPolling() {
  downloadInterval = setInterval(updatePreprocessStatus, 1000);
}

async function updatePreprocessStatus() {
  try {
    const response = await fetch("/api/download_status");
    const status = await response.json();

    // Update progress bar
    const progressBar = document.getElementById("preprocess-progress-bar");
    progressBar.style.width = `${status.progress}%`;
    progressBar.textContent = `${status.progress}%`;

    // Update message
    const message = document.getElementById("preprocess-message");
    message.textContent = status.message;

    // Check if complete
    if (!status.running && status.progress === 100) {
      clearInterval(downloadInterval);

      message.innerHTML =
        status.message + '<br><strong>✅ Bereit für Training!</strong>';

      // Reload page after 2 seconds
      setTimeout(() => {
        location.reload();
      }, 2000);
    }
  } catch (error) {
    console.error("Status error:", error);
  }
}

function resetPreprocessUI() {
  const btn = document.getElementById("btn-preprocess");
  btn.disabled = false;
  btn.textContent = "🔧 Preprocessing starten";
}
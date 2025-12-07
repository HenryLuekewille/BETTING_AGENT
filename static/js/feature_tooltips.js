/**
 * Feature Tooltips mit JavaScript
 * Robuste Lösung für komplexe Tooltip-Inhalte
 */

document.addEventListener('DOMContentLoaded', function() {
    initFeatureTooltips();
});

function initFeatureTooltips() {
    const tooltips = document.querySelectorAll('.tooltip[data-description]');
    
    tooltips.forEach(tooltip => {
        const tooltipEl = createTooltipElement(tooltip);
        
        tooltip.addEventListener('mouseenter', function(e) {
            showTooltip(tooltip, tooltipEl);
        });
        
        tooltip.addEventListener('mouseleave', function(e) {
            hideTooltip(tooltipEl);
        });
        
        // Touch support
        tooltip.addEventListener('touchstart', function(e) {
            e.preventDefault();
            toggleTooltip(tooltip, tooltipEl);
        });
    });
    
    // Close on scroll
    window.addEventListener('scroll', function() {
        document.querySelectorAll('.custom-tooltip.visible').forEach(el => {
            hideTooltip(el);
        });
    });
}

function createTooltipElement(triggerEl) {
    const description = triggerEl.getAttribute('data-description');
    const source = triggerEl.getAttribute('data-source');
    const derivation = triggerEl.getAttribute('data-derivation');
    
    const tooltip = document.createElement('div');
    tooltip.className = 'custom-tooltip';
    
    tooltip.innerHTML = `
        <div class="tooltip-content">
            <div class="tooltip-section">
                <span class="tooltip-icon">🔍</span>
                <strong>Beschreibung:</strong><br>
                <span class="tooltip-text">${description || 'Keine Beschreibung'}</span>
            </div>
            <div class="tooltip-section">
                <span class="tooltip-icon">⚙️</span>
                <strong>Quelle:</strong>
                <span class="tooltip-text">${source || 'Unbekannt'}</span>
            </div>
            <div class="tooltip-section">
                <span class="tooltip-icon">🧮</span>
                <strong>Berechnung:</strong><br>
                <span class="tooltip-text">${derivation || 'Nicht angegeben'}</span>
            </div>
        </div>
        <div class="tooltip-arrow"></div>
    `;
    
    document.body.appendChild(tooltip);
    return tooltip;
}

function showTooltip(triggerEl, tooltipEl) {
    const rect = triggerEl.getBoundingClientRect();
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
    
    tooltipEl.style.display = 'block';
    tooltipEl.style.opacity = '0';
    const tooltipRect = tooltipEl.getBoundingClientRect();
    
    let top = rect.top + scrollTop - tooltipRect.height - 15;
    let left = rect.left + scrollLeft + (rect.width / 2) - (tooltipRect.width / 2);
    
    // Check bounds
    if (top < scrollTop + 10) {
        top = rect.bottom + scrollTop + 15;
        tooltipEl.classList.add('tooltip-bottom');
    } else {
        tooltipEl.classList.remove('tooltip-bottom');
    }
    
    if (left < scrollLeft + 10) {
        left = scrollLeft + 10;
    } else if (left + tooltipRect.width > scrollLeft + window.innerWidth - 10) {
        left = scrollLeft + window.innerWidth - tooltipRect.width - 10;
    }
    
    tooltipEl.style.top = top + 'px';
    tooltipEl.style.left = left + 'px';
    
    setTimeout(() => {
        tooltipEl.classList.add('visible');
        tooltipEl.style.opacity = '1';
    }, 10);
}

function hideTooltip(tooltipEl) {
    tooltipEl.classList.remove('visible');
    tooltipEl.style.opacity = '0';
    setTimeout(() => {
        tooltipEl.style.display = 'none';
    }, 300);
}

function toggleTooltip(triggerEl, tooltipEl) {
    if (tooltipEl.classList.contains('visible')) {
        hideTooltip(tooltipEl);
    } else {
        document.querySelectorAll('.custom-tooltip.visible').forEach(el => {
            if (el !== tooltipEl) hideTooltip(el);
        });
        showTooltip(triggerEl, tooltipEl);
    }
}
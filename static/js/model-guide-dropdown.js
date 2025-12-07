/**
 * Model Guide Dropdown Functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('🔧 Model Guide Dropdowns initialisiert');
});

function toggleExplanation(button) {
    const card = button.closest('.parameter-card');
    const explanation = card.querySelector('.parameter-explanation');
    const icon = button.querySelector('.expand-icon');
    
    const isExpanded = explanation.classList.contains('expanded');
    
    if (isExpanded) {
        explanation.classList.remove('expanded');
        button.classList.remove('expanded');
        icon.style.transform = 'rotate(0deg)';
    } else {
        explanation.classList.add('expanded');
        button.classList.add('expanded');
        icon.style.transform = 'rotate(180deg)';
        
        setTimeout(() => {
            explanation.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'nearest' 
            });
        }, 300);
    }
}

// Keyboard support
document.addEventListener('keydown', function(e) {
    if (e.target.classList.contains('expand-btn')) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            e.target.click();
        }
    }
});
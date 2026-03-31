/**
 * FPTPlay Churn Prediction Web App
 * JavaScript for Interactivity and Animations
 */

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    initializeSidebar();
    initializeFormValidation();
    initializeTooltips();
    initializeAnimations();
});

// ========== SIDEBAR MANAGEMENT ==========

function initializeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('mainContent');
    const sidebarToggle = document.getElementById('sidebarToggle');
    
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('collapsed');
            mainContent.classList.toggle('expanded');
            
            // Save state to localStorage
            const isCollapsed = sidebar.classList.contains('collapsed');
            localStorage.setItem('sidebarCollapsed', isCollapsed);
        });
    }
    
    // Restore sidebar state from localStorage
    const sidebarCollapsed = localStorage.getItem('sidebarCollapsed');
    if (sidebarCollapsed === 'true') {
        sidebar.classList.add('collapsed');
        mainContent.classList.add('expanded');
    }
    
    // Mobile sidebar toggle
    if (window.innerWidth <= 768) {
        sidebar.classList.add('collapsed');
        mainContent.classList.add('expanded');
    }
}

// ========== FORM VALIDATION ==========

function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
    
    // Custom validation for number inputs
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('input', function() {
            validateNumberInput(this);
        });
    });
}

function validateNumberInput(input) {
    const value = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    
    if (isNaN(value)) {
        input.setCustomValidity('Vui lòng nhập số hợp lệ');
    } else if (min !== undefined && value < min) {
        input.setCustomValidity(`Giá trị phải >= ${min}`);
    } else if (max !== undefined && value > max) {
        input.setCustomValidity(`Giá trị phải <= ${max}`);
    } else {
        input.setCustomValidity('');
    }
}

// ========== TOOLTIPS ==========

function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// ========== ANIMATIONS ==========

function initializeAnimations() {
    // Fade in elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe all cards
    const cards = document.querySelectorAll('.card, .info-card, .stat-card, .kpi-card');
    cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(card);
    });
}

// ========== FILE UPLOAD HANDLING ==========

// Drag and drop file upload
const uploadArea = document.getElementById('uploadArea');
if (uploadArea) {
    const fileInput = document.getElementById('fileInput');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('dragover');
        });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('dragover');
        });
    });
    
    uploadArea.addEventListener('drop', handleDrop);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            showFileInfo(files[0]);
        }
    }
}

function showFileInfo(file) {
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    
    if (fileInfo && fileName && fileSize) {
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.style.display = 'block';
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ========== FORM AUTO-FILL (for testing) ==========

function autoFillForm(scenario) {
    const scenarios = {
        high_risk: {
            hours_m6: 25.0,
            hours_m5: 22.0,
            hours_m4: 18.0,
            hours_m3: 15.0,
            hours_m2: 10.0,
            hours_m1: 5.0,
            tenure_months: 24,
            is_promo_subscriber: 0,
            device_type: 'mobile',
            plan_type: 'basic',
            region: 'south'
        },
        low_risk: {
            hours_m6: 18.0,
            hours_m5: 19.0,
            hours_m4: 20.0,
            hours_m3: 22.0,
            hours_m2: 24.0,
            hours_m1: 25.0,
            tenure_months: 36,
            is_promo_subscriber: 1,
            device_type: 'tv',
            plan_type: 'premium',
            region: 'north'
        }
    };
    
    const data = scenarios[scenario];
    if (!data) return;
    
    Object.keys(data).forEach(key => {
        const input = document.querySelector(`[name="${key}"]`);
        if (input) {
            input.value = data[key];
        }
    });
}

// Add keyboard shortcut for dev mode (Ctrl+Shift+D)
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.shiftKey && e.key === 'D') {
        const devMenu = confirm('Dev Mode:\n\n1. Fill High Risk Scenario\n2. Fill Low Risk Scenario\n\nChoose 1 or 2');
        if (devMenu) {
            const choice = prompt('Enter 1 or 2:');
            if (choice === '1') {
                autoFillForm('high_risk');
            } else if (choice === '2') {
                autoFillForm('low_risk');
            }
        }
    }
});

// ========== UTILITY FUNCTIONS ==========

// Number formatting
function formatNumber(num, decimals = 0) {
    return num.toLocaleString('vi-VN', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

// Percentage formatting
function formatPercentage(value) {
    return (value * 100).toFixed(1) + '%';
}

// Currency formatting
function formatCurrency(value) {
    return '$' + value.toLocaleString('en-US', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    });
}

// ========== EXPORT FUNCTIONS ==========

// Make functions available globally
window.autoFillForm = autoFillForm;
window.formatNumber = formatNumber;
window.formatPercentage = formatPercentage;
window.formatCurrency = formatCurrency;

// ========== LOADING INDICATOR ==========

function showLoading() {
    const loading = document.createElement('div');
    loading.id = 'loadingOverlay';
    loading.innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-3">Đang xử lý...</p>
    `;
    loading.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.9);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    `;
    document.body.appendChild(loading);
}

function hideLoading() {
    const loading = document.getElementById('loadingOverlay');
    if (loading) {
        loading.remove();
    }
}

// Show loading on form submit
document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', function() {
        if (form.checkValidity()) {
            showLoading();
        }
    });
});

// ========== CONSOLE INFO ==========

console.log('%c FPTPlay Churn Prediction System ', 'background: #1976d2; color: white; font-size: 16px; padding: 10px;');
console.log('Version: 1.0.0');
console.log('Dev shortcuts:');
console.log('  - Ctrl+Shift+D: Auto-fill form');
console.log('Made with ❤️ for FPT Telecom');

// Contoh data untuk Malignant (Terkena)
const malignantExample = [
    17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
];

// Contoh data untuk Benign (Tidak Terkena)
const benignExample = [
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
    0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
    15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
];

/**
 * Mengisi form dengan contoh data
 * Fungsi ini dipanggil dari HTML melalui onclick attribute
 * @param {string} type - 'malignant' atau 'benign'
 */
// eslint-disable-next-line no-unused-vars
function fillExampleData(type) {
    const exampleData = type === 'malignant' ? malignantExample : benignExample;
    
    // Mengisi semua input field
    for (let i = 0; i < 30; i++) {
        const input = document.getElementById(`f${i}`);
        if (input) {
            input.value = exampleData[i];
            
            // Menambahkan efek visual saat mengisi
            input.style.backgroundColor = '#e8f5e9';
            setTimeout(() => {
                input.style.backgroundColor = '';
            }, 500);
        }
    }
    
    // Menampilkan notifikasi
    showNotification(
        type === 'malignant' 
            ? 'Data contoh "Terkena (Malignant)" telah diisi' 
            : 'Data contoh "Tidak Terkena (Benign)" telah diisi',
        type === 'malignant' ? 'danger' : 'success'
    );
    
    // Scroll ke form
    document.getElementById('predictionForm').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start' 
    });
}

/**
 * Menampilkan notifikasi
 * @param {string} message - Pesan notifikasi
 * @param {string} type - Tipe notifikasi ('success', 'danger', 'info')
 */
function showNotification(message, type = 'info') {
    // Membuat elemen notifikasi
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 80px; right: 20px; z-index: 9999; min-width: 300px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Menghapus notifikasi setelah 3 detik
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Animasi saat halaman dimuat
document.addEventListener('DOMContentLoaded', function() {
    console.log('Sistem Deteksi Kanker Payudara siap digunakan');
    
    // Inisialisasi Bootstrap tooltips (setelah Bootstrap dimuat)
    function initTooltips() {
        // eslint-disable-next-line no-undef
        if (typeof window.bootstrap !== 'undefined' || typeof bootstrap !== 'undefined') {
            // eslint-disable-next-line no-undef
            const Bootstrap = window.bootstrap || bootstrap;
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new Bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    }
    
    // Coba inisialisasi langsung
    initTooltips();
    
    // Jika belum berhasil, tunggu sebentar (Bootstrap mungkin masih loading)
    setTimeout(initTooltips, 500);
    
    // Animasi progress bar di halaman result
    const progressBar = document.getElementById('progress-bar');
    if (progressBar) {
        const probability = parseFloat(progressBar.getAttribute('data-probability')) || 0;
        progressBar.style.width = '0%';
        setTimeout(() => {
            progressBar.style.width = probability + '%';
        }, 100);
    }
    
    // Validasi form sebelum submit (hanya di halaman index)
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            // Validasi bahwa semua field terisi
            const inputs = form.querySelectorAll('input[type="number"]');
            let isValid = true;
            
            inputs.forEach(input => {
                if (!input.value || input.value.trim() === '') {
                    isValid = false;
                    input.style.borderColor = '#dc3545';
                } else {
                    input.style.borderColor = '';
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                showNotification('Mohon lengkapi semua field sebelum melakukan prediksi', 'danger');
            }
        });
    }
    
    // Animasi untuk input fields saat focus (hanya di halaman index)
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.style.transform = 'scale(1.02)';
            this.style.transition = 'transform 0.2s ease';
        });
        
        input.addEventListener('blur', function() {
            this.style.transform = 'scale(1)';
        });
    });
});

// Smooth scroll untuk anchor links
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

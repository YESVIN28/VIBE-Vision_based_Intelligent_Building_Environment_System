// DOM Elements
const loginBtn = document.getElementById('login-btn');
const logoutBtn = document.getElementById('logout-btn');
const loginModal = document.getElementById('login-modal');
const closeModal = document.querySelector('.close-modal');
const loginForm = document.getElementById('login-form');
const mainContent = document.getElementById('main-content');
const startDetectionBtn = document.getElementById('start-detection');
const stopDetectionBtn = document.getElementById('stop-detection');
const systemStatus = document.getElementById('system-status');
const fullscreenBtn = document.getElementById('fullscreen-btn');
const videoFeed = document.getElementById('video-feed');

// State variables
let isAuthenticated = false;
let isDetectionRunning = false;
let updateInterval;
let lastUpdateTime = 0;
let connectionRetries = 0;
const maxRetries = 3;

// API Configuration
const API_BASE_URL = '';
const API_ENDPOINTS = {
    startDetection: '/api/start_detection',
    stopDetection: '/api/stop_detection',
    getStatus: '/api/detection_status',
    videoFeed: '/video_feed',
    systemInfo: '/api/system_info'
};

// Initialize the application
function init() {
    console.log('Initializing Crowd Management System...');
    checkAuthStatus();
    setupEventListeners();
    updateUI();
    
    // Load system info on startup
    if (isAuthenticated) {
        loadSystemInfo();
        checkInitialDetectionStatus();
    }
}

// Authentication functions
function checkAuthStatus() {
    const token = localStorage.getItem('authToken');
    isAuthenticated = !!token;
    console.log('Authentication status:', isAuthenticated);
}

function setupEventListeners() {
    // Authentication event listeners
    loginBtn.addEventListener('click', () => {
        loginModal.style.display = 'flex';
        document.getElementById('username').focus();
    });
    
    logoutBtn.addEventListener('click', logout);
    closeModal.addEventListener('click', () => loginModal.style.display = 'none');
    loginForm.addEventListener('submit', handleLogin);
    
    // Detection control event listeners
    startDetectionBtn.addEventListener('click', startDetection);
    stopDetectionBtn.addEventListener('click', stopDetection);
    
    // Video controls
    fullscreenBtn.addEventListener('click', toggleFullscreen);
    videoFeed.addEventListener('click', toggleFullscreen);
    
    // Modal close on outside click
    window.addEventListener('click', (e) => {
        if (e.target === loginModal) {
            loginModal.style.display = 'none';
        }
    });
    
    // Fullscreen change listener
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    
    // Handle page visibility changes
    document.addEventListener('visibilitychange', handleVisibilityChange);
}

function updateUI() {
    // Update authentication UI
    loginBtn.style.display = isAuthenticated ? 'none' : 'block';
    logoutBtn.style.display = isAuthenticated ? 'block' : 'none';
    mainContent.style.display = isAuthenticated ? 'block' : 'none';
    
    // Update system status
    systemStatus.textContent = isDetectionRunning ? 'Running' : 'Stopped';
    systemStatus.className = `status-badge ${isDetectionRunning ? 'running' : 'stopped'}`;
    
    // Update control buttons
    startDetectionBtn.disabled = isDetectionRunning;
    stopDetectionBtn.disabled = !isDetectionRunning;
    
    // Update button styles
    if (isDetectionRunning) {
        startDetectionBtn.classList.add('disabled');
        stopDetectionBtn.classList.remove('disabled');
    } else {
        startDetectionBtn.classList.remove('disabled');
        stopDetectionBtn.classList.add('disabled');
    }
}

async function handleLogin(e) {
    e.preventDefault();
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value;

    // Simple authentication (replace with real authentication in production)
    if (username === 'admin' && password === 'admin123') {
        localStorage.setItem('authToken', `token-${Date.now()}`);
        isAuthenticated = true;
        loginModal.style.display = 'none';
        updateUI();
        
        // Load system information and check detection status
        await loadSystemInfo();
        await checkInitialDetectionStatus();
        
        console.log('Login successful');
    } else {
        alert('Invalid credentials. Please use admin/admin123');
        document.getElementById('password').value = '';
        document.getElementById('password').focus();
    }
}

function logout() {
    localStorage.removeItem('authToken');
    isAuthenticated = false;
    isDetectionRunning = false;
    stopRealTimeUpdates();
    resetUI();
    updateUI();
    console.log('Logged out successfully');
}

// Detection control functions
async function startDetection() {
    console.log('Starting detection...');
    startDetectionBtn.disabled = true;
    startDetectionBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
    
    try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.startDetection}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Detection start response:', result);
        
        isDetectionRunning = true;
        
        // Start video feed
        const timestamp = Date.now();
        videoFeed.src = `${API_BASE_URL}${API_ENDPOINTS.videoFeed}?t=${timestamp}`;
        videoFeed.onerror = handleVideoError;
        
        updateUI();
        startRealTimeUpdates();
        connectionRetries = 0;
        
        console.log('Detection started successfully');
        
    } catch (error) {
        console.error('Failed to start detection:', error);
        alert('Failed to start detection. Please check if the backend server is running.');
    } finally {
        startDetectionBtn.disabled = false;
        startDetectionBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
    }
}

async function stopDetection() {
    console.log('Stopping detection...');
    stopDetectionBtn.disabled = true;
    stopDetectionBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Stopping...';
    
    try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.stopDetection}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Detection stop response:', result);
        
        isDetectionRunning = false;
        stopRealTimeUpdates();
        
        // Reset video feed to placeholder
        videoFeed.src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQwIiBoZWlnaHQ9IjQ4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzMzIi8+CiAgPHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIyNCIgZmlsbD0iI2ZmZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkNhbWVyYSBGZWVkIE9mZmxpbmU8L3RleHQ+Cjwvc3ZnPg==";
        
        // Reset all stats
        resetStats();
        updateUI();
        
        console.log('Detection stopped successfully');
        
    } catch (error) {
        console.error('Failed to stop detection:', error);
        alert('Failed to stop detection. Please check if the backend server is running.');
    } finally {
        stopDetectionBtn.disabled = false;
        stopDetectionBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Detection';
    }
}

// Data fetching functions
async function checkInitialDetectionStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.getStatus}`);
        if (response.ok) {
            const data = await response.json();
            console.log('Initial detection status:', data);
            
            if (data.running) {
                isDetectionRunning = true;
                const timestamp = Date.now();
                videoFeed.src = `${API_BASE_URL}${API_ENDPOINTS.videoFeed}?t=${timestamp}`;
                startRealTimeUpdates();
            }
            
            updateDetectionStatus(data);
            updateUI();
        }
    } catch (error) {
        console.error('Error checking initial detection status:', error);
    }
}

async function fetchDetectionStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.getStatus}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        updateDetectionStatus(data);
        connectionRetries = 0; // Reset retry counter on successful fetch
        
    } catch (error) {
        console.error('Error fetching detection status:', error);
        connectionRetries++;
        
        if (connectionRetries >= maxRetries) {
            console.warn('Max connection retries reached, stopping updates');
            stopRealTimeUpdates();
            // Show connection error in UI
            showConnectionError();
        }
    }
}

async function loadSystemInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.systemInfo}`);
        if (response.ok) {
            const data = await response.json();
            updateSystemInfo(data);
        }
    } catch (error) {
        console.error('Error loading system info:', error);
    }
}

function updateDetectionStatus(data) {
    if (data.running !== undefined) {
        isDetectionRunning = data.running;
        updateUI();
    }

    if (data.stats) {
        const stats = data.stats;

        // Update main stats
        document.getElementById('people-count').textContent = stats.people_count || 0;
        document.getElementById('density').textContent = `${(stats.density || 0).toFixed(4)} ppl/m²`;

        // Update crowd level with confidence
        const crowdText = `${stats.crowd_level || 'Unknown'} (${Math.round((stats.crowd_confidence || 0) * 100)}%)`;
        document.getElementById('crowd-level').textContent = crowdText;

        // Update device status
        const deviceStatus = (stats.device_status || 'off').toUpperCase();
        document.getElementById('device-status').textContent = deviceStatus;

        // Update video stats
        document.getElementById('fps').textContent = (stats.fps || 0).toFixed(1);
        document.getElementById('processing-time').textContent = `${(stats.processing_time || 0).toFixed(2)} ms`;
        document.getElementById('frame-count').textContent = stats.frame_count || 0;

        // Update active tracks count
        const activeTracksCount = (stats.tracked_ids || []).length;
        document.getElementById('active-tracks').textContent = activeTracksCount;

        // Update tracking information
        updateTrackingTable(stats.tracking_info || []);
        updateTrackingSummary(stats.tracked_ids || [], stats.tracking_info || []);

        // Update device status indicator color
        updateDeviceStatusIndicator(deviceStatus);

        // ✅ NEW: Update device control UI (switches + sliders)
        if (stats.devices) {
            Object.entries(stats.devices).forEach(([deviceId, config]) => {
                const type = deviceId.split('-')[0]; // e.g., "fan-123" → "fan"
                const card = document.querySelector(`.device-card[data-device="${type}"]`);
                if (!card) return;

                // Power switch
                const switchEl = card.querySelector('.power-switch');
                const statusText = card.querySelector('.status-text');
                const statusIndicator = card.querySelector('.status-indicator');

                if (config.state) {
                    switchEl.classList.add('on');
                    statusText.textContent = 'Online';
                    statusIndicator.classList.add('active');
                } else {
                    switchEl.classList.remove('on');
                    statusText.textContent = 'Offline';
                    statusIndicator.classList.remove('active');
                }

                // Sliders (speed, brightness, volume, temperature)
                ['speed', 'brightness', 'volume', 'temperature'].forEach(control => {
                    if (config[control] !== undefined) {
                        const slider = card.querySelector(`input[data-control="${control}"]`);
                        const valueLabel = card.querySelector(`.slider-label[data-control="${control}"]`);
                        if (slider) slider.value = config[control];
                        if (valueLabel) valueLabel.textContent = `${config[control]}%`;
                    }
                });
            });
        }

        lastUpdateTime = Date.now();
    }
}


function updateTrackingTable(trackingInfo) {
    const tableBody = document.getElementById('tracking-data');
    
    if (!trackingInfo || trackingInfo.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="4" class="no-data">No active tracks</td></tr>';
        return;
    }
    
    tableBody.innerHTML = '';
    
    trackingInfo.forEach(track => {
        const row = document.createElement('tr');
        const statusClass = track.status.toLowerCase().replace(' ', '-');
        
        row.innerHTML = `
            <td class="track-id">${track.track_id}</td>
            <td class="track-position">${track.position}</td>
            <td class="track-confidence">${track.confidence}</td>
            <td class="track-status ${statusClass}">${track.status}</td>
        `;
        
        tableBody.appendChild(row);
    });
}

function updateTrackingSummary(trackedIds, trackingInfo) {
    const totalTracked = trackedIds.length;
    const activeTracks = trackingInfo.filter(track => track.status.toLowerCase() === 'confirmed').length;
    
    document.getElementById('total-tracked').textContent = totalTracked;
    document.getElementById('active-tracks-summary').textContent = activeTracks;
}

function updateSystemInfo(info) {
    if (info.device) {
        document.getElementById('processing-device').textContent = info.device;
    }
    if (info.yolo_model) {
        document.getElementById('yolo-model').textContent = info.yolo_model;
    }
    if (info.resnet_enabled !== undefined) {
        document.getElementById('resnet-enabled').textContent = info.resnet_enabled ? 'Yes' : 'No';
    }
    if (info.camera_fov) {
        document.getElementById('camera-fov').textContent = `${info.camera_fov} m²`;
    }
    if (info.density_threshold) {
        document.getElementById('density-threshold').textContent = `${info.density_threshold} ppl/m²`;
    }
}

function updateDeviceStatusIndicator(status) {
    const deviceStatusElement = document.getElementById('device-status');
    const deviceIcon = document.querySelector('.device-icon');
    
    if (status === 'ON') {
        deviceStatusElement.classList.add('status-on');
        deviceStatusElement.classList.remove('status-off');
        deviceIcon.style.backgroundColor = '#2ecc71';
    } else {
        deviceStatusElement.classList.add('status-off');
        deviceStatusElement.classList.remove('status-on');
        deviceIcon.style.backgroundColor = '#e74c3c';
    }
}

function resetStats() {
    document.getElementById('people-count').textContent = '0';
    document.getElementById('density').textContent = '0.00 ppl/m²';
    document.getElementById('crowd-level').textContent = 'Unknown (0%)';
    document.getElementById('device-status').textContent = 'OFF';
    document.getElementById('fps').textContent = '0.0';
    document.getElementById('processing-time').textContent = '0.00 ms';
    document.getElementById('frame-count').textContent = '0';
    document.getElementById('active-tracks').textContent = '0';
    document.getElementById('total-tracked').textContent = '0';
    document.getElementById('active-tracks-summary').textContent = '0';
    
    // Reset tracking table
    document.getElementById('tracking-data').innerHTML = '<tr><td colspan="4" class="no-data">No tracking data available</td></tr>';
    
    // Reset device status indicator
    updateDeviceStatusIndicator('OFF');
}

function resetUI() {
    resetStats();
    videoFeed.src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQwIiBoZWlnaHQ9IjQ4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzMzIi8+CiAgPHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIyNCIgZmlsbD0iI2ZmZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkNhbWVyYSBGZWVkIE9mZmxpbmU8L3RleHQ+Cjwvc3ZnPg==";
}

function showConnectionError() {
    // You could implement a more sophisticated error display here
    console.warn('Connection to backend lost');
}

// Real-time update functions
function startRealTimeUpdates() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    console.log('Starting real-time updates...');
    updateInterval = setInterval(() => {
        if (isDetectionRunning && isAuthenticated) {
            fetchDetectionStatus();
        }
    }, 1000); // Update every second
    
    // Initial fetch
    fetchDetectionStatus();
}

function stopRealTimeUpdates() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
        console.log('Stopped real-time updates');
    }
}

// Video control functions
function toggleFullscreen() {
    const videoWrapper = document.querySelector('.video-wrapper');
    
    if (!document.fullscreenElement) {
        videoWrapper.requestFullscreen().catch(err => {
            console.error('Error attempting to enable fullscreen:', err);
        });
    } else {
        document.exitFullscreen();
    }
}

function handleFullscreenChange() {
    const icon = fullscreenBtn.querySelector('i');
    if (document.fullscreenElement) {
        icon.className = 'fas fa-compress';
        videoFeed.style.cursor = 'zoom-out';
    } else {
        icon.className = 'fas fa-expand';
        videoFeed.style.cursor = 'pointer';
    }
}

function handleVideoError() {
    console.error('Video feed error');
    if (isDetectionRunning) {
        // Try to reload the video feed
        setTimeout(() => {
            const timestamp = Date.now();
            videoFeed.src = `${API_BASE_URL}${API_ENDPOINTS.videoFeed}?t=${timestamp}`;
        }, 2000);
    }
}

// Page visibility handling
function handleVisibilityChange() {
    if (document.hidden) {
        // Page is hidden, reduce update frequency or pause updates
        if (updateInterval) {
            clearInterval(updateInterval);
            updateInterval = setInterval(() => {
                if (isDetectionRunning && isAuthenticated) {
                    fetchDetectionStatus();
                }
            }, 5000); // Update every 5 seconds when hidden
        }
    } else {
        // Page is visible, resume normal update frequency
        if (isDetectionRunning && isAuthenticated) {
            startRealTimeUpdates();
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', init);

// Handle page unload
window.addEventListener('beforeunload', () => {
    stopRealTimeUpdates();
});

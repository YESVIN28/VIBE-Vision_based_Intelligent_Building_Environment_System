// Device Control System JavaScript
// Updated to work with the existing Flask API endpoints

// DOM Elements
const loginBtn = document.getElementById('login-btn');
const logoutBtn = document.getElementById('logout-btn');
const mainContent = document.getElementById('main-content');
const devicesGrid = document.getElementById('devices-grid');
const systemStatus = document.getElementById('system-status');

// State variables (using memory instead of localStorage)
let isAuthenticated = false;
let deviceStates = {};
let authToken = null;
let detectionRunning = false;
let currentStats = {};

// API Base URL
const API_BASE = '/api';

// Initialize the application
function init() {
    console.log('🚀 Initializing Device Control System...');
    checkAuthStatus();
    setupEventListeners();
    // Load devices first, then start control loop after devices are loaded
    loadDevicesFromServer().then(() => {
        console.log('✅ Devices loaded successfully');
        // Delay starting the density control to ensure DOM is ready
        setTimeout(() => {
            startDensityControlLoop();
            console.log('🔄 Density control loop started');
        }, 1000);
    }).catch(err => {
        console.error('❌ Error loading devices:', err);
    });
    loadSystemStatus();
    updateUI();
    
    // Start periodic updates
    setInterval(updateSystemStatus, 2000); // Update every 2 seconds
    setInterval(loadDevices, 2000);
}

function loadDevices() {
    fetch('http://192.168.137.143:8000/api/devices') // Replace <PI_IP> with your Pi's IP
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            updateDevicesUI(data.devices);
        }
    })
    .catch(error => {
        console.error('Error loading devices:', error);
    });
}


// Authentication functions
function checkAuthStatus() {
    // In a real application, you would check with your server
    // For now, we'll use a simple in-memory authentication state
    isAuthenticated = authToken !== null;
    console.log('Authentication status:', isAuthenticated);
}

function setupEventListeners() {
    // Authentication event listeners
    if (loginBtn) {
        loginBtn.addEventListener('click', () => {
            simulateLogin();
        });
    }
    
    if (logoutBtn) {
        logoutBtn.addEventListener('click', logout);
    }
    
    // Detection control buttons
    const startDetectionBtn = document.getElementById('start-detection-btn');
    const stopDetectionBtn = document.getElementById('stop-detection-btn');
    
    if (startDetectionBtn) {
        startDetectionBtn.addEventListener('click', startDetection);
    }
    
    if (stopDetectionBtn) {
        stopDetectionBtn.addEventListener('click', stopDetection);
    }
    
    // Device switch toggles
    document.querySelectorAll('.power-switch').forEach(switchEl => {
        switchEl.addEventListener('click', function() {
            const deviceId = this.getAttribute('data-device');
            toggleDeviceState(deviceId, this);
        });
    });
    
    // Slider controls
    document.querySelectorAll('.slider').forEach(slider => {
        slider.addEventListener('input', function() {
            const value = this.value;
            const currentValueSpan = this.parentElement.querySelector('.current-value');
            if (currentValueSpan) {
                currentValueSpan.textContent = value + '%';
            }
            
            const deviceCard = this.closest('.device-card');
            const deviceId = deviceCard.getAttribute('data-device');
            const controlType = this.getAttribute('data-control');
            
            updateDeviceControl(deviceId, controlType, value);
        });
    });
    
    // Add device button
    const addDeviceBtn = document.getElementById('add-device-btn');
    if (addDeviceBtn) {
        addDeviceBtn.addEventListener('click', function() {
            showAddDeviceModal();
        });
    }
    
    // Modal event listeners
    setupModalEventListeners();
}

function setupModalEventListeners() {
    const modal = document.getElementById('addDeviceModal');
    const closeBtn = document.querySelector('.close');
    
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            hideAddDeviceModal();
        });
    }
    
    // Close modal when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            hideAddDeviceModal();
        }
    });

    window.location.href = '/frontend/device-control.html';
        fetch('/frontend/device-control.html');
    
    // Device type selection
    document.querySelectorAll('.device-type').forEach(type => {
        type.addEventListener('click', function() {
            document.querySelectorAll('.device-type').forEach(t => t.classList.remove('selected'));
            this.classList.add('selected');
        });
    });
    
    // Form submission
    const deviceForm = document.getElementById('deviceForm');
    if (deviceForm) {
        deviceForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleAddDevice();
        });
    }
}

function simulateLogin() {
    // Simulate authentication
    authToken = 'simulated-token-' + Date.now();
    isAuthenticated = true;
    updateUI();
    console.log('Logged in successfully');
}

function logout() {
    authToken = null;
    isAuthenticated = false;
    deviceStates = {};
    updateUI();
    console.log('Logged out successfully');
}

function updateUI() {
    // Update authentication UI
    if (loginBtn) {
        loginBtn.style.display = isAuthenticated ? 'none' : 'block';
    }
    if (logoutBtn) {
        logoutBtn.style.display = isAuthenticated ? 'block' : 'none';
    }
    
    if (mainContent) {
        mainContent.style.display = isAuthenticated ? 'block' : 'none';
    }
}

// Detection Control Functions
function startDetection() {
    fetch(`${API_BASE}/start_detection`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('Detection started:', data);
        detectionRunning = true;
        updateDetectionUI();
        showNotification('Detection started successfully', 'success');
    })
    .catch(error => {
        console.error('Error starting detection:', error);
        showNotification('Failed to start detection', 'error');
    });
}

function stopDetection() {
    fetch(`${API_BASE}/stop_detection`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('Detection stopped:', data);
        detectionRunning = false;
        updateDetectionUI();
        showNotification('Detection stopped successfully', 'success');
    })
    .catch(error => {
        console.error('Error stopping detection:', error);
        showNotification('Failed to stop detection', 'error');
    });
}

function updateDetectionUI() {
    const startBtn = document.getElementById('start-detection-btn');
    const stopBtn = document.getElementById('stop-detection-btn');
    const statusIndicator = document.getElementById('detection-status');
    
    if (startBtn) startBtn.disabled = detectionRunning;
    if (stopBtn) stopBtn.disabled = !detectionRunning;
    
    if (statusIndicator) {
        statusIndicator.textContent = detectionRunning ? 'Running' : 'Stopped';
        statusIndicator.className = detectionRunning ? 'status-active' : 'status-inactive';
    }
}

// System Status Functions
function loadSystemStatus() {
    fetch(`${API_BASE}/detection_status`)
    .then(response => response.json())
    .then(data => {
        detectionRunning = data.running;
        currentStats = data.stats;
        updateSystemStatusUI(data);
        updateDetectionUI();
    })
    .catch(error => {
        console.error('Error loading system status:', error);
    });
}

function updateSystemStatus() {
    if (!isAuthenticated) return;
    
    fetch(`${API_BASE}/detection_status`)
    .then(response => response.json())
    .then(data => {
        detectionRunning = data.running;
        currentStats = data.stats;
        updateSystemStatusUI(data);
    })
    .catch(error => {
        console.error('Error updating system status:', error);
    });
}

function updateSystemStatusUI(data) {
    // Update people count
    const peopleCountEl = document.getElementById('people-count');
    if (peopleCountEl) {
        peopleCountEl.textContent = data.stats.people_count || 0;
    }
    
    // Update density
    const densityEl = document.getElementById('density-value');
    if (densityEl) {
        densityEl.textContent = (data.stats.density || 0).toFixed(4);
    }
    
    // Update device status
    const deviceStatusEl = document.getElementById('auto-device-status');
    if (deviceStatusEl) {
        deviceStatusEl.textContent = data.stats.device_status || 'off';
        deviceStatusEl.className = data.stats.device_status === 'on' ? 'status-on' : 'status-off';
    }
    
    // Update crowd level
    const crowdLevelEl = document.getElementById('crowd-level');
    if (crowdLevelEl) {
        crowdLevelEl.textContent = `${data.stats.crowd_level} (${data.stats.crowd_confidence}%)`;
    }
    
    // Update FPS
    const fpsEl = document.getElementById('fps-value');
    if (fpsEl) {
        fpsEl.textContent = data.stats.fps || 0;
    }
    
    // Update tracked IDs
    const trackedIdsEl = document.getElementById('tracked-ids');
    if (trackedIdsEl) {
        const ids = data.stats.tracked_ids || [];
        trackedIdsEl.textContent = ids.length > 0 ? ids.join(', ') : 'None';
    }
    
    // Update processing time
    const processingTimeEl = document.getElementById('processing-time');
    if (processingTimeEl) {
        processingTimeEl.textContent = `${data.stats.processing_time || 0}ms`;
    }
}

// Device Control Functions
function toggleDeviceState(deviceId, switchElement) {
    const isCurrentlyOn = switchElement.classList.contains('on');
    const newState = !isCurrentlyOn;

    // Optimistically update UI
    updateDeviceUI(deviceId, newState, switchElement);

    // Send to backend
    fetch(`${API_BASE}/device/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device: deviceId, state: newState })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status !== 'success') {
            // If backend says error, revert UI
            updateDeviceUI(deviceId, isCurrentlyOn, switchElement);
            showNotification(data.message, 'error');
        }
    })
    .catch(() => {
        // On network error, revert UI
        updateDeviceUI(deviceId, isCurrentlyOn, switchElement);
        showNotification('Network error occurred', 'error');
    });
}

function updateDeviceUI(deviceId, isOn, switchElement) {
    const deviceCard = switchElement.closest('.device-card');
    const statusIndicator = deviceCard.querySelector('.status-indicator');
    const statusText = deviceCard.querySelector('.status-text');
    
    if (isOn) {
        switchElement.classList.add('on');
        switchElement.classList.remove('off');
        if (statusIndicator) statusIndicator.classList.add('active');
        if (statusText) statusText.textContent = 'Online';
    } else {
        switchElement.classList.remove('on');
        switchElement.classList.add('off');
        if (statusIndicator) statusIndicator.classList.remove('active');
        if (statusText) statusText.textContent = 'Offline';
        
        // Reset all sliders to 0
        deviceCard.querySelectorAll('.slider').forEach(slider => {
            slider.value = 0;
            const currentValueSpan = slider.parentElement.querySelector('.current-value');
            if (currentValueSpan) {
                currentValueSpan.textContent = '0%';
            }
        });
    }
}

function updateDeviceControl(deviceId, controlType, value) {
    const deviceCard = document.querySelector(`[data-device="${deviceId}"]`);
    const slider = deviceCard.querySelector(`.slider[data-control="${controlType}"]`);
    const currentValueSpan = slider.parentElement.querySelector('.current-value');

    // Optimistically update UI
    slider.value = value;
    if (currentValueSpan) currentValueSpan.textContent = value + '%';

    // Send to backend
    fetch(`${API_BASE}/device/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device: deviceId, control: controlType, value: parseInt(value) })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status !== 'success') {
            // If backend says error, revert UI (will be fixed on next poll)
            showNotification(data.message, 'error');
        }
    })
    .catch(() => {
        showNotification('Network error occurred', 'error');
    });
}

function showAddDeviceModal() {
    const modal = document.getElementById('addDeviceModal');
    if (modal) {
        modal.style.display = 'block';
    }
}

function hideAddDeviceModal() {
    const modal = document.getElementById('addDeviceModal');
    if (modal) {
        modal.style.display = 'none';
        resetAddDeviceForm();
    }
}

function resetAddDeviceForm() {
    const form = document.getElementById('deviceForm');
    if (form) {
        form.reset();
    }
    document.querySelectorAll('.device-type').forEach(t => t.classList.remove('selected'));
}

function handleAddDevice() {
    const selectedType = document.querySelector('.device-type.selected');
    if (!selectedType) {
        showNotification('Please select a device type', 'error');
        return;
    }
    
    const deviceType = selectedType.getAttribute('data-type');
    const deviceName = document.getElementById('deviceName').value;
    const deviceRoom = document.getElementById('deviceRoom').value;
    
    if (!deviceName || !deviceRoom) {
        showNotification('Please fill in all fields', 'error');
        return;
    }
    
    // Send new device to server
    fetch(`${API_BASE}/device/add`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            type: deviceType,
            name: deviceName,
            room: deviceRoom
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('New device added:', data);
            // Create device card in UI
            createDeviceCard(data.device_id, data.device);
            hideAddDeviceModal();
            showNotification(data.message, 'success');
        } else {
            console.error('Error adding device:', data.message);
            showNotification('Error adding device: ' + data.message, 'error');
        }
    })
    .catch(error => {
        console.error('Network error:', error);
        showNotification('Network error. Please try again.', 'error');
    });
}

function createDeviceCard(deviceId, device) {
    const devicesGrid = document.getElementById('devices-grid');
    const addDeviceCard = document.querySelector('.add-device-card');
    
    if (!devicesGrid) return;
    
    const icons = {
        fan: 'fas fa-fan',
        light: 'fas fa-lightbulb',
        tv: 'fas fa-tv',
        ac: 'fas fa-snowflake'
    };
    
    const controls = {
        fan: { label: 'Speed Control', type: 'speed' },
        light: { label: 'Brightness', type: 'brightness' },
        tv: { label: 'Volume', type: 'volume' },
        ac: { label: 'Temperature', type: 'temperature' }
    };
    
    const deviceType = device.type || 'unknown';
    const deviceCard = document.createElement('div');
    deviceCard.className = 'device-card';
    deviceCard.setAttribute('data-device', deviceId);
    
    deviceCard.innerHTML = `
        <div class="device-header">
            <div class="device-title">
                <div class="device-icon">
                    <i class="${icons[deviceType] || 'fas fa-microchip'}"></i>
                </div>
                <span>${device.name} (${device.room})</span>
            </div>
            <div class="power-switch off" data-device="${deviceId}"></div>
        </div>
        <div class="device-status">
            <div class="status-indicator"></div>
            <span class="status-text">Offline</span>
        </div>
        <div class="controls-section">
            <div class="control-group">
                <label class="control-label">${controls[deviceType]?.label || 'Control'}</label>
                <div class="slider-container">
                    <input type="range" class="slider" min="0" max="100" value="0" data-control="${controls[deviceType]?.type || 'value'}">
                    <div class="slider-value">
                        <span>0%</span>
                        <span class="current-value">0%</span>
                        <span>100%</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="device-actions">
            <button class="remove-device-btn" onclick="removeDevice('${deviceId}')">
                <i class="fas fa-trash"></i> Remove
            </button>
        </div>
    `;
    
    if (addDeviceCard) {
        devicesGrid.insertBefore(deviceCard, addDeviceCard);
    } else {
        devicesGrid.appendChild(deviceCard);
    }
    
    // Add event listeners to new device
    const powerSwitch = deviceCard.querySelector('.power-switch');
    const slider = deviceCard.querySelector('.slider');
    
    if (powerSwitch) {
        powerSwitch.addEventListener('click', function() {
            toggleDeviceState(deviceId, this);
        });
    }
    
    if (slider) {
        slider.addEventListener('input', function() {
            const value = this.value;
            const currentValueSpan = this.parentElement.querySelector('.current-value');
            if (currentValueSpan) {
                currentValueSpan.textContent = value + '%';
            }
            
            const controlType = this.getAttribute('data-control');
            updateDeviceControl(deviceId, controlType, value);
        });
    }
}

function removeDevice(deviceId) {
    if (!confirm('Are you sure you want to remove this device?')) {
        return;
    }
    
    fetch(`${API_BASE}/device/remove/${deviceId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Remove device card from UI
            const deviceCard = document.querySelector(`[data-device="${deviceId}"]`);
            if (deviceCard) {
                deviceCard.remove();
            }
            delete deviceStates[deviceId];
            showNotification(data.message, 'success');
        } else {
            showNotification('Error removing device: ' + data.message, 'error');
        }
    })
    .catch(error => {
        console.error('Network error:', error);
        showNotification('Network error occurred', 'error');
    });
}

function loadDevicesFromServer() {
    console.log('📱 Loading devices from server...');
    return new Promise((resolve, reject) => {
        fetch('/api/devices')
            .then(async response => {
                if (!response.ok) {
                    const text = await response.text();
                    console.error('Error response from server:', text);
                    throw new Error(`Server responded with status ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('📱 Devices loaded:', data.devices);
                updateDevicesUI(data.devices);
                
                // Add debugging to see device structure
                if (data.devices) {
                    console.log('📊 Device structure:', JSON.stringify(data.devices, null, 2));
                }
                
                // Show debug info about device cards in DOM
                const deviceCards = document.querySelectorAll('.device-card');
                console.log(`📱 Found ${deviceCards.length} device cards in DOM:`);
                deviceCards.forEach(card => {
                    const id = card.getAttribute('data-device');
                    console.log(`- Device card: ${id}`);
                });
                
                resolve(data.devices);
            })
            .catch(error => {
                console.error('❌ Error loading devices:', error);
                reject(error);
            });
    });
}


function updateDevicesUI(devices) {
    const devicesGrid = document.getElementById('devices-grid');
    devicesGrid.innerHTML = ''; // Clear all

    // LED Device
    if (devices.led) {
        const led = devices.led;
        devicesGrid.innerHTML += `
        <div class="device-card" data-device="led">
            <div class="device-header">
                <div class="device-title">
                    <div class="device-icon"><i class="fas fa-lightbulb"></i></div>
                    <span>${led.name}</span>
                </div>
                <div class="power-switch ${led.state ? 'on' : 'off'}" data-device="led"></div>
            </div>
            <div class="device-status">
                <div class="status-indicator${led.online ? ' active' : ''}" style="background:${led.online ? '#48bb78' : '#e53e3e'}"></div>
                <span class="status-text">${led.online ? 'Online' : 'Offline'}</span>
            </div>
            <div class="controls-section">
                <div class="control-group">
                    <label class="control-label">Brightness</label>
                    <div class="slider-container">
                        <input type="range" class="slider" min="0" max="100" value="${led.brightness}" data-control="brightness">
                        <div class="slider-value">
                            <span>0%</span>
                            <span class="current-value">${led.brightness}%</span>
                            <span>100%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>`;
    }

    // DHT11 Device
    if (devices.dht11) {
        const dht = devices.dht11;
        devicesGrid.innerHTML += `
        <div class="device-card" data-device="dht11">
            <div class="device-header">
                <div class="device-title">
                    <div class="device-icon"><i class="fas fa-thermometer-half"></i></div>
                    <span>${dht.name}</span>
                </div>
            </div>
            <div class="device-status">
                <div class="status-indicator${dht.online ? ' active' : ''}" style="background:${dht.online ? '#48bb78' : '#e53e3e'}"></div>
                <span class="status-text">${dht.online ? 'Online' : 'Offline'}</span>
            </div>
            <div class="controls-section">
                <div class="control-group">
                    <label class="control-label">Temperature</label>
                    <div>${dht.temperature !== undefined ? dht.temperature + '°C' : 'N/A'}</div>
                </div>
            </div>
        </div>`;
    }

    // Add event listeners for the LED slider and switch
    document.querySelectorAll('.power-switch').forEach(switchEl => {
        switchEl.addEventListener('click', function() {
            const deviceId = this.getAttribute('data-device');
            const newState = !this.classList.contains('on');
            fetch('http://192.168.137.143:8000/api/device/toggle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ device: deviceId, state: newState })
            }).then(() => loadDevices());
        });
    });

    document.querySelectorAll('.slider').forEach(slider => {
        slider.addEventListener('input', function() {
            const value = this.value;
            const currentValueSpan = this.parentElement.querySelector('.current-value');
            if (currentValueSpan) currentValueSpan.textContent = value + '%';
        });
        slider.addEventListener('change', function() {
            const value = this.value;
            fetch('http://192.168.137.143:8000/api/device/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ device: 'led', control: 'brightness', value: parseInt(value) })
            }).then(() => loadDevices());
        });
    });
}

// Notification System
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <span>${message}</span>
        <button class="notification-close">&times;</button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
    
    // Manual close
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.addEventListener('click', () => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    });
}

// Load system information
function loadSystemInfo() {
    fetch(`${API_BASE}/system_info`)
    .then(response => response.json())
    .then(data => {
        console.log('System info:', data);
        // Update system info UI if elements exist
        const deviceEl = document.getElementById('system-device');
        if (deviceEl) deviceEl.textContent = data.device;
        
        const yoloEl = document.getElementById('yolo-model');
        if (yoloEl) yoloEl.textContent = data.yolo_model;
        
        const resnetEl = document.getElementById('resnet-enabled');
        if (resnetEl) resnetEl.textContent = data.resnet_enabled ? 'Yes' : 'No';
    })
    .catch(error => {
        console.error('Error loading system info:', error);
    });
}

// Density Control Config
const DENSITY_POLL_INTERVAL = 2000; // 2 seconds

function startDensityControlLoop() {
    console.log(`🔄 Starting density control loop (interval: ${DENSITY_POLL_INTERVAL}ms)`);
    
    // Run once immediately to check if everything works
    fetchAndApplyDensity().then(() => {
        console.log('✅ Initial density control check complete');
    }).catch(err => {
        console.error('❌ Error in initial density control:', err);
    });
    
    // Then start the interval
    setInterval(fetchAndApplyDensity, DENSITY_POLL_INTERVAL);
    // Also manually trigger the loop when detection status changes
    document.getElementById('start-detection-btn')?.addEventListener('click', () => {
        setTimeout(fetchAndApplyDensity, 1000);
    });
}

async function fetchAndApplyDensity() {
    try {
        // First get detection status to check people count
        const detectionRes = await fetch('/api/detection_status');
        const detectionData = await detectionRes.json();
        const peopleCount = detectionData.stats?.people_count || 0;
        
        // Get density data
        const densityRes = await fetch('/api/density');
        const densityData = await densityRes.json();
        const density = densityData.density || 0;

        // Clear debug log to track the automation process
        console.clear();
        console.log(`🔍 DETECTION DATA: People count: ${peopleCount}, Density: ${density}`);
        console.log(`⏱️ Time: ${new Date().toLocaleTimeString()}`);
        
        // Auto-control based on presence detection
        if (peopleCount > 0) {
            console.log("👤 PERSON DETECTED! Activating auto-control...");
            
            // Get all available devices in the DOM
            const deviceCards = document.querySelectorAll('.device-card');
            console.log(`📱 Found ${deviceCards.length} devices in DOM`);
            
            // List all devices found
            deviceCards.forEach(card => {
                const deviceId = card.getAttribute('data-device');
                console.log(`- Device found: ${deviceId}`);
            });
            
            // Turn on devices (if they're not already on)
            deviceCards.forEach(card => {
                const deviceId = card.getAttribute('data-device');
                if (deviceId === 'fan' || deviceId === 'light') {
                    turnOnDeviceIfNeeded(deviceId);
                }
            });
            
            // Calculate control values based on density
            // Map density to control value with a more sensitive scale
            // Higher density = higher control values (max 100)
            const fanSpeedValue = calculateControlValue(density, 0.3, 30, 100);
            const lightBrightnessValue = calculateControlValue(density, 0.3, 40, 100);
            
            console.log(`🎛️ AUTO-ADJUSTING: Fan: ${fanSpeedValue}%, Light: ${lightBrightnessValue}%`);
            
            // Apply to devices
            autoAdjustDevice('fan', 'speed', fanSpeedValue);
            autoAdjustDevice('light', 'brightness', lightBrightnessValue);
        } else {
            // No people detected - devices can remain in their current state
            console.log("❌ NO PEOPLE DETECTED - Devices remain in current state");
        }
    } catch (err) {
        console.error('❌ Density control error:', err);
    }
}

// Calculate control value based on density with custom scaling
function calculateControlValue(density, threshold, minValue, maxValue) {
    if (density <= 0) return 0;
    
    // Apply a non-linear scaling to make the control more responsive
    // Square root function gives more granular control at lower densities
    const scaledDensity = Math.sqrt(density) * 2;
    
    // Clamp the value between min and max
    return Math.min(maxValue, Math.max(minValue, Math.round(scaledDensity * 100)));
}

// Check and turn on a device if it's not already on
function turnOnDeviceIfNeeded(deviceId) {
    const deviceCard = document.querySelector(`[data-device="${deviceId}"]`);
    if (!deviceCard) {
        console.warn(`⚠️ Device not found: ${deviceId}`);
        return;
    }
    
    const powerSwitch = deviceCard.querySelector('.power-switch');
    if (!powerSwitch) {
        console.warn(`⚠️ Power switch not found for device: ${deviceId}`);
        return;
    }
    
    if (!powerSwitch.classList.contains('on')) {
        console.log(`🔌 AUTO-TURNING ON: ${deviceId}`);
        // Force change UI immediately
        powerSwitch.classList.add('on');
        powerSwitch.classList.remove('off');
        
        const statusIndicator = deviceCard.querySelector('.status-indicator');
        const statusText = deviceCard.querySelector('.status-text');
        
        if (statusIndicator) statusIndicator.classList.add('active');
        if (statusText) statusText.textContent = 'Online';
        
        // Then update server state
        toggleDeviceState(deviceId, powerSwitch);
    } else {
        console.log(`✓ Device already on: ${deviceId}`);
    }
}

function autoAdjustDevice(deviceId, controlType, value) {
    const deviceCard = document.querySelector(`[data-device="${deviceId}"]`);
    if (!deviceCard) {
        console.warn(`⚠️ Cannot adjust - Device not found: ${deviceId}`);
        return;
    }
    const slider = deviceCard.querySelector(`.slider[data-control="${controlType}"]`);
    if (!slider) {
        console.warn(`⚠️ Control slider not found: ${deviceId} - ${controlType}`);
        return;
    }
    
    const powerSwitch = deviceCard.querySelector('.power-switch');
    if (!powerSwitch) {
        console.warn(`⚠️ Power switch not found: ${deviceId}`);
        return;
    }

    // Ensure device is on before adjusting
    if (!powerSwitch.classList.contains('on')) {
        console.log(`🔌 Auto-turning on ${deviceId} before adjusting`);
        turnOnDeviceIfNeeded(deviceId);
    }
    
    // Only update if the value is different (reduces unnecessary updates)
    if (parseInt(slider.value) !== value) {
        console.log(`🎚️ Adjusting ${deviceId} ${controlType} to ${value}%`);
        
        // Force UI update immediately for responsive feel
        slider.value = value;
        
        // This triggers browser UI update for the slider
        const event = new Event('input', { bubbles: true });
        slider.dispatchEvent(event);
        
        const currentValueSpan = slider.parentElement.querySelector('.current-value');
        if (currentValueSpan) {
            currentValueSpan.textContent = value + '%';
        }
        
        // Update device control via API
        fetch(`${API_BASE}/device/control`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                device: deviceId,
                control: controlType,
                value: parseInt(value)
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                console.log(`✓ Successfully adjusted ${deviceId} ${controlType}`);
                // Update local state
                if (!deviceStates[deviceId]) deviceStates[deviceId] = {};
                deviceStates[deviceId][controlType] = parseInt(value);
            } else {
                console.error(`❌ Error adjusting ${deviceId}: ${data.message}`);
            }
        })
        .catch(error => {
            console.error('❌ Network error in auto-adjust:', error);
        });
    } else {
        console.log(`✓ ${deviceId} ${controlType} already at ${value}%`);
    }
}

// Poll and sync devices from server
function pollAndSyncDevices() {
    fetch('/api/devices')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                updateDevicesUI(data.devices);
            }
        })
        .catch(err => console.error('Device poll error:', err));
}

// Call this every 2 seconds
setInterval(pollAndSyncDevices, 2000);

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    init();
    loadSystemInfo();
    startDensityControlLoop();
});
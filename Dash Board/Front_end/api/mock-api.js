// This is a mock API for development purposes only
// In production, you would connect to your real backend API

console.log('Mock API loaded - for development only');

// Mock API endpoints
const mockApi = {
    login: async (username, password) => {
        return new Promise((resolve) => {
            setTimeout(() => {
                if (username === 'admin' && password === 'admin123') {
                    resolve({
                        success: true,
                        token: 'mock-token-123'
                    });
                } else {
                    resolve({
                        success: false,
                        message: 'Invalid credentials'
                    });
                }
            }, 500);
        });
    },
    
    logout: async () => {
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({ success: true });
            }, 300);
        });
    },
    
    startDetection: async () => {
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({ 
                    success: true,
                    message: 'Detection started'
                });
            }, 800);
        });
    },
    
    stopDetection: async () => {
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({ 
                    success: true,
                    message: 'Detection stopped'
                });
            }, 500);
        });
    },
    
    getStatus: async () => {
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    running: true,
                    stats: {
                        people_count: Math.floor(Math.random() * 20),
                        density: (Math.random() * 2).toFixed(2),
                        device_status: Math.random() > 0.5 ? 'on' : 'off',
                        crowd_level: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
                        crowd_confidence: (Math.random() * 0.3 + 0.6).toFixed(2),
                        tracked_ids: Array.from({length: Math.floor(Math.random() * 10)}, (_, i) => 100 + i),
                        processing_time: Math.floor(Math.random() * 10 + 35),
                        fps: (Math.random() * 5 + 20).toFixed(1),
                        frame_count: Math.floor(Math.random() * 1000)
                    }
                });
            }, 400);
        });
    }
};

// Uncomment to use mock API instead of real API in scripts.js
/*
window.mockApi = mockApi;
*/
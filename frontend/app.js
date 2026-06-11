// VoiceDoc Intelligence - Frontend JavaScript
// Web Speech API + WebSocket + Real-time Updates

const API_BASE_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

let ws = null;
let recognition = null;
let sessionId = null;

// Initialize Web Speech API
function initSpeechRecognition() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        alert('Speech recognition not supported in this browser. Please use Chrome or Edge.');
        return null;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
        document.getElementById('voiceStatus').textContent = '🎤 Listening...';
        document.getElementById('voiceBtn').classList.add('bg-red-500', 'hover:bg-red-600');
        document.getElementById('voiceBtn').classList.remove('bg-blue-500', 'hover:bg-blue-600');
    };

    recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript + ' ';
            } else {
                interimTranscript += transcript;
            }
        }

        document.getElementById('transcript').innerHTML = 
            `<p class="font-semibold">${finalTranscript}</p>
             <p class="text-gray-500 italic">${interimTranscript}</p>`;

        if (finalTranscript) {
            sendVoiceCommand(finalTranscript.trim());
        }
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        document.getElementById('voiceStatus').textContent = `❌ Error: ${event.error}`;
        resetVoiceButton();
    };

    recognition.onend = () => {
        document.getElementById('voiceStatus').textContent = '✅ Done';
        resetVoiceButton();
    };

    return recognition;
}

// Reset voice button state
function resetVoiceButton() {
    const btn = document.getElementById('voiceBtn');
    btn.classList.remove('bg-red-500', 'hover:bg-red-600');
    btn.classList.add('bg-blue-500', 'hover:bg-blue-600');
    btn.innerHTML = '<i class="fas fa-microphone mr-2"></i> Start Speaking';
}

// Initialize WebSocket connection
function initWebSocket() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('✅ WebSocket connected');
        sessionId = Date.now().toString();
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
        console.log('❌ WebSocket disconnected');
        setTimeout(initWebSocket, 3000); // Reconnect after 3 seconds
    };
}

// Handle incoming WebSocket messages
function handleWebSocketMessage(data) {
    switch(data.type) {
        case 'progress':
            updateProgress(data);
            break;
        case 'result':
            displayResult(data);
            break;
        case 'complete':
            handleCompletion(data);
            break;
        case 'error':
            displayError(data);
            break;
    }
}

// Send voice command to backend
async function sendVoiceCommand(command) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                command: command,
                session_id: sessionId
            })
        });

        if (response.ok) {
            document.getElementById('progressSection').classList.remove('hidden');
            document.getElementById('resultsSection').classList.remove('hidden');
        } else {
            console.error('Failed to send command');
        }
    } catch (error) {
        console.error('Error sending command:', error);
    }
}

// Update progress display
function updateProgress(data) {
    const container = document.getElementById('progressContainer');
    const agentName = data.agent;
    const progress = data.progress;
    const status = data.status;

    let progressHtml = container.querySelector(`[data-agent="${agentName}"]`);
    
    if (!progressHtml) {
        progressHtml = document.createElement('div');
        progressHtml.setAttribute('data-agent', agentName);
        progressHtml.className = 'mb-4 p-4 bg-gray-50 rounded-lg';
        container.appendChild(progressHtml);
    }

    const statusIcon = status === 'completed' ? '✅' : 
                       status === 'running' ? '⚙️' : 
                       status === 'failed' ? '❌' : '⏳';

    progressHtml.innerHTML = `
        <div class="flex justify-between items-center mb-2">
            <span class="font-semibold">${statusIcon} ${agentName}</span>
            <span class="text-sm text-gray-600">${progress}%</span>
        </div>
        <div class="w-full bg-gray-200 rounded-full h-2">
            <div class="bg-blue-500 h-2 rounded-full transition-all" style="width: ${progress}%"></div>
        </div>
    `;
}

// Display results
function displayResult(data) {
    const container = document.getElementById('resultsContainer');
    const resultDiv = document.createElement('div');
    resultDiv.className = 'mb-4 p-4 border border-gray-200 rounded-lg';
    
    resultDiv.innerHTML = `
        <h3 class="font-semibold text-lg mb-2">${data.title || 'Document'}</h3>
        <p class="text-gray-600 text-sm mb-2">${data.url || ''}</p>
        <div class="flex gap-4 text-sm text-gray-500">
            <span>📄 ${data.chunks || 0} chunks</span>
            <span>⭐ Score: ${data.score || 0}/10</span>
        </div>
    `;
    
    container.appendChild(resultDiv);
}

// Handle completion
function handleCompletion(data) {
    document.getElementById('querySection').classList.remove('hidden');
    
    // Show completion notification
    const notification = document.createElement('div');
    notification.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-4 rounded-lg shadow-lg';
    notification.innerHTML = `
        <p class="font-semibold">✅ Processing Complete!</p>
        <p class="text-sm">${data.message}</p>
    `;
    document.body.appendChild(notification);
    
    setTimeout(() => notification.remove(), 5000);
}

// Display error
function displayError(data) {
    const notification = document.createElement('div');
    notification.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-4 rounded-lg shadow-lg';
    notification.innerHTML = `
        <p class="font-semibold">❌ Error</p>
        <p class="text-sm">${data.message}</p>
    `;
    document.body.appendChild(notification);
    
    setTimeout(() => notification.remove(), 5000);
}

// Handle query submission
async function submitQuery(query) {
    const resultsContainer = document.getElementById('queryResults');
    
    // Show loading
    resultsContainer.innerHTML = '<p class="text-gray-500">🔍 Searching...</p>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: sessionId
            })
        });

        const data = await response.json();
        
        if (response.ok) {
            resultsContainer.innerHTML = `
                <div class="bg-blue-50 p-4 rounded-lg mb-4">
                    <p class="font-semibold mb-2">Answer:</p>
                    <p class="text-gray-700">${data.answer}</p>
                </div>
                ${data.sources ? `
                    <div class="mt-4">
                        <p class="font-semibold mb-2">Sources:</p>
                        <ul class="list-disc list-inside text-sm text-gray-600">
                            ${data.sources.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            `;
        } else {
            resultsContainer.innerHTML = '<p class="text-red-500">❌ Failed to get answer</p>';
        }
    } catch (error) {
        console.error('Query error:', error);
        resultsContainer.innerHTML = '<p class="text-red-500">❌ Error processing query</p>';
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize speech recognition
    recognition = initSpeechRecognition();
    
    // Initialize WebSocket
    initWebSocket();
    
    // Voice button
    document.getElementById('voiceBtn').addEventListener('click', () => {
        if (recognition) {
            recognition.start();
        }
    });
    
    // Query button
    document.getElementById('queryBtn').addEventListener('click', () => {
        const query = document.getElementById('queryInput').value.trim();
        if (query) {
            submitQuery(query);
        }
    });
    
    // Enter key for query
    document.getElementById('queryInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const query = e.target.value.trim();
            if (query) {
                submitQuery(query);
            }
        }
    });
});

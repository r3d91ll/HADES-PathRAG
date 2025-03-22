document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and panes
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // Add active class to clicked button
            button.classList.add('active');
            
            // Show corresponding tab pane
            const tabId = button.getAttribute('data-tab');
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });
    
    // Range input value display
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        const valueDisplay = input.nextElementSibling;
        valueDisplay.textContent = input.value;
        
        input.addEventListener('input', () => {
            valueDisplay.textContent = input.value;
        });
    });
    
    // API endpoint (assumes the backend is running on the same host)
    const API_BASE_URL = '/mcp';
    
    // Helper function for API calls
    async function callApi(endpoint, data) {
        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API error: ${response.status} - ${errorText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API call error:', error);
            return { error: error.message };
        }
    }
    
    // Helper function to display results
    function displayResults(containerId, results) {
        const container = document.getElementById(containerId);
        container.style.display = 'block';
        
        const jsonElement = container.querySelector('.json-results');
        jsonElement.textContent = JSON.stringify(results, null, 2);
        
        // Scroll to results
        container.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    // 1. PathRAG Query Form
    const queryForm = document.getElementById('query-form');
    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const data = {
            query: document.getElementById('query-text').value,
            min_weight: parseFloat(document.getElementById('min-weight').value),
            max_distance: parseInt(document.getElementById('max-distance').value),
            direction: document.getElementById('direction').value ? 
                      parseInt(document.getElementById('direction').value) : null,
            domain_filter: document.getElementById('domain-filter').value || null,
            as_of_version: document.getElementById('as-of-version').value || null
        };
        
        // Remove null values for cleaner requests
        Object.keys(data).forEach(key => {
            if (data[key] === null || data[key] === '') {
                delete data[key];
            }
        });
        
        const results = await callApi('/pathrag/retrieve', data);
        displayResults('query-results', results);
    });
    
    // 2. Create Relationship Form
    const relationshipForm = document.getElementById('relationship-form');
    relationshipForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        let metadata = null;
        try {
            const metadataText = document.getElementById('rel-metadata').value;
            if (metadataText) {
                metadata = JSON.parse(metadataText);
            }
        } catch (error) {
            alert('Invalid JSON in metadata field');
            return;
        }
        
        const data = {
            from_entity: document.getElementById('from-entity').value,
            to_entity: document.getElementById('to-entity').value,
            weight: parseFloat(document.getElementById('rel-weight').value),
            direction: parseInt(document.getElementById('rel-direction').value),
            valid_from: document.getElementById('valid-from').value || null,
            valid_until: document.getElementById('valid-until').value || null,
            metadata: metadata
        };
        
        // Remove null values for cleaner requests
        Object.keys(data).forEach(key => {
            if (data[key] === null || data[key] === '') {
                delete data[key];
            }
        });
        
        const results = await callApi('/pathrag/create_relationship', data);
        displayResults('relationship-results', results);
    });
    
    // 3. Assume Identity Form
    const identityForm = document.getElementById('identity-form');
    identityForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const data = {
            user_id: document.getElementById('user-id').value,
            object_id: document.getElementById('object-id').value,
            duration_minutes: parseInt(document.getElementById('duration').value)
        };
        
        const results = await callApi('/pathrag/assume_identity', data);
        displayResults('identity-results', results);
    });
    
    // 4. Verify Access Form
    const accessForm = document.getElementById('access-form');
    accessForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const data = {
            user_id: document.getElementById('access-user-id').value,
            resource_id: document.getElementById('resource-id').value,
            min_weight: parseFloat(document.getElementById('access-min-weight').value),
            identity_token_id: document.getElementById('token-id').value || null
        };
        
        // Remove null values for cleaner requests
        Object.keys(data).forEach(key => {
            if (data[key] === null || data[key] === '') {
                delete data[key];
            }
        });
        
        const results = await callApi('/pathrag/verify_access', data);
        displayResults('access-results', results);
    });
    
    // 5. Ollama Chat
    const chatForm = document.getElementById('chat-form');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage('user', message);
        chatInput.value = '';
        
        // Create thinking indicator
        const thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'system-message';
        thinkingDiv.textContent = 'Thinking...';
        chatMessages.appendChild(thinkingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Call Ollama API
        const data = {
            prompt: message,
            model: 'llama3',  // Default model, could be made configurable
            system: 'You are an AI assistant for the HADES-PathRAG system. You help users with understanding XnX notation and PathRAG concepts.'
        };
        
        try {
            const response = await callApi('/ollama/generate', data);
            // Remove thinking indicator
            chatMessages.removeChild(thinkingDiv);
            
            // Add assistant response
            if (response.error) {
                addMessage('system', `Error: ${response.error}`);
            } else {
                addMessage('assistant', response.response);
            }
        } catch (error) {
            chatMessages.removeChild(thinkingDiv);
            addMessage('system', `Error: ${error.message}`);
        }
    });
    
    function addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = role + '-message';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Add XnX Notation visualization on model view changes
    function updateXnXVisualization() {
        // This would be expanded in a full implementation to show
        // real-time visualization of XnX notation as parameters change
        console.log("XnX visualization would update here");
    }
    
    // Listen for weight and direction changes to update visualization
    document.getElementById('rel-weight').addEventListener('input', updateXnXVisualization);
    document.getElementById('rel-direction').addEventListener('change', updateXnXVisualization);
});

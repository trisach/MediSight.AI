// frontend/script.js

// --- Configuration ---
const BACKEND_URL = "http://localhost:8000"; // Adjust if needed

// --- DOM Elements ---
// Upload Section
const pdfFileInput = document.getElementById('pdf-file');
const submitButton = document.getElementById('submit-button');
const loadingSpinner = document.getElementById('loading-spinner');
const uploadError = document.getElementById('upload-error');
const uploadStatus = document.getElementById('upload-status');

// Analysis Section
const analysisSection = document.getElementById('analysis-section');
const detailedAnalysisP = document.getElementById('detailed-analysis');
// Removed resultsTableBody and resultsTableEmpty as the section is gone
const potentialRisksUl = document.getElementById('potential-risks');
const potentialRisksEmpty = document.getElementById('potential-risks-empty');
// Get the single list element for recommendations
const recommendationsListUl = document.getElementById('recommendations-list');
const recommendationsListEmpty = document.getElementById('recommendations-list-empty');
// Removed rawLLMResponseContainer and rawLLMResponsePre

// Chat Section
const chatSection = document.getElementById('chat-section');
const chatHistoryDiv = document.getElementById('chat-history');
const chatInput = document.getElementById('chat-input');
const sendChatButton = document.getElementById('send-chat-button');
const chatLoadingSpinner = document.getElementById('chat-loading-spinner');
const chatError = document.getElementById('chat-error');

// --- Event Listeners ---
submitButton.addEventListener('click', handleAnalyzeRequest);
sendChatButton.addEventListener('click', handleChatRequest);
chatInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        handleChatRequest();
    }
});

// --- Functions ---

/**
 * Handles the PDF upload and analysis request.
 */
async function handleAnalyzeRequest() {
    // (Keep this function the same as in med_report_analyzer_frontend_js_v2)
    const file = pdfFileInput.files[0];
    if (!file) {
        showUploadError("Please select a PDF file first.");
        return;
    }
    if (file.type !== "application/pdf") {
        showUploadError("Invalid file type. Only PDF files are allowed.");
        return;
    }
    hideUploadError();
    hideAnalysis();
    hideChat();
    showLoading();
    setUploadStatus("Uploading and analyzing... This may take a minute.");
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await fetch(`${BACKEND_URL}/analyze`, {
            method: 'POST',
            body: formData,
        });
        if (!response.ok) {
            let errorDetail = `Analysis failed (Status: ${response.status})`;
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || `Error ${response.status}: ${response.statusText}`;
            } catch (e) {
                 errorDetail = `Error ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorDetail);
        }
        const data = await response.json();
        displayAnalysis(data); // Call the updated displayAnalysis
        showAnalysis();
        showChat();
        setUploadStatus("Analysis complete.");
    } catch (error) {
        console.error("Analysis error:", error);
        showUploadError(`Analysis failed: ${error.message}`);
        setUploadStatus("");
    } finally {
        hideLoading();
    }
}

/**
 * Handles sending a chat message and displaying the response.
 */
async function handleChatRequest() {
    // (Keep this function the same as in med_report_analyzer_frontend_js_v2)
    const message = chatInput.value.trim();
    if (!message) { return; }
    addChatMessage(message, 'user');
    chatInput.value = '';
    showChatLoading();
    hideChatError();
    try {
        const response = await fetch(`${BACKEND_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', },
            body: JSON.stringify({ message: message }),
        });
        if (!response.ok) {
             let errorDetail = `Chat request failed (Status: ${response.status})`;
             try {
                const errorData = await response.json();
                errorDetail = errorData.detail || `Error ${response.status}: ${response.statusText}`;
            } catch (e) {
                 errorDetail = `Error ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorDetail);
        }
        const data = await response.json();
        addChatMessage(data.response, 'assistant');
    } catch (error) {
        console.error("Chat error:", error);
        showChatError(`Chat error: ${error.message}`);
        addChatMessage(`Sorry, I encountered an error: ${error.message}`, 'assistant', true);
    } finally {
        hideChatLoading();
    }
}


/**
 * Helper function to populate a list (UL) element with icons.
 * @param {HTMLElement} listElement - The UL element to populate.
 * @param {string[]} items - Array of strings to add as list items.
 * @param {HTMLElement} emptyMessageElement - The element to show if the items array is empty or null.
 * @param {string} emptyText - The text to display in the emptyMessageElement.
 * @param {string} [iconClass='fa-solid fa-check text-sky-600'] - Font Awesome classes for the list item icon.
 */
function populateList(listElement, items, emptyMessageElement, emptyText, iconClass = 'fa-solid fa-check text-sky-600') {
    // (Keep this function the same as in med_report_analyzer_frontend_js_v2)
    listElement.innerHTML = '';
    if (items && items.length > 0) {
        emptyMessageElement.classList.add('hidden');
        items.forEach(item => {
            const li = document.createElement('li');
            li.classList.add('flex', 'items-start', 'gap-2', 'text-slate-600');
            li.innerHTML = `<i class="${iconClass} icon mt-1 flex-shrink-0"></i><span>${item}</span>`;
            listElement.appendChild(li);
        });
    } else {
        emptyMessageElement.textContent = emptyText;
        emptyMessageElement.classList.remove('hidden');
    }
}

/**
 * Displays the analysis results received from the backend in the UI.
 * (Updated for simplified response structure)
 * @param {object} data - The analysis data object from the backend.
 */
function displayAnalysis(data) {
    // Detailed Analysis Section
    const detailedAnalysisText = data.detailed_analysis || 'No overall summary provided.';
    detailedAnalysisP.textContent = detailedAnalysisText;
    detailedAnalysisP.classList.toggle('italic', detailedAnalysisText === 'No overall summary provided.');
    detailedAnalysisP.parentElement.classList.toggle('hidden', !data.detailed_analysis);

    // Results Overview Section - REMOVED
    // No code needed here anymore

    // Potential Health Risks Section
    populateList(
        potentialRisksUl,
        data.potential_risks,
        potentialRisksEmpty,
        'No potential risks identified.',
        'fa-solid fa-triangle-exclamation text-orange-500' // Specific icon
    );

    // Recommendations Section (Simplified)
    // Now populates the single list using data.recommendations directly
    populateList(
        recommendationsListUl, // Target the single UL element
        data.recommendations, // Use the direct list from the updated API response
        recommendationsListEmpty,
        'No specific recommendations provided.',
        'fa-solid fa-check text-sky-600' // Use a consistent icon
    );

    // Raw LLM Response Section - REMOVED
    // No code needed here anymore
}


/**
 * Adds a message (user or assistant) to the chat history UI.
 * @param {string} message - The chat message text.
 * @param {'user' | 'assistant'} sender - Indicates who sent the message.
 * @param {boolean} [isError=false] - Optional flag to style assistant messages as errors.
 */
function addChatMessage(message, sender, isError = false) {
    // (Keep this function the same as in med_report_analyzer_frontend_js_v2)
    const placeholder = chatHistoryDiv.querySelector('p.text-slate-500');
    if (placeholder) { placeholder.remove(); }
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-bubble');
    messageDiv.classList.add(sender === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant');
    if (isError && sender === 'assistant') {
         messageDiv.classList.add('chat-bubble-error');
    }
    messageDiv.textContent = message;
    chatHistoryDiv.appendChild(messageDiv);
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
}


// --- UI State Management Functions ---
// (Keep all these functions the same as in med_report_analyzer_frontend_js_v2)
function showLoading() { loadingSpinner.classList.remove('hidden'); submitButton.disabled = true; }
function hideLoading() { loadingSpinner.classList.add('hidden'); submitButton.disabled = false; }
function showChatLoading() { chatLoadingSpinner.classList.remove('hidden'); sendChatButton.disabled = true; chatInput.disabled = true; }
function hideChatLoading() { chatLoadingSpinner.classList.add('hidden'); sendChatButton.disabled = false; chatInput.disabled = false; }
function showUploadError(message) { uploadError.textContent = message; uploadError.classList.remove('hidden'); }
function hideUploadError() { uploadError.classList.add('hidden'); }
function showChatError(message) { chatError.textContent = message; chatError.classList.remove('hidden'); }
function hideChatError() { chatError.classList.add('hidden'); }
function setUploadStatus(message) { if (message) { uploadStatus.textContent = message; uploadStatus.classList.remove('hidden'); } else { uploadStatus.classList.add('hidden'); } }
function showAnalysis() { analysisSection.classList.remove('hidden'); }
function hideAnalysis() { analysisSection.classList.add('hidden'); }
function showChat() { chatSection.classList.remove('hidden'); chatHistoryDiv.innerHTML = '<p class="text-slate-500 text-sm text-center italic">Chat history will appear here.</p>'; }
function hideChat() { chatSection.classList.add('hidden'); }

// --- Initial Page Load State ---
hideAnalysis();
hideChat();
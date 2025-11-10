// Global state
let currentVizId = null;
let sidePaneVizualizations = {};
let currentSidePaneTab = null;
let currentSubTab = 'chart';

// Initialize on page load
$(document).ready(function() {
    setupEventHandlers();
});

function setupEventHandlers() {
    // Chat controls
    $('#send-btn').click(sendMessage);
    $('#clear-btn').click(clearChat);
    $('#transfer-btn').click(transferVisualization);

    // Enter key in chat input
    $('#chat-input').keypress(function(e) {
        if (e.which === 13 && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Side pane chat
    $('#side-chat-send').click(sendSideChatMessage);

    // Sub-tabs
    $('#sub-tab-bar button').click(function() {
        const tab = $(this).data('tab');
        switchSubTab(tab);
    });
}

function sendMessage() {
    const message = $('#chat-input').val().trim();
    if (!message) return;

    // Add user message to chat
    appendChatMessage('user', message);
    $('#chat-input').val('');

    // Show thinking animation
    showThinkingAnimation();

    // Send to backend
    $.ajax({
        url: '/chat',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ message: message }),
        success: function(response) {
            // Remove thinking animation
            removeThinkingAnimation();

            // Debug logging
            console.log('Response received:', response);

            if (!response.success) {
                if (response.clarification) {
                    let clarificationHtml = `<strong>Need More Information:</strong><br><br>`;
                    clarificationHtml += `<div class="clarification-details">`;
                    clarificationHtml += `${response.clarification}<br><br>`;

                    // Add context details if available
                    if (response.context_type) {
                        clarificationHtml += `<strong>📝 Analysis Details:</strong><br>`;
                        clarificationHtml += `&nbsp;&nbsp;&nbsp;&nbsp;• Context type: ${response.context_type}<br>`;
                        if (response.reasoning) {
                            clarificationHtml += `&nbsp;&nbsp;&nbsp;&nbsp;• Reasoning: ${response.reasoning}<br>`;
                        }
                        if (response.suggestions && response.suggestions.length > 0) {
                            clarificationHtml += `&nbsp;&nbsp;&nbsp;&nbsp;• Try asking questions like: <br>`;
                            response.suggestions.forEach(suggestion => {
                                clarificationHtml += `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- "${suggestion}"<br>`;
                            });
                        }
                        if (response.query_issues && response.query_issues.length > 0) {
                            clarificationHtml += `&nbsp;&nbsp;&nbsp;&nbsp;• Issues identified: <br>`;
                            response.query_issues.forEach(issue => {
                                clarificationHtml += `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- ${issue}<br>`;
                            });
                        }
                    }
                    clarificationHtml += `</div>`;
                    appendChatMessage('assistant', clarificationHtml);
                } else {
                    appendChatMessage('assistant', response.error || 'Could not process request');
                }
                return;
            }

            // Build response message
            let responseText = '';

            if (response.sql) {
                responseText += `<strong>Generated SQL:</strong><br>`;
                responseText += `<div class="sql-code">${escapeHtml(response.sql)}</div><br>`;
            }

            if (response.no_data) {
                responseText += 'The query returned no data.';
                if (response.note) {
                    responseText += `<br><em>${response.note}</em>`;
                }
            } else {
                responseText += `<strong>Results:</strong> ${response.rows} rows, ${response.columns} columns<br><br>`;

                if (response.has_visualization) {
                    responseText += '<strong>Visualization created!</strong> You can send it to the side pane.<br><br>';

                    // Store current viz ID
                    currentVizId = response.viz_id;

                    // Show transfer button
                    $('#transfer-btn').show();

                    // Display visualization
                    displayVisualization(response.figure);
                }

                if (response.sample_data) {
                    responseText += '<strong>Sample data:</strong><br>';
                    responseText += response.sample_data;
                }
            }

            appendChatMessage('assistant', responseText);
        },
        error: function(xhr, status, error) {
            removeThinkingAnimation();
            appendChatMessage('assistant', `Error: ${error}`);
        }
    });
}

function appendChatMessage(role, content) {
    const messageClass = role === 'user' ? 'user-message' : 'assistant-message';
    const label = role === 'user' ? 'You' : 'AI';

    const messageHtml = `
        <div class="chat-message ${messageClass}">
            <strong>${label}:</strong> ${content}
        </div>
    `;

    // Remove welcome message if it exists
    $('.welcome-message').remove();

    $('#chat-display').append(messageHtml);
    $('#chat-display').scrollTop($('#chat-display')[0].scrollHeight);
}

function clearChat() {
    $('#chat-display').html(`
        <div class="welcome-message">
            <h4>Welcome to AI Data Assistant</h4>
            <p><strong>Try asking:</strong> "Show me the distribution of employees who have a title containing the word 'Sales'?"</p>
            <p>"Show me names of all the employees from the USA"</p>
        </div>
    `);
    $('#current-viz-container').empty();
    $('#transfer-btn').hide();
    currentVizId = null;
}

function displayVisualization(figure) {
    const container = document.getElementById('current-viz-container');
    container.style.height = '370px';
    Plotly.newPlot(container, figure.data, figure.layout);
}

function transferVisualization() {
    if (!currentVizId) return;

    $.ajax({
        url: '/transfer_visualization',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ viz_id: currentVizId }),
        success: function(response) {
            if (response.success) {
                // Add to side pane
                addVisualizationToSidePane(response.tab_name, response.viz_id);

                appendChatMessage('assistant', '<strong>Visualization sent to side pane!</strong>');
            }
        },
        error: function(xhr, status, error) {
            appendChatMessage('assistant', `Error transferring visualization: ${error}`);
        }
    });
}

function addVisualizationToSidePane(tabName, vizId) {
    // Store visualization reference
    sidePaneVizualizations[tabName] = vizId;

    // Update tab bar
    updateTabBar();

    // Switch to new tab
    switchToTab(tabName);
}

function updateTabBar() {
    const tabBar = $('#tab-bar');
    tabBar.empty();

    const tabs = Object.keys(sidePaneVizualizations);

    if (tabs.length === 0) {
        tabBar.html('<div class="no-charts">No charts</div>');
        return;
    }

    tabs.forEach(tabName => {
        const button = $(`<button class="btn btn-secondary">${tabName}</button>`);
        button.click(() => switchToTab(tabName));
        tabBar.append(button);
    });
}

function switchToTab(tabName) {
    currentSidePaneTab = tabName;
    currentSubTab = 'chart';

    // Update tab buttons
    $('#tab-bar button').removeClass('btn-primary').addClass('btn-secondary');
    $(`#tab-bar button:contains('${tabName}')`).removeClass('btn-secondary').addClass('btn-primary');

    // Show sub-tabs
    $('#sub-tab-bar').show();
    switchSubTab('chart');
}

function switchSubTab(tab) {
    currentSubTab = tab;

    // Update sub-tab buttons
    $('#sub-tab-bar button').removeClass('btn-primary').addClass('btn-secondary');
    $(`#sub-tab-bar button[data-tab='${tab}']`).removeClass('btn-secondary').addClass('btn-primary');

    // Load content for sub-tab
    loadSubTabContent();
}

function loadSubTabContent() {
    if (!currentSidePaneTab || !sidePaneVizualizations[currentSidePaneTab]) return;

    const vizId = sidePaneVizualizations[currentSidePaneTab];

    $.ajax({
        url: `/get_visualization/${vizId}`,
        method: 'GET',
        success: function(response) {
            if (!response.success) return;

            const contentArea = $('#side-content-area');
            const chatInterface = $('#side-chat-interface');

            if (currentSubTab === 'chart') {
                contentArea.show();
                chatInterface.hide();
                contentArea.empty();

                const chartDiv = $('<div>').attr('id', 'side-chart').css('height', '400px');
                contentArea.append(chartDiv);

                Plotly.newPlot('side-chart', response.chart.data, response.chart.layout);

            } else if (currentSubTab === 'sql') {
                contentArea.show();
                chatInterface.hide();
                contentArea.html(`
                    <div class="config-display">
                        <h4>SQL Query</h4>
                        <pre>${escapeHtml(response.sql)}</pre>
                        <p><strong>Original Question:</strong> ${response.original_question}</p>
                        <p><strong>Data Shape:</strong> ${response.data_shape}</p>
                    </div>
                `);

            } else if (currentSubTab === 'vis_config') {
                contentArea.show();
                chatInterface.hide();
                const configJson = JSON.stringify(response.vis_config, null, 2);
                contentArea.html(`
                    <div class="config-display">
                        <h4>Visualization Config</h4>
                        <pre>${escapeHtml(configJson)}</pre>
                    </div>
                `);

            } else if (currentSubTab === 'data') {
                contentArea.show();
                chatInterface.hide();
                contentArea.html(`
                    <div class="data-info">
                        <h4>Sample Data</h4>
                        <p><strong>Shape:</strong> ${response.data_shape}</p>
                        <p><strong>Columns:</strong> ${response.columns.join(', ')}</p>
                        <div class="table-container">
                            ${response.sample_data}
                        </div>
                    </div>
                `);

            } else if (currentSubTab === 'chat') {
                contentArea.hide();
                chatInterface.show();
                updateChatDisplay();
            }
        }
    });
}

function updateChatDisplay() {
    const vizId = sidePaneVizualizations[currentSidePaneTab];
    if (!vizId) {
        $('#side-chat-response').html('<p>Start a conversation about this visualization!</p>');
        return;
    }

    // Fetch conversation history from server
    $.ajax({
        url: `/get_conversation_history/${vizId}`,
        method: 'GET',
        success: function(response) {
            if (response.success) {
                const history = response.history;

                if (history.length === 0) {
                    $('#side-chat-response').html('<p>Start a conversation about this visualization!</p>');
                    return;
                }

                let html = '';
                history.forEach(entry => {
                    const cls = entry.type === 'user' ? 'user-entry' : 'assistant-entry';
                    const name = entry.type === 'user' ? 'You' : 'Assistant';
                    html += `<div class="conversation-entry ${cls}"><strong>${name}:</strong> ${entry.message}</div>`;
                });

                $('#side-chat-response').html(html);
                $('#side-chat-response').scrollTop($('#side-chat-response')[0].scrollHeight);
            }
        },
        error: function() {
            $('#side-chat-response').html('<p>Error loading conversation history.</p>');
        }
    });
}

function sendSideChatMessage() {
    const message = $('#side-chat-input').val().trim();
    if (!message || !currentSidePaneTab) return;

    const vizId = sidePaneVizualizations[currentSidePaneTab];
    if (!vizId) return;

    $('#side-chat-input').val('');

    // Show thinking animation in side chat
    showSideChatThinking();

    // Send modification request
    $.ajax({
        url: '/modify_visualization',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            viz_id: vizId,
            modification: message
        }),
        success: function(response) {
            // Remove side chat thinking animation
            removeSideChatThinking();

            // Update chat display to show new conversation
            updateChatDisplay();

            if (response.success && response.chart) {
                // Update chart if we're on the chart tab
                if (currentSubTab === 'chart') {
                    const contentArea = $('#side-content-area');
                    contentArea.empty();
                    const chartDiv = $('<div>').attr('id', 'side-chart').css('height', '400px');
                    contentArea.append(chartDiv);
                    Plotly.newPlot('side-chart', response.chart.data, response.chart.layout);
                }
            }
        },
        error: function(xhr, status, error) {
            // Remove side chat thinking animation
            removeSideChatThinking();

            // Refresh chat display to show any error messages
            updateChatDisplay();
        }
    });
}

function showThinkingAnimation() {
    const thinkingHtml = `
        <div class="chat-message assistant-message agent-thinking" id="thinking-indicator">
            <div class="thinking-text">Agent is thinking</div>
            <div class="thinking-dots">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    `;

    // Remove welcome message if it exists
    $('.welcome-message').remove();

    $('#chat-display').append(thinkingHtml);
    $('#chat-display').scrollTop($('#chat-display')[0].scrollHeight);
}

function removeThinkingAnimation() {
    $('#thinking-indicator').remove();
}

function showSideChatThinking() {
    const thinkingHtml = `
        <div class="conversation-entry assistant-entry" id="side-thinking-indicator">
            <div style="display: flex; align-items: center;">
                <span style="margin-right: 8px;"><strong>Assistant:</strong> Analysing your request</span>
                <div class="thinking-dots">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        </div>
    `;

    $('#side-chat-response').append(thinkingHtml);
    $('#side-chat-response').scrollTop($('#side-chat-response')[0].scrollHeight);
}

function removeSideChatThinking() {
    $('#side-thinking-indicator').remove();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
/**
 * Deep Research Agent - Web Dashboard JavaScript
 */

class DeepResearchApp {
    constructor() {
        this.apiBase = window.location.origin;
        this.currentTaskId = null;
        this.eventSource = null;

        this.elements = {
            queryInput: document.getElementById('queryInput'),
            formatSelect: document.getElementById('formatSelect'),
            researchBtn: document.getElementById('researchBtn'),
            progressSection: document.getElementById('progressSection'),
            progressBar: document.getElementById('progressBar'),
            progressPercent: document.getElementById('progressPercent'),
            progressStep: document.getElementById('progressStep'),
            reasoningSection: document.getElementById('reasoningSection'),
            reasoningTree: document.getElementById('reasoningTree'),
            resultsSection: document.getElementById('resultsSection'),
            resultsContent: document.getElementById('resultsContent'),
            confidenceBadge: document.getElementById('confidenceBadge'),
            sourceCount: document.getElementById('sourceCount'),
            copyBtn: document.getElementById('copyBtn'),
            downloadBtn: document.getElementById('downloadBtn'),
            newSearchBtn: document.getElementById('newSearchBtn'),
            statsBtn: document.getElementById('statsBtn'),
            clearBtn: document.getElementById('clearBtn'),
            statsModal: document.getElementById('statsModal'),
            statsContent: document.getElementById('statsContent'),
            closeStats: document.getElementById('closeStats')
        };

        this.bindEvents();
    }

    bindEvents() {
        // Research button
        this.elements.researchBtn.addEventListener('click', () => this.startResearch());

        // Enter key in textarea
        this.elements.queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.startResearch();
            }
        });

        // Copy button
        this.elements.copyBtn.addEventListener('click', () => this.copyResults());

        // Download button
        this.elements.downloadBtn.addEventListener('click', () => this.downloadResults());

        // New search button
        this.elements.newSearchBtn.addEventListener('click', () => this.resetUI());

        // Stats button
        this.elements.statsBtn.addEventListener('click', () => this.showStats());

        // Close stats modal
        this.elements.closeStats.addEventListener('click', () => this.hideStats());
        this.elements.statsModal.addEventListener('click', (e) => {
            if (e.target === this.elements.statsModal) {
                this.hideStats();
            }
        });

        // Clear button
        this.elements.clearBtn.addEventListener('click', () => this.clearMemory());
    }

    async startResearch() {
        const query = this.elements.queryInput.value.trim();
        if (!query) {
            this.showNotification('Please enter a research query', 'error');
            return;
        }

        const format = this.elements.formatSelect.value;

        // Show progress, hide results
        this.elements.progressSection.style.display = 'block';
        this.elements.resultsSection.style.display = 'none';
        this.elements.progressSection.classList.add('fade-in');

        // Disable button
        this.elements.researchBtn.disabled = true;
        this.elements.researchBtn.innerHTML = '<span class="spinner"></span> Researching...';

        try {
            // Start research
            const response = await fetch(`${this.apiBase}/research`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    format: format,
                    async_mode: false
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }

            const result = await response.json();
            this.currentTaskId = result.task_id;

            // Update progress to 100%
            this.updateProgress(1.0, 'Complete');

            // Show results
            this.showResults(result);

        } catch (error) {
            console.error('Research error:', error);
            this.showNotification(`Research failed: ${error.message}`, 'error');
            this.elements.progressSection.style.display = 'none';
        } finally {
            // Re-enable button
            this.elements.researchBtn.disabled = false;
            this.elements.researchBtn.innerHTML = '<span class="btn-icon">🔍</span> Start Research';
        }
    }

    updateProgress(progress, step) {
        const percent = Math.round(progress * 100);
        this.elements.progressBar.style.width = `${percent}%`;
        this.elements.progressPercent.textContent = `${percent}%`;
        this.elements.progressStep.textContent = step;
    }

    showResults(result) {
        // Hide progress
        this.elements.progressSection.style.display = 'none';

        // Show results section
        this.elements.resultsSection.style.display = 'block';
        this.elements.resultsSection.classList.add('fade-in');

        // Update metadata
        const confidence = result.confidence || 0;
        this.elements.confidenceBadge.textContent = `${Math.round(confidence * 100)}% Confidence`;
        this.elements.sourceCount.textContent = `${result.sources_count} Sources`;

        // Render content
        this.elements.resultsContent.innerHTML = this.renderMarkdown(result.content || 'No content available');
    }

    renderMarkdown(text) {
        // Simple markdown rendering
        let html = text
            // Headers
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            // Bold
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Code blocks
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            // Inline code
            .replace(/`(.*?)`/g, '<code>$1</code>')
            // Lists
            .replace(/^\s*[-•] (.*$)/gm, '<li>$1</li>')
            .replace(/^\d+\. (.*$)/gm, '<li>$1</li>')
            // Paragraphs
            .replace(/\n\n/g, '</p><p>')
            // Line breaks
            .replace(/\n/g, '<br>');

        // Wrap in paragraphs
        html = '<p>' + html + '</p>';

        // Fix list wrapping
        html = html.replace(/(<li>.*<\/li>)+/g, '<ul>$&</ul>');

        return html;
    }

    copyResults() {
        const content = this.elements.resultsContent.innerText;
        navigator.clipboard.writeText(content).then(() => {
            this.showNotification('Copied to clipboard!', 'success');
        }).catch(err => {
            this.showNotification('Failed to copy', 'error');
        });
    }

    downloadResults() {
        const content = this.elements.resultsContent.innerText;
        const format = this.elements.formatSelect.value;
        const ext = format === 'json' ? 'json' : 'md';
        const filename = `research_${Date.now()}.${ext}`;

        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();

        URL.revokeObjectURL(url);
        this.showNotification(`Downloaded ${filename}`, 'success');
    }

    resetUI() {
        this.elements.queryInput.value = '';
        this.elements.progressSection.style.display = 'none';
        this.elements.resultsSection.style.display = 'none';
        this.elements.progressBar.style.width = '0%';
        this.elements.progressPercent.textContent = '0%';
        this.currentTaskId = null;
    }

    async showStats() {
        this.elements.statsModal.style.display = 'flex';

        try {
            const response = await fetch(`${this.apiBase}/stats`);
            const stats = await response.json();

            this.elements.statsContent.innerHTML = `
                <div class="stat-group">
                    <h4>Reasoning Engine</h4>
                    <div class="stat-item">
                        <span>Tree Nodes</span>
                        <span class="stat-value">${stats.reasoning?.total_nodes || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span>Max Depth</span>
                        <span class="stat-value">${stats.reasoning?.max_depth || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span>Explorations</span>
                        <span class="stat-value">${stats.reasoning?.explorations || 0}</span>
                    </div>
                </div>
                
                <div class="stat-group">
                    <h4>Memory System</h4>
                    <div class="stat-item">
                        <span>Short-term Entries</span>
                        <span class="stat-value">${stats.memory?.short_term?.total_entries || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span>Long-term Entries</span>
                        <span class="stat-value">${stats.memory?.long_term?.total_entries || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span>Citations</span>
                        <span class="stat-value">${stats.memory?.citations?.total_sources || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span>Tracked Failures</span>
                        <span class="stat-value">${stats.memory?.failures?.total_failures || 0}</span>
                    </div>
                </div>
            `;
        } catch (error) {
            this.elements.statsContent.innerHTML = `<p>Failed to load stats: ${error.message}</p>`;
        }
    }

    hideStats() {
        this.elements.statsModal.style.display = 'none';
    }

    async clearMemory() {
        if (!confirm('Clear all memory? This cannot be undone.')) {
            return;
        }

        try {
            await fetch(`${this.apiBase}/clear-memory`, { method: 'POST' });
            this.showNotification('Memory cleared', 'success');
        } catch (error) {
            this.showNotification('Failed to clear memory', 'error');
        }
    }

    showNotification(message, type = 'info') {
        // Simple notification (could be enhanced with a toast library)
        const colors = {
            success: '#10b981',
            error: '#ef4444',
            info: '#6366f1'
        };

        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            padding: 1rem 1.5rem;
            background: ${colors[type] || colors.info};
            color: white;
            border-radius: 8px;
            font-weight: 500;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            z-index: 2000;
            animation: fadeIn 0.3s ease;
        `;
        notification.textContent = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transition = 'opacity 0.3s';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DeepResearchApp();
});

class CryoEMApp {
    constructor() {
        this.selectedFile = null;
        this.currentTask = null;
        this.resultImage = null;
        this.heatmapImage = null;
        this.particles = [];
        this.imageScale = 1;
        this.showHeatmap = true;
        this.showBoxes = true;
        this.heatmapOpacity = 0.5;
        this.selectedParticleIndex = -1;
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadSystemStatus();
        this.loadTasks();
    }

    bindEvents() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                this.handleFileSelect(e.dataTransfer.files[0]);
            }
        });
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });
        document.getElementById('confidenceThreshold').addEventListener('input', (e) => {
            document.getElementById('confidenceValue').textContent = e.target.value;
        });
        document.getElementById('clearFileBtn').addEventListener('click', () => this.clearFile());
        document.getElementById('uploadBtn').addEventListener('click', () => this.uploadAndPick());
        document.getElementById('refreshBtn').addEventListener('click', () => this.loadTasks());
        document.getElementById('showHeatmap').addEventListener('change', (e) => {
            this.showHeatmap = e.target.checked;
            this.renderCanvas();
        });
        document.getElementById('showBoxes').addEventListener('change', (e) => {
            this.showBoxes = e.target.checked;
            this.renderCanvas();
        });
        document.getElementById('heatmapOpacity').addEventListener('input', (e) => {
            this.heatmapOpacity = parseFloat(e.target.value);
            this.renderCanvas();
        });
        document.getElementById('zoomLevel').addEventListener('input', (e) => {
            this.imageScale = parseFloat(e.target.value);
            document.getElementById('zoomValue').textContent = Math.round(this.imageScale * 100) + '%';
            this.renderCanvas();
        });
        document.getElementById('downloadImageBtn').addEventListener('click', () => this.downloadResultImage());
        document.getElementById('downloadCoordsBtn').addEventListener('click', () => this.downloadCoordinates());
        document.getElementById('downloadProbBtn').addEventListener('click', () => this.downloadProbabilityMap());
    }

    async loadSystemStatus() {
        try {
            const response = await fetch('/api/system/info');
            const data = await response.json();
            const serviceStatus = document.getElementById('serviceStatus');
            const modelStatus = document.getElementById('modelStatus');
            const gpuStatus = document.getElementById('gpuStatus');
            serviceStatus.textContent = '在线';
            serviceStatus.className = 'status-value online';
            if (data.model_loaded) {
                modelStatus.textContent = '已加载';
                modelStatus.className = 'status-value online';
            } else {
                modelStatus.textContent = '未加载';
                modelStatus.className = 'status-value warning';
            }
            if (data.gpu_available) {
                gpuStatus.textContent = `可用 (${data.gpu_memory_gb?.toFixed(1) || '?'} GB)`;
                gpuStatus.className = 'status-value online';
            } else {
                gpuStatus.textContent = 'CPU模式';
                gpuStatus.className = 'status-value warning';
            }
        } catch (error) {
            console.error('Failed to load system status:', error);
            const statusElements = ['serviceStatus', 'modelStatus', 'gpuStatus'];
            statusElements.forEach(id => {
                const el = document.getElementById(id);
                el.textContent = '连接失败';
                el.className = 'status-value offline';
            });
        }
    }

    async loadTasks() {
        try {
            const response = await fetch('/api/pick/tasks?limit=50');
            const tasks = await response.json();
            this.renderTasks(tasks);
        } catch (error) {
            console.error('Failed to load tasks:', error);
            this.showToast('加载任务列表失败', 'error');
        }
    }

    renderTasks(tasks) {
        const container = document.getElementById('tasksList');
        if (tasks.length === 0) {
            container.innerHTML = '<p class="empty-state">暂无任务</p>';
            return;
        }
        container.innerHTML = tasks.map(task => `
            <div class="task-item ${this.currentTask?.task_id === task.task_id ? 'selected' : ''}"
                 data-task-id="${task.task_id}">
                <div class="task-info">
                    <div class="task-name">${this.escapeHtml(task.file_name)}</div>
                    <div class="task-meta">
                        <span>尺寸: ${task.image_info.width}x${task.image_info.height}</span>
                        ${task.num_particles > 0 ? `<span>颗粒: ${task.num_particles}</span>` : ''}
                        <span>${this.formatDate(task.created_at)}</span>
                    </div>
                </div>
                <span class="task-status status-${task.status}">${this.getStatusText(task.status)}</span>
            </div>
        `).join('');
        container.querySelectorAll('.task-item').forEach(el => {
            el.addEventListener('click', () => {
                const taskId = el.dataset.taskId;
                this.selectTask(taskId);
            });
        });
        const processingTasks = tasks.filter(t => t.status === 'pending' || t.status === 'processing');
        if (processingTasks.length > 0) {
            setTimeout(() => this.loadTasks(), 3000);
        }
    }

    async selectTask(taskId) {
        try {
            const response = await fetch(`/api/pick/tasks/${taskId}`);
            const task = await response.json();
            this.currentTask = task;
            this.renderTasks(await this.fetchTasks());
            if (task.status === 'completed') {
                document.getElementById('resultsSection').hidden = false;
                this.loadResult(task);
            } else if (task.status === 'failed') {
                this.showToast(`任务失败: ${task.error_message || '未知错误'}`, 'error');
            } else {
                document.getElementById('resultsSection').hidden = true;
                this.showToast('任务处理中，请稍候...');
                setTimeout(() => this.selectTask(taskId), 2000);
            }
        } catch (error) {
            console.error('Failed to load task:', error);
            this.showToast('加载任务失败', 'error');
        }
    }

    async fetchTasks() {
        const response = await fetch('/api/pick/tasks?limit=50');
        return await response.json();
    }

    async loadResult(task) {
        this.particles = task.particles || [];
        document.getElementById('totalParticles').textContent = task.num_particles;
        const avgConfidence = this.particles.length > 0
            ? this.particles.reduce((sum, p) => sum + p.score, 0) / this.particles.length
            : 0;
        document.getElementById('avgConfidence').textContent = avgConfidence.toFixed(2);
        if (task.processing_times) {
            document.getElementById('processingTime').textContent = task.processing_times.total.toFixed(2) + 's';
        }
        this.renderParticlesTable();
        const imageUrl = task.result_image_url;
        const probUrl = task.probability_map_url;
        await Promise.all([
            this.loadImage(`/api${imageUrl}`).then(img => { this.resultImage = img; }),
            this.loadImage(`/api${probUrl}`).then(img => { this.heatmapImage = img; })
        ]);
        this.renderCanvas();
    }

    async loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });
    }

    renderCanvas() {
        const canvas = document.getElementById('resultCanvas');
        if (!this.resultImage) return;
        const width = this.resultImage.width * this.imageScale;
        const height = this.resultImage.height * this.imageScale;
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, width, height);
        ctx.drawImage(this.resultImage, 0, 0, width, height);
        if (this.showHeatmap && this.heatmapImage) {
            ctx.globalAlpha = this.heatmapOpacity;
            ctx.drawImage(this.heatmapImage, 0, 0, width, height);
            ctx.globalAlpha = 1;
        }
        if (this.showBoxes && this.particles.length > 0) {
            this.particles.forEach((p, index) => {
                const x = p.x * this.imageScale;
                const y = p.y * this.imageScale;
                const size = (p.radius || 32) * this.imageScale;
                ctx.strokeStyle = index === this.selectedParticleIndex
                    ? '#ff6b6b'
                    : `hsl(${120 * p.score}, 100%, 50%)`;
                ctx.lineWidth = index === this.selectedParticleIndex ? 3 : 2;
                ctx.strokeRect(x - size / 2, y - size / 2, size, size);
                ctx.beginPath();
                ctx.moveTo(x - 10, y);
                ctx.lineTo(x + 10, y);
                ctx.moveTo(x, y - 10);
                ctx.lineTo(x, y + 10);
                ctx.stroke();
            });
        }
    }

    renderParticlesTable() {
        const tbody = document.getElementById('particlesTableBody');
        const displayParticles = this.particles.slice(0, 100);
        tbody.innerHTML = displayParticles.map((p, index) => `
            <tr class="${index === this.selectedParticleIndex ? 'selected' : ''}"
                data-index="${index}">
                <td>${index + 1}</td>
                <td>${p.x.toFixed(1)}</td>
                <td>${p.y.toFixed(1)}</td>
                <td class="${this.getConfidenceClass(p.score)}">${(p.score * 100).toFixed(1)}%</td>
                <td>${p.radius ? p.radius.toFixed(1) : '-'}</td>
                <td>${p.snr ? p.snr.toFixed(2) : '-'}</td>
            </tr>
        `).join('');
        tbody.querySelectorAll('tr').forEach(tr => {
            tr.addEventListener('click', () => {
                this.selectedParticleIndex = parseInt(tr.dataset.index);
                this.renderParticlesTable();
                this.renderCanvas();
            });
        });
    }

    getConfidenceClass(score) {
        if (score >= 0.8) return 'confidence-high';
        if (score >= 0.5) return 'confidence-medium';
        return 'confidence-low';
    }

    handleFileSelect(file) {
        const allowedTypes = ['.mrc', '.mrcs', '.tif', '.tiff'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowedTypes.includes(ext)) {
            this.showToast(`不支持的文件格式: ${ext}`, 'error');
            return;
        }
        if (file.size > 1024 * 1024 * 1024) {
            this.showToast('文件大小超过 1GB 限制', 'error');
            return;
        }
        this.selectedFile = file;
        document.getElementById('fileInfo').hidden = false;
        document.getElementById('selectedFileName').textContent = file.name;
        document.getElementById('selectedFileSize').textContent = this.formatFileSize(file.size);
        document.getElementById('uploadBtn').disabled = false;
    }

    clearFile() {
        this.selectedFile = null;
        document.getElementById('fileInput').value = '';
        document.getElementById('fileInfo').hidden = true;
        document.getElementById('uploadBtn').disabled = true;
    }

    async uploadAndPick() {
        if (!this.selectedFile) return;
        const uploadBtn = document.getElementById('uploadBtn');
        const loader = document.getElementById('uploadLoader');
        uploadBtn.disabled = true;
        loader.hidden = false;
        try {
            const formData = new FormData();
            formData.append('file', this.selectedFile);
            const params = new URLSearchParams();
            params.append('confidence_threshold', document.getElementById('confidenceThreshold').value);
            params.append('min_distance', document.getElementById('minDistance').value);
            params.append('export_format', document.getElementById('exportFormat').value);
            const response = await fetch(`/api/pick/upload?${params.toString()}`, {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || '上传失败');
            }
            const result = await response.json();
            this.showToast(`任务已创建: ${result.task_id}`);
            this.clearFile();
            await this.loadTasks();
            setTimeout(() => this.selectTask(result.task_id), 500);
        } catch (error) {
            console.error('Upload failed:', error);
            this.showToast(`上传失败: ${error.message}`, 'error');
        } finally {
            uploadBtn.disabled = false;
            loader.hidden = true;
        }
    }

    downloadResultImage() {
        if (!this.currentTask) return;
        window.open(`/api${this.currentTask.result_image_url}`, '_blank');
    }

    async downloadCoordinates() {
        if (!this.currentTask) return;
        const format = document.getElementById('exportFormat').value;
        try {
            const response = await fetch(`/api/results/${this.currentTask.task_id}/export`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    task_id: this.currentTask.task_id,
                    format: format,
                    include_quality: true
                })
            });
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${this.currentTask.task_id}_coordinates.${format}`;
            a.click();
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Download failed:', error);
            this.showToast('下载失败', 'error');
        }
    }

    downloadProbabilityMap() {
        if (!this.currentTask) return;
        window.open(`/api${this.currentTask.probability_map_url}`, '_blank');
    }

    getStatusText(status) {
        const map = {
            pending: '等待中',
            processing: '处理中',
            completed: '已完成',
            failed: '失败'
        };
        return map[status] || status;
    }

    formatDate(dateStr) {
        const date = new Date(dateStr);
        return date.toLocaleString('zh-CN', {
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
        if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
        return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showToast(message, type = 'success') {
        const toast = document.getElementById('toast');
        const toastMessage = document.getElementById('toastMessage');
        toastMessage.textContent = message;
        toast.className = `toast ${type}`;
        toast.hidden = false;
        setTimeout(() => {
            toast.hidden = true;
        }, 3000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new CryoEMApp();
});

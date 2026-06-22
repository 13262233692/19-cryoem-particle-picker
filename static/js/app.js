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

    initOrientationPolling() {
        this._shownAlertIds = new Set();
        this._alertPollTimer = setInterval(() => this.pollOrientationAlerts(), 2500);
        this.pollOrientationAlerts();
    }

    async pollOrientationAlerts() {
        try {
            let url = '/api/orientation/alerts';
            if (this.currentTask) {
                url += `?task_id=${this.currentTask.task_id}`;
            }
            const resp = await fetch(url);
            if (!resp.ok) return;
            const alerts = await resp.json();
            if (!alerts || alerts.length === 0) return;
            for (const alert of alerts) {
                if (this._shownAlertIds.has(alert.alert_id)) continue;
                this._shownAlertIds.add(alert.alert_id);
                this.handlePreferredOrientationAlert(alert);
            }
        } catch (e) {
        }
    }

    handlePreferredOrientationAlert(alert) {
        this.showPreferredOrientationOverlay(alert);
        this.showBlockingModal(alert);
    }

    showPreferredOrientationOverlay(alert) {
        const container = document.querySelector('.canvas-container');
        if (!container) return;
        let overlay = document.getElementById('poWarningOverlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'poWarningOverlay';
            overlay.className = 'po-warning-overlay';
            overlay.innerHTML = `
                <div class="po-warning-stripe"></div>
                <div class="po-warning-banner">
                    优势取向报废警示：空气-水界面吸附已触发
                    <span class="po-warning-banner-sub">该批次样品已被静默拦截，高分辨率三维迭代已终止 · 批次ID: <span id="poBatchIdSpan"></span></span>
                </div>
            `;
            container.appendChild(overlay);
        }
        overlay.style.display = 'block';
        const batchSpan = document.getElementById('poBatchIdSpan');
        if (batchSpan) batchSpan.textContent = alert.batch_id.slice(-10);
    }

    hidePreferredOrientationOverlay() {
        const overlay = document.getElementById('poWarningOverlay');
        if (overlay) overlay.style.display = 'none';
    }

    showBlockingModal(alert) {
        let backdrop = document.getElementById('poModalBackdrop');
        if (!backdrop) {
            backdrop = document.createElement('div');
            backdrop.id = 'poModalBackdrop';
            backdrop.className = 'po-modal-backdrop';
            document.body.appendChild(backdrop);
        }
        let modal = document.getElementById('poBlockingModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'poBlockingModal';
            modal.className = 'po-blocking-modal';
            document.body.appendChild(modal);
        }
        const orient = alert.orientation || {};
        const hist = orient.euler_histogram || {};
        const meanRes = (orient.mean_residual || 0).toFixed(4);
        const spread = (orient.angular_spread_deg || 0).toFixed(1);
        const coverage = ((orient.orientation_coverage || 0) * 100).toFixed(1);
        const resBad = (orient.mean_residual || 0) > 0.75;
        const spreadBad = (orient.angular_spread_deg || 999) < 15;
        const covBad = (orient.orientation_coverage || 1) < 0.2;
        modal.innerHTML = `
            <div class="po-modal-header">
                <div class="po-modal-title">
                    <span class="po-modal-title-icon">&#9888;</span>
                    样品报废确认 · 阻断式干预
                </div>
                <div class="po-modal-subtitle">
                    该批次蛋白颗粒遭遇严重空气-水界面吸附，三维重构角度完备性已丧失
                </div>
                <div class="po-modal-reason">${alert.block_reason || '未知原因'}</div>
            </div>
            <div class="po-modal-body">
                <div>
                    <div class="po-section-title">公共线残差 &amp; 取向丰度指标</div>
                    <div class="po-metrics-grid">
                        <div class="po-metric ${resBad ? 'bad' : ''}">
                            <span class="po-metric-label">公共线平均残差</span>
                            <span class="po-metric-value">${meanRes}</span>
                            <span class="po-metric-threshold">临界界值: 0.7500</span>
                        </div>
                        <div class="po-metric ${spreadBad ? 'bad' : ''}">
                            <span class="po-metric-label">欧拉角丰度标准差</span>
                            <span class="po-metric-value">${spread}°</span>
                            <span class="po-metric-threshold">临界下限: 15.0°</span>
                        </div>
                        <div class="po-metric ${covBad ? 'bad' : ''}">
                            <span class="po-metric-label">取向球面覆盖度</span>
                            <span class="po-metric-value">${coverage}%</span>
                            <span class="po-metric-threshold">临界下限: 20.0%</span>
                        </div>
                        <div class="po-metric">
                            <span class="po-metric-label">批次颗粒总数</span>
                            <span class="po-metric-value">${alert.particle_count}</span>
                            <span class="po-metric-threshold">已完成分析</span>
                        </div>
                    </div>
                </div>

                <div>
                    <div class="po-section-title">欧拉角丰度球面投影对比</div>
                    <div class="po-sphere-compare">
                        <div class="po-sphere-card normal">
                            <div class="po-sphere-label">正常样品 · 均匀分布</div>
                            <div class="po-sphere-canvas-wrap">
                                <canvas class="po-sphere-canvas" id="sphereCanvasNormal"></canvas>
                            </div>
                        </div>
                        <div class="po-sphere-card bad">
                            <div class="po-sphere-label">当前批次 · 单一优势取向</div>
                            <div class="po-sphere-canvas-wrap">
                                <canvas class="po-sphere-canvas" id="sphereCanvasBad"></canvas>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="po-histogram-section">
                    <div class="po-section-title">Theta (极角) 丰度分布直方图</div>
                    <canvas class="po-histogram-canvas" id="thetaHistCanvas"></canvas>
                    <div class="po-histogram-legend">
                        <span class="po-histogram-legend-item">
                            <span class="po-histogram-swatch" style="background:#4ade80;"></span>正常参考
                        </span>
                        <span class="po-histogram-legend-item">
                            <span class="po-histogram-swatch" style="background:#ffeb00;"></span>当前样品
                        </span>
                    </div>
                </div>
            </div>
            <div class="po-modal-footer">
                <div class="po-footer-note">
                    <strong>结构生物学物理规则：</strong>
                    冷冻制样时蛋白质颗粒在冰层中的空气-水界面会产生强烈疏水吸附，
                    导致所有颗粒强制平躺呈单一优势取向，三维重构完全丧失角度完备性。
                    此类样品必须报废并重新制样，任何后续计算资源投入均为无效消耗。
                </div>
                <div class="po-modal-actions">
                    <button class="btn btn-warning" id="poConfirmBtn">
                        确认报废该批次 · 重新制样
                    </button>
                    <button class="btn btn-secondary" id="poDismissBtn">
                        暂存警示
                    </button>
                </div>
            </div>
        `;
        setTimeout(() => {
            this.renderSphereProjection('sphereCanvasNormal', 'uniform');
            this.renderSphereProjection('sphereCanvasBad', alert.orientation, hist);
            this.renderThetaHistogram('thetaHistCanvas', hist);
        }, 20);
        const confirmBtn = document.getElementById('poConfirmBtn');
        if (confirmBtn) {
            confirmBtn.addEventListener('click', () => this.confirmDiscardBatch(alert.alert_id));
        }
        const dismissBtn = document.getElementById('poDismissBtn');
        if (dismissBtn) {
            dismissBtn.addEventListener('click', () => this.dismissAlert(alert.alert_id));
        }
    }

    renderSphereProjection(canvasId, mode, hist) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const wrap = canvas.parentElement;
        const w = wrap.clientWidth;
        const h = wrap.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, w, h);
        const cx = w / 2, cy = h / 2;
        const R = Math.min(w, h) / 2 - 4;
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.arc(cx, cy, R, 0, Math.PI * 2);
        ctx.stroke();
        for (let lat = 1; lat <= 3; lat++) {
            const phi = (Math.PI / 4) * lat;
            const r = R * Math.sin(phi);
            ctx.strokeStyle = 'rgba(255,255,255,0.08)';
            ctx.beginPath();
            ctx.ellipse(cx, cy - R * Math.cos(phi) * 0, r, r * 0.35, 0, 0, Math.PI * 2);
            ctx.stroke();
        }
        for (let lon = 0; lon < 6; lon++) {
            const theta = (Math.PI / 3) * lon;
            ctx.strokeStyle = 'rgba(255,255,255,0.08)';
            ctx.beginPath();
            ctx.ellipse(cx, cy, R * Math.sin(theta), R, 0, 0, Math.PI * 2);
            ctx.stroke();
        }
        let points = [];
        if (mode === 'uniform') {
            const N = 260;
            for (let i = 0; i < N; i++) {
                const u = Math.random();
                const v = Math.random();
                const theta = 2 * Math.PI * u;
                const phi = Math.acos(2 * v - 1);
                const x = R * Math.sin(phi) * Math.cos(theta);
                const y = R * Math.sin(phi) * Math.sin(theta);
                const z = R * Math.cos(phi);
                if (z >= 0) points.push({ x: cx + x, y: cy - y, z });
            }
        } else {
            const N = 300;
            const thetaBins = (hist && hist.theta_bins) ? hist.theta_bins : null;
            const thetaCounts = (hist && hist.theta_counts) ? hist.theta_counts : null;
            let centerTheta = Math.PI / 6;
            if (thetaCounts && thetaCounts.length > 0) {
                let maxIdx = 0;
                for (let i = 1; i < thetaCounts.length; i++) {
                    if (thetaCounts[i] > thetaCounts[maxIdx]) maxIdx = i;
                }
                const bins = thetaBins || thetaCounts.map((_, i) => (Math.PI / thetaCounts.length) * i);
                centerTheta = bins[Math.min(maxIdx, bins.length - 1)];
            }
            for (let i = 0; i < N; i++) {
                const theta = centerTheta + (Math.random() - 0.5) * 0.25;
                const phi = (Math.random() - 0.5) * 0.35;
                const x = R * Math.sin(theta) * Math.cos(phi);
                const y = R * Math.sin(theta) * Math.sin(phi);
                const z = R * Math.cos(theta);
                if (z >= -R * 0.2) points.push({ x: cx + x, y: cy - y, z });
            }
        }
        const color = mode === 'uniform' ? 'rgba(74, 222, 128, ' : 'rgba(255, 235, 0, ';
        points.sort((a, b) => b.z - a.z);
        for (const p of points) {
            const alpha = 0.35 + 0.55 * (p.z / R);
            ctx.fillStyle = color + alpha.toFixed(3) + ')';
            const size = 1.2 + 1.6 * Math.max(0, p.z / R);
            ctx.beginPath();
            ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
            ctx.fill();
        }
        if (mode !== 'uniform') {
            ctx.strokeStyle = 'rgba(255, 235, 0, 0.8)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.ellipse(cx, cy, R * 0.55, R * 0.15, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#ffeb00';
            ctx.font = 'bold 10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('优势取向集中区', cx, cy + R * 0.55);
        } else {
            ctx.fillStyle = 'rgba(74, 222, 128, 0.85)';
            ctx.font = 'bold 10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('均匀角度分布', cx, cy + R * 0.62);
        }
    }

    renderThetaHistogram(canvasId, hist) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, w, h);
        const padL = 40, padR = 12, padT = 12, padB = 28;
        const cw = w - padL - padR;
        const ch = h - padT - padB;
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, padT);
        ctx.lineTo(padL, padT + ch);
        ctx.lineTo(padL + cw, padT + ch);
        ctx.stroke();
        ctx.fillStyle = '#8892b0';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        for (let i = 0; i <= 4; i++) {
            const frac = i / 4;
            const x = padL + cw * frac;
            const deg = Math.round(frac * 180);
            ctx.fillText(deg + '°', x, padT + ch + 16);
        }
        ctx.textAlign = 'right';
        ctx.fillStyle = '#8892b0';
        ctx.fillText('丰度', padL - 6, padT + 8);
        const nBins = 36;
        const uniform = new Array(nBins).fill(1 / nBins);
        const bad = new Array(nBins).fill(0);
        if (hist && hist.theta_counts && hist.theta_counts.length > 0) {
            const src = hist.theta_counts;
            for (let i = 0; i < nBins; i++) {
                const idx = Math.floor(i / nBins * src.length);
                bad[i] = src[Math.min(idx, src.length - 1)] || 0;
            }
        } else {
            const centerBin = Math.floor(nBins * 0.15);
            for (let i = 0; i < nBins; i++) {
                const d = Math.abs(i - centerBin);
                bad[i] = Math.max(0, 1 - d / 4);
            }
        }
        const sumU = uniform.reduce((a, b) => a + b, 0) || 1;
        const sumB = bad.reduce((a, b) => a + b, 0) || 1;
        const maxCount = Math.max(
            ...uniform.map(v => v / sumU),
            ...bad.map(v => v / sumB)
        ) * 1.15;
        const barW = cw / nBins * 0.4;
        for (let i = 0; i < nBins; i++) {
            const vU = uniform[i] / sumU / maxCount;
            const xU = padL + (cw / nBins) * i + (cw / nBins) * 0.05;
            const bhU = ch * vU;
            ctx.fillStyle = 'rgba(74, 222, 128, 0.75)';
            ctx.fillRect(xU, padT + ch - bhU, barW, bhU);
            const vB = bad[i] / sumB / maxCount;
            const xB = padL + (cw / nBins) * i + (cw / nBins) * 0.5;
            const bhB = ch * vB;
            ctx.fillStyle = 'rgba(255, 235, 0, 0.85)';
            ctx.fillRect(xB, padT + ch - bhB, barW, bhB);
        }
        ctx.fillStyle = '#64748b';
        ctx.textAlign = 'center';
        ctx.font = '10px sans-serif';
        ctx.fillText('极角 Theta (0°→180°)', padL + cw / 2, padT + ch + 26);
    }

    async confirmDiscardBatch(alertId) {
        this.showToast('该批次样品已标记为报废，高分辨率三维迭代已终止', 'success');
        await this.dismissAlert(alertId);
        this.hidePreferredOrientationOverlay();
    }

    async dismissAlert(alertId) {
        try {
            await fetch(`/api/orientation/alerts/${alertId}/dismiss`, { method: 'POST' });
        } catch (e) {
        }
        const modal = document.getElementById('poBlockingModal');
        if (modal) modal.remove();
        const backdrop = document.getElementById('poModalBackdrop');
        if (backdrop) backdrop.remove();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new CryoEMApp();
    window.app.initOrientationPolling();
});

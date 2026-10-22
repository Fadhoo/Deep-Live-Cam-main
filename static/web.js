const form = document.querySelector('#swap-form');
const videoInput = document.querySelector('#video-input');
const faceInput = document.querySelector('#face-input');
const consent = document.querySelector('#consent');
const strength = document.querySelector('#effect-strength');
const strengthValue = document.querySelector('#strength-value');
const previewTime = document.querySelector('#preview-time');
const previewButton = document.querySelector('#preview-button');
const previewHint = document.querySelector('#preview-hint');
const submitButton = document.querySelector('#submit-button');
const panel = document.querySelector('#job-panel');
const stage = document.querySelector('#job-stage');
const message = document.querySelector('#job-message');
const indicator = document.querySelector('#job-indicator');
const progress = document.querySelector('#progress-track');
const download = document.querySelector('#download-link');
const resultVideo = document.querySelector('#result-video');
const cancelButton = document.querySelector('#cancel-button');
const deleteButton = document.querySelector('#delete-button');
const savedResultsList = document.querySelector('#saved-results-list');
const details = document.querySelector('#job-details');
const logs = document.querySelector('#job-logs');
const comparePanel = document.querySelector('#compare-panel');
const compareFrame = document.querySelector('#compare-frame');
const originalFrame = document.querySelector('#original-frame');
const swappedFrame = document.querySelector('#swapped-frame');
const resultMode = document.querySelector('#result-mode');
const compareMode = document.querySelector('#compare-mode');
const splitControl = document.querySelector('#split-control');
const splitRange = document.querySelector('#split-range');
const videoPreview = document.querySelector('#video-preview');

let jobId = localStorage.getItem('deep-live-cam-job');
let jobStatus = null;
let busy = false;
let videoUrl;
let faceUrl;
const savedStrength = Number(sessionStorage.getItem('deep-live-cam-strength'));
const savedTime = Number(sessionStorage.getItem('deep-live-cam-time'));
if (sessionStorage.getItem('deep-live-cam-strength') !== null &&
    Number.isInteger(savedStrength) && savedStrength >= 0 && savedStrength <= 100) {
  strength.value = String(savedStrength);
  strengthValue.value = `${savedStrength}%`;
}
if (sessionStorage.getItem('deep-live-cam-time') !== null &&
    Number.isFinite(savedTime) && savedTime >= 0 && savedTime <= 36000) {
  previewTime.value = String(savedTime);
}

function updateReady() {
  const staged = Boolean(jobId && jobStatus === 'ready');
  const active = ['queued', 'processing', 'previewing'].includes(jobStatus);
  videoInput.required = !staged;
  faceInput.required = !staged;
  const filesReady = (videoInput.files.length && faceInput.files.length) || staged;
  const canAct = Boolean(filesReady && consent.checked && !busy && !active && jobStatus !== 'complete');
  submitButton.disabled = !canAct;
  previewButton.disabled = !canAct;
  videoInput.disabled = busy || active;
  faceInput.disabled = busy || active;
  strength.disabled = busy || active;
  previewTime.disabled = busy || active;
}

function invalidatePreview() {
  comparePanel.hidden = true;
  previewHint.textContent = 'Preview a frame to see this blend before creating the full video.';
}

function invalidateJob(discardStaged = true) {
  const oldId = jobId;
  const oldStatus = jobStatus;
  jobId = null;
  jobStatus = null;
  localStorage.removeItem('deep-live-cam-job');
  panel.hidden = true;
  resultVideo.removeAttribute('src');
  resultVideo.hidden = true;
  invalidatePreview();
  if (discardStaged && oldId && ['ready', 'failed'].includes(oldStatus)) {
    fetch(`/api/jobs/${encodeURIComponent(oldId)}`, {
      method: 'DELETE', headers: {'X-Requested-With': 'Deep-Live-Cam'},
    }).catch(() => {});
  }
}

async function refreshSavedResults() {
  try {
    const jobs = await readJson(await fetch('/api/jobs'));
    savedResultsList.replaceChildren();
    if (!jobs.length) {
      const empty = document.createElement('p');
      empty.textContent = 'Finished videos will appear here until you delete them.';
      savedResultsList.append(empty);
      return;
    }
    for (const job of jobs) {
      const row = document.createElement('div');
      row.className = 'saved-result';
      const label = document.createElement('span');
      label.textContent = new Date(job.created_at * 1000).toLocaleString();
      const link = document.createElement('a');
      link.href = job.download_url;
      link.textContent = 'Download';
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.textContent = 'Delete';
      remove.setAttribute('aria-label', `Delete result from ${label.textContent}`);
      remove.addEventListener('click', async () => {
        remove.disabled = true;
        try {
          const response = await fetch(`/api/jobs/${encodeURIComponent(job.id)}`, {
            method: 'DELETE', headers: {'X-Requested-With': 'Deep-Live-Cam'},
          });
          if (!response.ok) throw new Error('Could not delete this result.');
          if (jobId === job.id) invalidateJob(false);
          await refreshSavedResults();
        } catch (error) {
          remove.disabled = false;
          showError(error);
        }
      });
      row.append(label, link, remove);
      savedResultsList.append(row);
    }
  } catch (error) {
    savedResultsList.textContent = `Could not load saved results: ${error.message}`;
  }
}

function preview(input, kind) {
  const file = input.files[0];
  const element = document.querySelector(`#${kind}-preview`);
  const placeholder = document.querySelector(`#${kind}-placeholder`);
  const name = document.querySelector(`#${kind}-name`);
  if (kind === 'video' && videoUrl) URL.revokeObjectURL(videoUrl);
  if (kind === 'face' && faceUrl) URL.revokeObjectURL(faceUrl);
  if (!file) {
    element.hidden = true;
    placeholder.hidden = false;
    name.hidden = true;
  } else {
    const url = URL.createObjectURL(file);
    if (kind === 'video') videoUrl = url;
    else faceUrl = url;
    element.src = url;
    element.hidden = false;
    placeholder.hidden = true;
    name.textContent = file.name;
    name.hidden = false;
  }
  invalidateJob();
  updateReady();
}

function configureDrop(id, input, kind) {
  const zone = document.querySelector(id);
  zone.addEventListener('dragover', event => { event.preventDefault(); zone.classList.add('dragging'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
  zone.addEventListener('drop', event => {
    event.preventDefault();
    zone.classList.remove('dragging');
    if (event.dataTransfer.files.length) {
      input.files = event.dataTransfer.files;
      preview(input, kind);
    }
  });
  input.addEventListener('change', () => preview(input, kind));
}

configureDrop('#video-drop', videoInput, 'video');
configureDrop('#face-drop', faceInput, 'face');
consent.addEventListener('change', updateReady);
strength.addEventListener('input', () => {
  strengthValue.value = `${strength.value}%`;
  sessionStorage.setItem('deep-live-cam-strength', strength.value);
  invalidatePreview();
});
previewTime.addEventListener('input', () => {
  sessionStorage.setItem('deep-live-cam-time', previewTime.value);
  invalidatePreview();
});
videoPreview.addEventListener('seeked', () => {
  previewTime.value = videoPreview.currentTime.toFixed(1);
  sessionStorage.setItem('deep-live-cam-time', previewTime.value);
  invalidatePreview();
});

function setCompareMode(enabled) {
  compareFrame.classList.toggle('comparing', enabled);
  splitControl.hidden = !enabled;
  compareMode.classList.toggle('active', enabled);
  resultMode.classList.toggle('active', !enabled);
  compareMode.setAttribute('aria-pressed', String(enabled));
  resultMode.setAttribute('aria-pressed', String(!enabled));
}
resultMode.addEventListener('click', () => setCompareMode(false));
compareMode.addEventListener('click', () => setCompareMode(true));
splitRange.addEventListener('input', () => compareFrame.style.setProperty('--split', `${splitRange.value}%`));

function showJob(job) {
  const previousStatus = jobStatus;
  jobId = job.id;
  jobStatus = job.status;
  localStorage.setItem('deep-live-cam-job', job.id);
  panel.hidden = false;
  stage.textContent = job.status;
  message.textContent = job.message;
  const active = ['queued', 'processing', 'previewing'].includes(job.status);
  indicator.hidden = !active;
  progress.hidden = !active;
  cancelButton.hidden = !active;
  cancelButton.disabled = job.message.startsWith('Cancelling');
  deleteButton.hidden = active;
  download.hidden = job.status !== 'complete';
  if (job.download_url) {
    download.href = job.download_url;
    resultVideo.src = job.result_url;
    resultVideo.hidden = false;
  } else {
    resultVideo.removeAttribute('src');
    resultVideo.hidden = true;
  }
  logs.textContent = (job.logs || []).join('\n');
  details.hidden = !(job.logs || []).length;

  const frameMatches = job.preview_url &&
    job.preview_strength === Number(strength.value) &&
    Math.abs(job.preview_seconds - Number(previewTime.value)) < 0.05;
  if (frameMatches) {
    const version = Date.now();
    originalFrame.src = `${job.original_url}?v=${version}`;
    swappedFrame.src = `${job.preview_url}?v=${version}`;
    comparePanel.hidden = false;
    previewHint.textContent = 'Use Compare to reveal the original frame.';
  } else if (job.status !== 'complete') {
    comparePanel.hidden = true;
  }
  updateReady();
  if (job.status === 'complete' && previousStatus !== 'complete') refreshSavedResults();
}

function showError(error) {
  panel.hidden = false;
  stage.textContent = 'Could not continue';
  message.textContent = error.message;
  indicator.hidden = true;
  progress.hidden = true;
  busy = false;
  updateReady();
}

async function readJson(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'The request failed.');
  return data;
}

async function ensureJob() {
  if (jobId && jobStatus === 'ready') return jobId;
  panel.hidden = false;
  stage.textContent = 'Uploading';
  message.textContent = 'Copying files to the local processor…';
  indicator.hidden = false;
  progress.hidden = false;
  const payload = new FormData();
  payload.append('video', videoInput.files[0]);
  payload.append('face', faceInput.files[0]);
  const response = await fetch('/api/jobs', {
    method: 'POST', body: payload, headers: {'X-Requested-With': 'Deep-Live-Cam'},
  });
  const job = await readJson(response);
  showJob(job);
  return job.id;
}

async function pollJob(id, stopAtReady) {
  for (;;) {
    await new Promise(resolve => setTimeout(resolve, 1200));
    const response = await fetch(`/api/jobs/${encodeURIComponent(id)}`);
    if (response.status === 404) invalidateJob(false);
    const job = await readJson(response);
    showJob(job);
    if (job.status === 'complete' || job.status === 'failed' || (stopAtReady && job.status === 'ready')) return;
  }
}

previewButton.addEventListener('click', async () => {
  if (previewButton.disabled) return;
  busy = true;
  updateReady();
  try {
    const id = await ensureJob();
    const payload = new FormData();
    payload.append('strength', strength.value);
    payload.append('seconds', previewTime.value);
    const job = await readJson(await fetch(`/api/jobs/${encodeURIComponent(id)}/preview`, {
      method: 'POST', body: payload, headers: {'X-Requested-With': 'Deep-Live-Cam'},
    }));
    showJob(job);
    await pollJob(id, true);
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    updateReady();
  }
});

cancelButton.addEventListener('click', async () => {
  if (!jobId || cancelButton.disabled) return;
  cancelButton.disabled = true;
  try {
    showJob(await readJson(await fetch(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: 'POST', headers: {'X-Requested-With': 'Deep-Live-Cam'},
    })));
  } catch (error) {
    showError(error);
  }
});

deleteButton.addEventListener('click', async () => {
  if (!jobId || deleteButton.disabled) return;
  deleteButton.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
      method: 'DELETE', headers: {'X-Requested-With': 'Deep-Live-Cam'},
    });
    if (!response.ok) throw new Error('Could not delete this job.');
    invalidateJob(false);
    updateReady();
    await refreshSavedResults();
  } catch (error) {
    deleteButton.disabled = false;
    showError(error);
  }
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (submitButton.disabled) return;
  busy = true;
  updateReady();
  try {
    const id = await ensureJob();
    const payload = new FormData();
    payload.append('strength', strength.value);
    const job = await readJson(await fetch(`/api/jobs/${encodeURIComponent(id)}/render`, {
      method: 'POST', body: payload, headers: {'X-Requested-With': 'Deep-Live-Cam'},
    }));
    showJob(job);
    await pollJob(id, false);
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    updateReady();
  }
});

refreshSavedResults();

if (jobId) {
  busy = true;
  updateReady();
  fetch(`/api/jobs/${encodeURIComponent(jobId)}`)
    .then(readJson).then(async job => {
      showJob(job);
      if (['queued', 'processing', 'previewing'].includes(job.status)) {
        await pollJob(job.id, job.status === 'previewing');
      }
    })
    .catch(error => { showError(error); })
    .finally(() => { busy = false; updateReady(); });
}

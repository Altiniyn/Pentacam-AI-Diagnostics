/* CornOrb — Frontend Application Logic */
document.addEventListener('DOMContentLoaded', () => {
  // ── Elements ──────────────────────────────────────────────────────
  const modeTabs = document.querySelectorAll('.mode-tab');
  const panels = document.querySelectorAll('.upload-panel');
  const resultsSection = document.getElementById('results-section');
  const uploadSection = document.getElementById('upload-section');
  const resultsMaps = document.getElementById('results-maps');
  const resetBtn = document.getElementById('reset-btn');
  const toast = document.getElementById('error-toast');
  const toastMsg = document.getElementById('toast-message');

  // Individual mode elements
  const mapNames = ['axial', 'anterior', 'posterior', 'pachymetry'];
  const individualFiles = {};

  // ── Tab Switching ─────────────────────────────────────────────────
  modeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const mode = tab.dataset.mode;
      modeTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      panels.forEach(p => p.classList.remove('active'));
      document.getElementById(`panel-${mode}`).classList.add('active');
    });
  });

  // ── Individual Map Uploads ────────────────────────────────────────
  mapNames.forEach(name => {
    const input = document.getElementById(`input-${name}`);
    const zone = document.getElementById(`dropzone-${name}`);
    const placeholder = document.getElementById(`placeholder-${name}`);
    const preview = document.getElementById(`preview-${name}`);

    input.addEventListener('change', e => {
      const file = e.target.files[0];
      if (!file) return;
      individualFiles[name] = file;
      const reader = new FileReader();
      reader.onload = ev => {
        preview.src = ev.target.result;
        preview.classList.remove('hidden');
        placeholder.classList.add('hidden');
        zone.classList.add('has-file');
      };
      reader.readAsDataURL(file);
      checkIndividualReady();
    });
  });

  function checkIndividualReady() {
    const ready = mapNames.every(n => individualFiles[n]);
    document.getElementById('btn-individual').disabled = !ready;
  }

  // ── Composite Upload ──────────────────────────────────────────────
  const inputComposite = document.getElementById('input-composite');
  const previewComposite = document.getElementById('preview-composite');
  const placeholderComposite = document.getElementById('placeholder-composite');
  const zoneComposite = document.getElementById('dropzone-composite');

  inputComposite.addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      previewComposite.src = ev.target.result;
      previewComposite.classList.remove('hidden');
      placeholderComposite.classList.add('hidden');
      zoneComposite.classList.add('has-file');
    };
    reader.readAsDataURL(file);
    document.getElementById('btn-composite').disabled = false;
  });

  // ── PDF Upload ────────────────────────────────────────────────────
  const inputPdf = document.getElementById('input-pdf');
  const placeholderPdf = document.getElementById('placeholder-pdf');
  const pdfFilename = document.getElementById('pdf-filename');
  const zonePdf = document.getElementById('dropzone-pdf');

  inputPdf.addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    pdfFilename.textContent = `📄 ${file.name}`;
    pdfFilename.classList.remove('hidden');
    placeholderPdf.classList.add('hidden');
    zonePdf.classList.add('has-file');
    document.getElementById('btn-pdf').disabled = false;
  });

  // ── Form Submissions ──────────────────────────────────────────────
  document.getElementById('form-individual').addEventListener('submit', async e => {
    e.preventDefault();
    const fd = new FormData();
    mapNames.forEach(n => fd.append(n, individualFiles[n]));
    await submitDiagnosis('/api/diagnose/individual', fd, 'btn-individual');
  });

  document.getElementById('form-composite').addEventListener('submit', async e => {
    e.preventDefault();
    const fd = new FormData();
    fd.append('composite', inputComposite.files[0]);
    await submitDiagnosis('/api/diagnose/composite', fd, 'btn-composite');
  });

  document.getElementById('form-pdf').addEventListener('submit', async e => {
    e.preventDefault();
    const fd = new FormData();
    fd.append('pdf_file', inputPdf.files[0]);
    await submitDiagnosis('/api/diagnose/pdf', fd, 'btn-pdf');
  });

  async function submitDiagnosis(url, formData, btnId) {
    const btn = document.getElementById(btnId);
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');
    btn.disabled = true;
    btnText.classList.add('hidden');
    btnLoader.classList.remove('hidden');

    try {
      const res = await fetch(url, { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Server error');
      showResults(data);
    } catch (err) {
      showToast(err.message || 'An error occurred');
    } finally {
      btn.disabled = false;
      btnText.classList.remove('hidden');
      btnLoader.classList.add('hidden');
    }
  }

  // ── Show Results ──────────────────────────────────────────────────
  function showResults(data) {
    // Mode label
    const modeLabels = {
      individual: 'Mode: Individual Maps Upload',
      composite: 'Mode: Composite Image — Maps auto-cropped',
      pdf: 'Mode: PDF Report — Maps auto-extracted'
    };
    document.getElementById('results-mode-label').textContent = modeLabels[data.mode] || '';

    // Render map previews
    resultsMaps.innerHTML = '';
    if (data.maps) {
      data.maps.forEach(m => {
        const card = document.createElement('div');
        card.className = 'result-map-card';
        card.innerHTML = `<img src="${m.data}" alt="${m.name}"><div class="result-map-name">${m.name}</div>`;
        resultsMaps.appendChild(card);
      });
    }

    // Clinical Data (OCR)
    const clinicalSection = document.getElementById('clinical-data-section');
    const clinicalBody = document.getElementById('clinical-data-body');
    const extractedDataContainer = document.getElementById('clinical-extracted-data');
    
    if (data.clinical_data && data.clinical_data.values && data.clinical_data.values.length > 0) {
      clinicalSection.classList.remove('hidden');
      clinicalBody.innerHTML = '';
      data.clinical_data.values.forEach(item => {
        const tr = document.createElement('tr');
        const riskClass = item.risk ? `val-${item.risk}` : '';
        const badge = item.risk_label 
          ? `<span class="risk-badge risk-${item.risk}">${item.risk_label}</span>` 
          : `<span class="val-extracted">—</span>`;
        tr.innerHTML = `
          <td>${item.name}</td>
          <td class="${riskClass}">${item.value} ${item.unit}</td>
          <td>${badge}</td>
        `;
        clinicalBody.appendChild(tr);
      });
      extractedDataContainer.classList.remove('hidden');
      
      // Inject Clinical Rules Overall Diagnosis above AI Model if available
      let rulesContainer = document.getElementById('clinical-rules-overall');
      if (!rulesContainer) {
        rulesContainer = document.createElement('div');
        rulesContainer.id = 'clinical-rules-overall';
        rulesContainer.className = 'clinical-rules-result';
        clinicalSection.appendChild(rulesContainer);
      }
      const cRisk = data.clinical_data.overall_risk;
      const cLabel = data.clinical_data.overall_diagnosis;
      const cIcon = cRisk === 'normal' ? '✅' : (cRisk === 'danger' ? '⚠️' : '🔍');
      rulesContainer.innerHTML = `
        <div class="rules-card risk-${cRisk}">
            <div class="rules-icon">${cIcon}</div>
            <div class="rules-info">
                <h4>Clinical Rules Assessment</h4>
                <p>${cLabel}</p>
                <small>Based on standard thresholds for K Max and Pachymetry</small>
            </div>
        </div>
      `;
    } else {
      clinicalSection.classList.add('hidden');
    }

    // AI Diagnosis result
    const diagnosisCard = document.getElementById('diagnosis-card');
    
    if (data.clinical_data && data.clinical_data.overall_risk === 'normal') {
        // User requested: hide the model evaluation if clinical rules say Normal
        diagnosisCard.classList.add('hidden');
    } else {
        diagnosisCard.classList.remove('hidden');
        
        const isNormal = data.prediction === 'Normal';
        const cls = isNormal ? 'normal' : 'keratoconus';
        const icon = isNormal ? '✅' : '⚠️';

        const diagIcon = document.getElementById('diagnosis-icon');
        diagIcon.className = `diagnosis-icon ${cls}`;
        diagIcon.textContent = icon;

        const diagLabel = document.getElementById('diagnosis-label');
        diagLabel.className = `diagnosis-label ${cls}`;
        diagLabel.textContent = "AI Model: " + data.prediction;

        document.getElementById('diagnosis-confidence').textContent = `Confidence: ${data.confidence}%`;
        document.getElementById('prob-normal').textContent = `${data.probabilities.Normal}%`;
        document.getElementById('prob-keratoconus').textContent = `${data.probabilities.Keratoconus}%`;

        // Animate bars
        setTimeout(() => {
          document.getElementById('bar-normal').style.width = `${data.probabilities.Normal}%`;
          document.getElementById('bar-keratoconus').style.width = `${data.probabilities.Keratoconus}%`;
        }, 100);
    }

    // Show results, hide upload
    uploadSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
  }

  // ── Reset ─────────────────────────────────────────────────────────
  resetBtn.addEventListener('click', () => {
    resultsSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');

    // Reset individual
    mapNames.forEach(n => {
      delete individualFiles[n];
      document.getElementById(`input-${n}`).value = '';
      document.getElementById(`preview-${n}`).classList.add('hidden');
      document.getElementById(`placeholder-${n}`).classList.remove('hidden');
      document.getElementById(`dropzone-${n}`).classList.remove('has-file');
    });
    document.getElementById('btn-individual').disabled = true;

    // Reset composite
    inputComposite.value = '';
    previewComposite.classList.add('hidden');
    previewComposite.src = '';
    placeholderComposite.classList.remove('hidden');
    zoneComposite.classList.remove('has-file');
    document.getElementById('btn-composite').disabled = true;

    // Reset PDF
    inputPdf.value = '';
    pdfFilename.classList.add('hidden');
    placeholderPdf.classList.remove('hidden');
    zonePdf.classList.remove('has-file');
    document.getElementById('btn-pdf').disabled = true;

    // Reset bars
    document.getElementById('bar-normal').style.width = '0%';
    document.getElementById('bar-keratoconus').style.width = '0%';
    
    // Reset Data
    document.getElementById('clinical-data-section').classList.add('hidden');

    uploadSection.scrollIntoView({ behavior: 'smooth' });
  });

  // ── Toast ─────────────────────────────────────────────────────────
  function showToast(msg) {
    toastMsg.textContent = msg;
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), 5000);
  }
});

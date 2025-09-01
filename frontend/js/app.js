/* =========================
   API Configuration
========================= */
const API_BASE = 'http://localhost:8000/api';

/* =========================
   API Functions
========================= */
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
    }
    
    return await response.json();
}

async function processUploadedFiles() {
    const response = await fetch(`${API_BASE}/process-uploaded`, {
        method: 'POST'
    });
    
    if (!response.ok) {
        throw new Error(`Processing failed: ${response.statusText}`);
    }
    
    return await response.json();
}

async function checkProcessingStatus() {
    const response = await fetch(`${API_BASE}/status`);
    return await response.json();
}

async function getResults(filename) {
    const response = await fetch(`${API_BASE}/results/${encodeURIComponent(filename)}`);
    return await response.json();
}

async function listFiles() {
    const response = await fetch(`${API_BASE}/files`);
    return await response.json();
}

/* =========================
   Global Data Storage
========================= */
let CURRENT_RESULTS = null;
let PROCESSING_INTERVAL = null;

/* =========================
   Theme toggle (light/dark)
========================= */
const themeToggle = document.getElementById("themeToggle");
themeToggle.addEventListener("click", () => {
  document.body.classList.toggle("dark");

  if (document.body.classList.contains("dark")) {
    document.body.classList.replace("bg-gray-50","bg-gray-900");
    document.body.classList.replace("text-gray-900","text-gray-100");
    themeToggle.textContent = "☀️";

    document.querySelectorAll("[class*='text-gray-']").forEach(el => {
      if (el.classList.contains("text-gray-900")) {
        el.classList.replace("text-gray-900","text-gray-100");
      }
      if (el.classList.contains("text-gray-800")) {
        el.classList.replace("text-gray-800","text-gray-200");
      }
      if (el.classList.contains("text-gray-700")) {
        el.classList.replace("text-gray-700","text-gray-300");
      }
      if (el.classList.contains("text-gray-600")) {
        el.classList.replace("text-gray-600","text-gray-400");
      }
    });

  } else {
    document.body.classList.replace("bg-gray-900","bg-gray-50");
    document.body.classList.replace("text-gray-100","text-gray-900");
    themeToggle.textContent = "🌙";

    document.querySelectorAll("[class*='text-gray-']").forEach(el => {
      if (el.classList.contains("text-gray-100")) {
        el.classList.replace("text-gray-100","text-gray-900");
      }
      if (el.classList.contains("text-gray-200")) {
        el.classList.replace("text-gray-200","text-gray-800");
      }
      if (el.classList.contains("text-gray-300")) {
        el.classList.replace("text-gray-300","text-gray-700");
      }
      if (el.classList.contains("text-gray-400")) {
        el.classList.replace("text-gray-400","text-gray-600");
      }
    });
  }
});

/* =========================
   Sections + nav handlers
========================= */
const uploadSection = document.getElementById("uploadSection");
const loadingSection = document.getElementById("loadingSection");
const llmSection = document.getElementById("llmSection");
const adSection = document.getElementById("adSection");
const evaluationSection = document.getElementById("evaluationSection");

const analyzeBtn = document.getElementById("analyzeBtn");
const progressFill = document.getElementById("progressFill");
const loadingText = document.getElementById("loadingText");

const backToUpload = document.getElementById("backToUpload");
const backToLLM = document.getElementById("backToLLM");
const backToAd = document.getElementById("backToAd");

/* =========================
   Initialize Dashboard
========================= */
document.addEventListener('DOMContentLoaded', async () => {
    await checkExistingFiles();
});

async function checkExistingFiles() {
    try {
        const fileList = await listFiles();
        
        if (fileList.files && fileList.files.length > 0) {
            const processedFile = fileList.files.find(f => f.fully_processed);
            
            if (processedFile) {
                // Load existing results
                console.log('Loading existing results for:', processedFile.filename);
                await loadExistingResults(processedFile.filename);
            }
        }
    } catch (error) {
        console.error('Error checking existing files:', error);
    }
}

async function loadExistingResults(filename) {
    try {
        const results = await getResults(filename);
        
        // Fix data structure mapping
        CURRENT_RESULTS = {
            visualization_data: results.visualization_data || results,
            cross_validation: results.cross_validation
        };
        
        uploadSection.classList.add("hidden");
        llmSection.classList.remove("hidden");
        
        renderMetadataDashboard(CURRENT_RESULTS);
        setupSynopsisObserver();
        
    } catch (error) {
        console.error('Error loading results:', error);
    }
}

/* =========================
   File Upload Handler
========================= */
const fileInput = document.getElementById('fileInput');
if (fileInput) {
    fileInput.addEventListener('change', handleFileUpload);
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    try {
        console.log('Uploading file:', file.name);
        
        // Upload file
        const uploadResult = await uploadFile(file);
        console.log('Upload successful:', uploadResult);
        
        // Start processing
        startProcessing();
        
    } catch (error) {
        console.error('Upload failed:', error);
        alert('Upload failed: ' + error.message);
    }
}

/* =========================
   Analyze button (for existing files)
========================= */
analyzeBtn.addEventListener("click", async () => {
    try {
        // Check if there are files to process
        const fileList = await listFiles();
        
        if (!fileList.files || fileList.files.length === 0) {
            alert('No files found to process. Please upload a transcript file first.');
            return;
        }
        
        console.log('Starting processing for existing files...');
        await processUploadedFiles();
        startProcessing();
        
    } catch (error) {
        console.error('Processing failed:', error);
        alert('Processing failed: ' + error.message);
    }
});

/* =========================
   Processing Management
========================= */
function startProcessing() {
    uploadSection.classList.add("hidden");
    loadingSection.classList.remove("hidden");
    
    // Start progress monitoring
    PROCESSING_INTERVAL = setInterval(updateProcessingStatus, 2000);
}

async function updateProcessingStatus() {
    try {
        const status = await checkProcessingStatus();
        
        // Update progress bar
        progressFill.style.width = status.progress + "%";
        loadingText.textContent = status.current_step;
        
        if (!status.is_processing) {
            clearInterval(PROCESSING_INTERVAL);
            
            if (status.error) {
                console.error('Processing error:', status.error);
                loadingText.textContent = 'Processing failed: ' + status.error;
                setTimeout(() => {
                    loadingSection.classList.add("hidden");
                    uploadSection.classList.remove("hidden");
                }, 3000);
            } else {
                // Processing completed successfully
                console.log('Processing completed successfully');
                setTimeout(async () => {
                    await loadProcessingResults();
                }, 1000);
            }
        }
        
    } catch (error) {
        console.error('Error checking status:', error);
        clearInterval(PROCESSING_INTERVAL);
    }
}

async function loadProcessingResults() {
    try {
        const fileList = await listFiles();
        const latestFile = fileList.files.find(f => f.fully_processed);
        
        if (latestFile) {
            const results = await getResults(latestFile.filename);
            
            // Fix data structure
            CURRENT_RESULTS = {
                visualization_data: results.visualization_data || results,
                cross_validation: results.cross_validation
            };
            
            loadingSection.classList.add("hidden");
            llmSection.classList.remove("hidden");
            
            renderMetadataDashboard(CURRENT_RESULTS);
            setupSynopsisObserver();
        }
        
    } catch (error) {
        console.error('Error loading results:', error);
    }
}

/* =========================
   Render Metadata Dashboard (Updated to use API data)
========================= */
let sentimentChart, keywordsChart, emotionsChart;

function renderMetadataDashboard(results) {
    if (!results || !results.visualization_data) {
        console.error('No visualization data found in results');
        return;
    }
    
    const data = results.visualization_data.metadata_tab;
    
    // Lead characters
    const lead = document.getElementById("leadCharacter");
    if (data.lead_characters && data.lead_characters.length > 0) {
        lead.innerHTML = `
          <div class="flex flex-wrap gap-2">
            ${data.lead_characters
              .sort((a, b) => b.importance - a.importance)
              .slice(0, 3)
              .map(c => `<span class="pill-badge">${c.name}</span>`)
              .join("")}
          </div>
        `;
    }
    
    // Genres
    const genreList = document.getElementById("genreList");
    if (data.genre_classification) {
        genreList.innerHTML = data.genre_classification
            .map(g => `
              <li class="flex items-center justify-between">
                <span>${g.genre}</span>
                <span class="list-badge">${g.confidence}%</span>
              </li>`
            ).join("");
    }
    
    // Content Classification
    const classificationList = document.getElementById("classificationList");
    if (data.content_classification_plot && data.content_classification_plot.categories) {
        classificationList.innerHTML = data.content_classification_plot.categories
            .map(c => `
              <li class="flex items-center justify-between">
                <span class="text-gray-700">${c}</span>
                <span class="dot"></span>
              </li>`
            ).join("");
    }
    
    // Keywords Chart
    if (data.keywords_plot && data.keywords_plot.keywords.length > 0) {
        const topKeywords = data.keywords_plot.keywords.slice(0, 5);
        const topPercentages = data.keywords_plot.percentages.slice(0, 5);
        
        const keywordsList = document.getElementById("keywordsList");
        keywordsList.innerHTML = topKeywords
            .map((k, i) => `<span class="badge">${k} (${topPercentages[i]}%)</span>`)
            .join("");
        
        const kctx = document.getElementById("keywordsChart");
        if (keywordsChart) keywordsChart.destroy();
        keywordsChart = new Chart(kctx, {
            type: "bar",
            data: {
                labels: topKeywords,
                datasets: [{
                    data: topPercentages,
                    backgroundColor: "#6366f1"
                }]
            },
            options: {
                indexAxis: "y",
                maintainAspectRatio: false,
                responsive: true,
                plugins: { legend: { display: false } }
            }
        });
    }
    
    // Sentiment Pie
    if (data.sentiment_pie && data.sentiment_pie.labels.length > 0) {
        const sctx = document.getElementById("sentimentChart");
        if (sentimentChart) sentimentChart.destroy();
        sentimentChart = new Chart(sctx, {
            type: "doughnut",
            data: {
                labels: data.sentiment_pie.labels,
                datasets: [{
                    data: data.sentiment_pie.values,
                    backgroundColor: ["#34d399", "#f87171", "#fbbf24"]
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: "right" } },
                cutout: "60%"
            }
        });
    }
    
    // Named Entities
    if (data.named_entities) {
        renderNER(data.named_entities);
    }
    
    // Emotions Chart
    if (data.emotion_pie && data.emotion_pie.labels.length > 0) {
        const emotionsCtx = document.getElementById("emotionsChart");
        if (emotionsChart) emotionsChart.destroy();
        emotionsChart = new Chart(emotionsCtx, {
            type: "pie",
            data: {
                labels: data.emotion_pie.labels,
                datasets: [{
                    data: data.emotion_pie.values,
                    backgroundColor: ["#60a5fa", "#f87171", "#9ca3af", "#fbbf24"]
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: "bottom" } }
            }
        });
    }
    
    // Synopsis
    if (data.synopsis_summary) {
        document.getElementById("synopsis").textContent = data.synopsis_summary.synopsis;
    }
}

/* =========================
   NER Summary (counts + lists)
========================= */
function renderNER(namedEntities) {
  const nerDiv = document.getElementById("nerSummary");

  nerDiv.innerHTML = `
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center mb-6">
      <div class="p-3 rounded-lg bg-indigo-50">
        <p class="text-lg font-bold">${namedEntities.people_count || 0}</p>
        <p class="text-xs text-gray-600">People</p>
      </div>
      <div class="p-3 rounded-lg bg-indigo-50">
        <p class="text-lg font-bold">${namedEntities.locations_count || 0}</p>
        <p class="text-xs text-gray-600">Locations</p>
      </div>
      <div class="p-3 rounded-lg bg-indigo-50">
        <p class="text-lg font-bold">${namedEntities.organizations_count || 0}</p>
        <p class="text-xs text-gray-600">Organizations</p>
      </div>
      <div class="p-3 rounded-lg bg-indigo-50">
        <p class="text-lg font-bold">${namedEntities.total || 0}</p>
        <p class="text-xs text-gray-600">Total Entities</p>
      </div>
    </div>

    <div class="grid md:grid-cols-3 gap-6 text-sm text-left">
      <div>
        <h4 class="font-semibold mb-2">👤 People</h4>
        <ul class="list-disc list-inside space-y-1 text-gray-700">
          ${(namedEntities.people || []).map(p => `<li>${p}</li>`).join("")}
        </ul>
      </div>

      <div>
        <h4 class="font-semibold mb-2">📍 Locations</h4>
        <ul class="list-disc list-inside space-y-1 text-gray-700">
          ${(namedEntities.locations || []).map(l => `<li>${l}</li>`).join("")}
        </ul>
      </div>

      <div>
        <h4 class="font-semibold mb-2">🏛 Organizations</h4>
        <ul class="list-disc list-inside space-y-1 text-gray-700">
          ${(namedEntities.organizations || []).map(o => `<li>${o}</li>`).join("")}
        </ul>
      </div>
    </div>
  `;
}

/* =========================
   Floating CTA on Synopsis
========================= */
function setupSynopsisObserver() {
  const synopsisCard = document.getElementById("synopsisCard");
  const btn = document.getElementById("floatingToAdInsights");
  const io = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          btn.classList.remove("hidden");
          btn.classList.add("pop");
        } else {
          btn.classList.add("hidden");
          btn.classList.remove("pop");
        }
      });
    },
    { threshold: 0.6 }
  );
  io.observe(synopsisCard);

  btn.addEventListener("click", () => {
    llmSection.classList.add("hidden");
    adSection.classList.remove("hidden");
    renderAdInsights();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

/* =========================
   Ad Insights rendering (Updated to use API data)
========================= */
function renderAdInsights() {
    if (!CURRENT_RESULTS || !CURRENT_RESULTS.visualization_data) {
        console.error('No ad insights data available');
        return;
    }
    
    const adData = CURRENT_RESULTS.visualization_data.ad_insights_tab;
    
    // Timeline chart
    const ctxA = document.getElementById("adChart");
    if (window._adChart) window._adChart.destroy();
    
    if (adData.ad_placement_timeline && adData.ad_placement_timeline.placements) {
        window._adChart = new Chart(ctxA, {
            type: "line",
            data: {
                labels: adData.ad_placement_timeline.placements.map(p => p.timestamp),
                datasets: [{
                    label: "Suitability",
                    data: adData.ad_placement_timeline.placements.map(p => p.suitability),
                    borderColor: "#6366f1",
                    backgroundColor: "rgba(99,102,241,0.15)",
                    fill: true,
                    tension: 0.35
                }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
        
        // Timeline list
        const adList = document.getElementById("adList");
        adList.innerHTML = adData.ad_placement_timeline.placements
            .map(p => `<li>🕒 <b>${p.timestamp}</b> — ${p.scene} (${p.suitability}%)</li>`)
            .join("");
    }
    
    // Recommendations list
    if (adData.ad_recommendations) {
        const adRecList = document.getElementById("adRecList");
        adRecList.innerHTML = adData.ad_recommendations
            .map(r => `
              <li>
                <div class="font-medium">${r.scene} (${r.suitability}%)</div>
                <div class="text-sm text-gray-500">${r.reasoning}</div>
                <div class="flex flex-wrap gap-2 mt-1">
                  ${r.ad_types.map(ad => `<span class="badge">${ad}</span>`).join("")}
                </div>
              </li>`)
            .join("");
    }
}

/* =========================
   Evaluation Charts (Updated to use API data)
========================= */
document.getElementById("toEvaluation").addEventListener("click", () => {
  adSection.classList.add("hidden");
  evaluationSection.classList.remove("hidden");
  
  if (CURRENT_RESULTS && CURRENT_RESULTS.cross_validation) {
      renderEvaluation(CURRENT_RESULTS.cross_validation);
  } else {
      console.error('No cross validation data available');
  }
});

function renderEvaluation(DATA) {
  if (!DATA) return;
  
  // Overall Confidence Gauge
const reliability = DATA.validation_summary?.overall_reliability || 50;

makeChart("confidenceGauge", {
  type: "doughnut",
  data: {
    labels: ["Reliability", "Remaining"],
    datasets: [{
      data: [reliability, 100 - reliability],
      backgroundColor: [
        reliability >= 70 ? "#34d399" :  // green if good
        reliability >= 40 ? "#fbbf24" :  // amber if medium
                           "#f87171",   // red if low
        "#e5e7eb"
      ],
      borderWidth: 0
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    cutout: "75%",
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false },
      // Friend’s addition → center text plugin
      beforeDraw: (chart) => {
        const {ctx, chartArea: {width, height}} = chart;
        ctx.save();
        ctx.font = "bold 20px sans-serif";
        ctx.fillStyle = "#111827"; // dark gray
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(`${reliability}%`, width / 2, height / 2);
      }
    }
  }
});


  // Venn Diagram
  const venn = DATA.confidence_scores?.venn_data || { llm_only: 0, nlp_only: 0, both: 0, total: 1 };
  const total = venn.total || 1;

  makeChart("confusionVenn", {
    type: "doughnut",
    data: {
      labels: ["LLM only", "NLP only", "Both"],
      datasets: [{
        data: [venn.llm_only, venn.nlp_only, venn.both],
        backgroundColor: ["#6366f1", "#34d399", "#fbbf24"]
      }]
    },
    options: { responsive: true, maintainAspectRatio: false }
  });

  // Confusion Matrix Table
  const confusionHTML = `
    <table class="w-full text-xs border border-gray-300">
      <thead class="bg-gray-50">
        <tr>
          <th class="border px-2 py-1">Category</th>
          <th class="border px-2 py-1">Count</th>
          <th class="border px-2 py-1">%</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="border px-2 py-1">LLM Only</td>
          <td class="border px-2 py-1">${venn.llm_only}</td>
          <td class="border px-2 py-1">${((venn.llm_only/total)*100).toFixed(1)}%</td>
        </tr>
        <tr>
          <td class="border px-2 py-1">NLP Only</td>
          <td class="border px-2 py-1">${venn.nlp_only}</td>
          <td class="border px-2 py-1">${((venn.nlp_only/total)*100).toFixed(1)}%</td>
        </tr>
        <tr>
          <td class="border px-2 py-1">Both</td>
          <td class="border px-2 py-1">${venn.both}</td>
          <td class="border px-2 py-1">${((venn.both/total)*100).toFixed(1)}%</td>
        </tr>
      </tbody>
    </table>`;
  document.getElementById("confusionTable").innerHTML = confusionHTML;

  // Radar Chart
  const triangleData = DATA.performance_metrics?.triangle_chart_data;
  if (triangleData) {
      makeChart("triangleRadar", {
        type: "radar",
        data: {
          labels: triangleData.categories,
          datasets: [
            {
              label: "LLM",
              data: triangleData.llm_metrics,
              borderColor: "#6366f1",
              backgroundColor: "rgba(99,102,241,0.2)"
            },
            {
              label: "NLP",
              data: triangleData.nlp_metrics,
              borderColor: "#34d399",
              backgroundColor: "rgba(52,211,153,0.2)"
            }
          ]
        },
        options: { responsive: true, maintainAspectRatio: false }
      });
  }
}

// Chart management
const evalCharts = {};

function makeChart(id, cfg) {
  const el = document.getElementById(id);
  if (evalCharts[id]) { evalCharts[id].destroy(); }
  evalCharts[id] = new Chart(el.getContext('2d'), cfg);
}

/* =========================
   Back navigation
========================= */
backToUpload.addEventListener("click", () => {
  llmSection.classList.add("hidden");
  uploadSection.classList.remove("hidden");
});

backToLLM.addEventListener("click", () => {
  adSection.classList.add("hidden");
  llmSection.classList.remove("hidden");
});

if (backToAd) {
  backToAd.addEventListener("click", () => {
    evaluationSection.classList.add("hidden");
    adSection.classList.remove("hidden");
  });
}
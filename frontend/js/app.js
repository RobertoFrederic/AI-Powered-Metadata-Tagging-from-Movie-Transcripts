
/* =========================
   Theme toggle (light/dark)
========================= */
// const themeToggle = document.getElementById("themeToggle");
// themeToggle.addEventListener("click", () => {
//   document.body.classList.toggle("dark");
//   if (document.body.classList.contains("dark")) {
//     document.body.classList.replace("bg-gray-50", "bg-gray-900");
//     document.body.classList.replace("text-gray-900", "text-gray-100");
//     themeToggle.textContent = "☀️";
//   } else {
//     document.body.classList.replace("bg-gray-900", "bg-gray-50");
//     document.body.classList.replace("text-gray-100", "text-gray-900");
//     themeToggle.textContent = "🌙";
//   }
// });
themeToggle.addEventListener("click", () => {
  document.body.classList.toggle("dark");

  if (document.body.classList.contains("dark")) {
    document.body.classList.replace("bg-gray-50","bg-gray-900");
    document.body.classList.replace("text-gray-900","text-gray-100");
    themeToggle.textContent = "☀️";

    // 🔹 Update all text colors globally
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

    // 🔹 Restore all text colors globally
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
   Dummy JSON (replace with API call if needed)
========================= */
const META_DATA = {
  "metadata_tab": {
    "lead_characters": [
      {
        "name": "Harry Potter",
        "importance": 30,
        "mentions": 50,
        "emotion": "sadness"
      },
      {
        "name": "Uncle Vernon",
        "importance": 20,
        "mentions": 35,
        "emotion": "anger"
      },
      {
        "name": "Aunt Petunia",
        "importance": 15,
        "mentions": 25,
        "emotion": "neutral"
      },
      {
        "name": "Dudley Dursley",
        "importance": 10,
        "mentions": 18,
        "emotion": "anger"
      },
      {
        "name": "Albus Dumbledore",
        "importance": 8,
        "mentions": 12,
        "emotion": "neutral"
      },
      {
        "name": "Professor McGonagall",
        "importance": 7,
        "mentions": 10,
        "emotion": "concern"
      },
      {
        "name": "Hagrid",
        "importance": 5,
        "mentions": 8,
        "emotion": "sadness"
      }
    ],
    "genre_classification": [
      {
        "genre": "Fantasy",
        "confidence": 95.0
      },
      {
        "genre": "Drama",
        "confidence": 80.0
      }
    ],
    "content_classification_plot": {
      "categories": [
        "Fantasy",
        "Drama"
      ],
      "scores": [
        95.0,
        80.0
      ]
    },
    "keywords_plot": {
      "keywords": [
        "Magic",
        "Wizard",
        "Hogwarts",
        "Wand",
        "Letters",
        "Owls",
        "School",
        "Parents",
        "Cupboard",
        "Birthday"
      ],
      "percentages": [
        6.0,
        6.0,
        5.0,
        5.0,
        4.0,
        3.0,
        3.0,
        2.0,
        2.0,
        2.0
      ]
    },
    "sentiment_pie": {
      "labels": [
        "Positive",
        "Negative",
        "Neutral"
      ],
      "values": [
        19.8,
        15.9,
        64.2
      ]
    },
    "emotion_pie": {
      "labels": [
        "sadness",
        "anger",
        "neutral",
        "concern"
      ],
      "values": [
        2,
        2,
        2,
        1
      ]
    },
    "named_entities": {
      "people_count": 7,
      "people": [
        "Harry Potter",
        "Uncle Vernon",
        "Aunt Petunia",
        "Dudley Dursley",
        "Albus Dumbledore",
        "Professor McGonagall",
        "Hagrid"
      ],
      "locations_count": 5,
      "locations": [
        "Privet Drive",
        "Hogwarts",
        "Cupboard Under the Stairs",
        "Zoo",
        "Bristol"
      ],
      "organizations_count": 1,
      "organizations": [
        "Hogwarts School of Witchcraft and Wizardry"
      ],
      "total": 13
    },
    "emotion_radar": {
      "labels": [
        "neutral",
        "joy",
        "anger",
        "fear",
        "sadness"
      ],
      "values": [
        130.0,
        0,
        155.0,
        0,
        90.0
      ]
    },
    "synopsis_summary": {
      "synopsis": "Harry Potter, a neglected boy living with his cruel relatives, discovers on his eleventh birthday that he is a wizard and destined for Hogwarts School of Witchcraft and Wizardry.  The Dursleys try desperately to prevent him from learning the truth, but an onslaught of letters and owls delivers the news.",
      "word_count": 37193,
      "duration": 206
    }
  },
  "ad_insights_tab": {
    "ad_placement_timeline": {
      "placements": [
        {
          "id": 1,
          "timestamp": "00:05:00",
          "suitability": 90.0,
          "scene": "Scene where Dudley receives his birthday presents...."
        },
        {
          "id": 2,
          "timestamp": "00:10:00",
          "suitability": 80.0,
          "scene": "The scene at the zoo before the snake incident...."
        },
        {
          "id": 3,
          "timestamp": "00:15:00",
          "suitability": 70.0,
          "scene": "The montage of the Dursleys trying to destroy the ..."
        },
        {
          "id": 4,
          "timestamp": "00:18:00",
          "suitability": 60.0,
          "scene": "The scene where Harry is sitting in the cupboard a..."
        },
        {
          "id": 5,
          "timestamp": "00:19:00",
          "suitability": 80.0,
          "scene": "The final scene with the owls delivering letters...."
        }
      ],
      "total_slots": 5
    },
    "ad_recommendations": [
      {
        "placement_id": 1,
        "scene": "Scene where Dudley receives his birthday presents.",
        "ad_types": [
          "Toy commercials",
          "Candy commercials"
        ],
        "suitability": 90.0,
        "reasoning": "The scene is focused on material possessions, making toy and candy ads a natural fit."
      },
      {
        "placement_id": 2,
        "scene": "The scene at the zoo before the snake incident.",
        "ad_types": [
          "Family travel ads",
          "Zoo/wildlife conservation ads"
        ],
        "suitability": 80.0,
        "reasoning": "The scene is set in a family-friendly environment."
      },
      {
        "placement_id": 3,
        "scene": "The montage of the Dursleys trying to destroy the letters.",
        "ad_types": [
          "Insurance commercials",
          "Security system ads"
        ],
        "suitability": 70.0,
        "reasoning": "The scene highlights the Dursleys' paranoia and fear."
      },
      {
        "placement_id": 4,
        "scene": "The scene where Harry is sitting in the cupboard after being punished.",
        "ad_types": [
          "Educational ads",
          "Child welfare ads"
        ],
        "suitability": 60.0,
        "reasoning": "The scene highlights Harry's loneliness and isolation."
      },
      {
        "placement_id": 5,
        "scene": "The final scene with the owls delivering letters.",
        "ad_types": [
          "Postal service ads",
          "Delivery service ads"
        ],
        "suitability": 80.0,
        "reasoning": "The scene is visually dynamic and emphasizes the delivery of messages."
      }
    ],
    "placement_strategy": {
      "strategy": "scene_transition_based",
      "total_recommended_slots": 10,
      "average_suitability": 85.2
    }
  },
  "processing_info": {
    "llm_processing_time": "2025-08-31T19:19:42.488183",
    "nlp_processing_time": 9.64,
    "total_chunks": 2240
  }
}.metadata_tab;


const AD_DATA = {
  "metadata_tab": {
    "lead_characters": [
      {
        "name": "Harry Potter",
        "importance": 30,
        "mentions": 50,
        "emotion": "sadness"
      },
      {
        "name": "Uncle Vernon",
        "importance": 20,
        "mentions": 35,
        "emotion": "anger"
      },
      {
        "name": "Aunt Petunia",
        "importance": 15,
        "mentions": 25,
        "emotion": "neutral"
      },
      {
        "name": "Dudley Dursley",
        "importance": 10,
        "mentions": 18,
        "emotion": "anger"
      },
      {
        "name": "Albus Dumbledore",
        "importance": 8,
        "mentions": 12,
        "emotion": "neutral"
      },
      {
        "name": "Professor McGonagall",
        "importance": 7,
        "mentions": 10,
        "emotion": "concern"
      },
      {
        "name": "Hagrid",
        "importance": 5,
        "mentions": 8,
        "emotion": "sadness"
      }
    ],
    "genre_classification": [
      {
        "genre": "Fantasy",
        "confidence": 95.0
      },
      {
        "genre": "Drama",
        "confidence": 80.0
      }
    ],
    "content_classification_plot": {
      "categories": [
        "Fantasy",
        "Drama"
      ],
      "scores": [
        95.0,
        80.0
      ]
    },
    "keywords_plot": {
      "keywords": [
        "Magic",
        "Wizard",
        "Hogwarts",
        "Wand",
        "Letters",
        "Owls",
        "School",
        "Parents",
        "Cupboard",
        "Birthday"
      ],
      "percentages": [
        6.0,
        6.0,
        5.0,
        5.0,
        4.0,
        3.0,
        3.0,
        2.0,
        2.0,
        2.0
      ]
    },
    "sentiment_pie": {
      "labels": [
        "Positive",
        "Negative",
        "Neutral"
      ],
      "values": [
        19.8,
        15.9,
        64.2
      ]
    },
    "emotion_pie": {
      "labels": [
        "sadness",
        "anger",
        "neutral",
        "concern"
      ],
      "values": [
        2,
        2,
        2,
        1
      ]
    },
    "named_entities": {
      "people_count": 7,
      "people": [
        "Harry Potter",
        "Uncle Vernon",
        "Aunt Petunia",
        "Dudley Dursley",
        "Albus Dumbledore",
        "Professor McGonagall",
        "Hagrid"
      ],
      "locations_count": 5,
      "locations": [
        "Privet Drive",
        "Hogwarts",
        "Cupboard Under the Stairs",
        "Zoo",
        "Bristol"
      ],
      "organizations_count": 1,
      "organizations": [
        "Hogwarts School of Witchcraft and Wizardry"
      ],
      "total": 13
    },
    "emotion_radar": {
      "labels": [
        "neutral",
        "joy",
        "anger",
        "fear",
        "sadness"
      ],
      "values": [
        130.0,
        0,
        155.0,
        0,
        90.0
      ]
    },
    "synopsis_summary": {
      "synopsis": "Harry Potter, a neglected boy living with his cruel relatives, discovers on his eleventh birthday that he is a wizard and destined for Hogwarts School of Witchcraft and Wizardry.  The Dursleys try desperately to prevent him from learning the truth, but an onslaught of letters and owls delivers the news.",
      "word_count": 37193,
      "duration": 206
    }
  },
  "ad_insights_tab": {
    "ad_placement_timeline": {
      "placements": [
        {
          "id": 1,
          "timestamp": "00:05:00",
          "suitability": 90.0,
          "scene": "Scene where Dudley receives his birthday presents...."
        },
        {
          "id": 2,
          "timestamp": "00:10:00",
          "suitability": 80.0,
          "scene": "The scene at the zoo before the snake incident...."
        },
        {
          "id": 3,
          "timestamp": "00:15:00",
          "suitability": 70.0,
          "scene": "The montage of the Dursleys trying to destroy the ..."
        },
        {
          "id": 4,
          "timestamp": "00:18:00",
          "suitability": 60.0,
          "scene": "The scene where Harry is sitting in the cupboard a..."
        },
        {
          "id": 5,
          "timestamp": "00:19:00",
          "suitability": 80.0,
          "scene": "The final scene with the owls delivering letters...."
        }
      ],
      "total_slots": 5
    },
    "ad_recommendations": [
      {
        "placement_id": 1,
        "scene": "Scene where Dudley receives his birthday presents.",
        "ad_types": [
          "Toy commercials",
          "Candy commercials"
        ],
        "suitability": 90.0,
        "reasoning": "The scene is focused on material possessions, making toy and candy ads a natural fit."
      },
      {
        "placement_id": 2,
        "scene": "The scene at the zoo before the snake incident.",
        "ad_types": [
          "Family travel ads",
          "Zoo/wildlife conservation ads"
        ],
        "suitability": 80.0,
        "reasoning": "The scene is set in a family-friendly environment."
      },
      {
        "placement_id": 3,
        "scene": "The montage of the Dursleys trying to destroy the letters.",
        "ad_types": [
          "Insurance commercials",
          "Security system ads"
        ],
        "suitability": 70.0,
        "reasoning": "The scene highlights the Dursleys' paranoia and fear."
      },
      {
        "placement_id": 4,
        "scene": "The scene where Harry is sitting in the cupboard after being punished.",
        "ad_types": [
          "Educational ads",
          "Child welfare ads"
        ],
        "suitability": 60.0,
        "reasoning": "The scene highlights Harry's loneliness and isolation."
      },
      {
        "placement_id": 5,
        "scene": "The final scene with the owls delivering letters.",
        "ad_types": [
          "Postal service ads",
          "Delivery service ads"
        ],
        "suitability": 80.0,
        "reasoning": "The scene is visually dynamic and emphasizes the delivery of messages."
      }
    ],
    "placement_strategy": {
      "strategy": "scene_transition_based",
      "total_recommended_slots": 10,
      "average_suitability": 85.2
    }
  },
  "processing_info": {
    "llm_processing_time": "2025-08-31T19:19:42.488183",
    "nlp_processing_time": 9.64,
    "total_chunks": 2240
  }
}.ad_insights_tab;
const EVAL_DATA = {
  "confidence_scores": {
    "llm_confidence": 0.72,
    "nlp_confidence": 0.88,
    "agreement_score": 0.39,
    "overall_confidence": 0.66,
    "venn_data": {
      "llm_only": 8,
      "nlp_only": 8,
      "both": 2,
      "total": 18
    }
  },
  "performance_metrics": {
    "processing_speed": {
      "llm_time": 20,
      "nlp_time": 17.07,
      "speed_ratio": 0.85
    },
    "accuracy_metrics": {
      "sentiment_accuracy": {
        "llm": 70.0,
        "nlp": 52.7
      },
      "entity_extraction_accuracy": {
        "llm": 90.0,
        "nlp": 95.0
      },
      "keyword_relevance": {
        "llm": 2540.0,
        "nlp": 16.66
      }
    },
    "coverage_analysis": {
      "llm_coverage": 7,
      "nlp_coverage": 60,
      "coverage_ratio": 0.12
    },
    "triangle_chart_data": {
      "llm_metrics": [
        70.0,
        60.0,
        35
      ],
      "nlp_metrics": [
        52.7,
        85.0,
        100
      ],
      "categories": [
        "Accuracy",
        "Speed",
        "Coverage"
      ]
    }
  },
  "accuracy_comparison": {
    "sentiment_comparison": {
      "llm_classification": "neutral",
      "nlp_classification": "neutral",
      "llm_confidence": 70.0,
      "nlp_confidence": 52.7,
      "agreement": true
    },
    "entity_comparison": {
      "llm_total_entities": 7,
      "nlp_total_entities": 60,
      "entity_overlap": 0.06666666666666667,
      "precision_score": 0.07
    },
    "keyword_comparison": {
      "llm_keyword_count": 10,
      "nlp_keyword_count": 10,
      "keyword_overlap_percentage": 11.1,
      "top_shared_keywords": [
        "stig",
        "bobby"
      ]
    },
    "overall_accuracy_scores": {
      "llm": 80.0,
      "nlp": 73.8
    }
  },
  "validation_summary": {
    "validation_status": "low",
    "recommendation": "Low agreement - Requires manual review and adjustment",
    "confidence_delta": 0.16,
    "overall_reliability": 66.5
  }
};

/* =========================
   Analyze simulation
========================= */
analyzeBtn.addEventListener("click", () => {
  uploadSection.classList.add("hidden");
  loadingSection.classList.remove("hidden");

  let progress = 0;
  const steps = [
    "Extracting keywords…",
    "Detecting sentiment & emotion…",
    "Identifying entities…",
    "Summarizing & classifying…",
    "Finalizing results…"
  ];
  let i = 0;

  const stepper = setInterval(() => {
    progress += 20;
    if (progress > 100) progress = 100;
    progressFill.style.width = progress + "%";
    loadingText.textContent = steps[i] || "Analyzing transcript…";
    i++;

    if (progress >= 100) {
      clearInterval(stepper);
      setTimeout(() => {
        loadingSection.classList.add("hidden");
        llmSection.classList.remove("hidden");
        renderMetadataDashboard();
        setupSynopsisObserver();
      }, 350);
    }
  }, 900);
});

/* =========================
   Render Metadata Dashboard
========================= */
let sentimentChart, keywordsChart, emotionsChart;

function renderMetadataDashboard() {
  // Lead characters
  // const lead = document.getElementById("leadCharacter");
  // lead.innerHTML = `<span class="pill-badge">${META_DATA.lead_characters[0].name}</span>`;
const lead = document.getElementById("leadCharacter");
lead.innerHTML = `
  <div class="flex flex-wrap gap-2">
    ${META_DATA.lead_characters
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 3)
      .map(c => `<span class="pill-badge">${c.name}</span>`)
      .join("")}
  </div>
`;
  // Genres
  const genreList = document.getElementById("genreList");
  genreList.innerHTML = META_DATA.genre_classification
    .map(
      g => `
      <li class="flex items-center justify-between">
        <span>${g.genre}</span>
        <span class="list-badge">${g.confidence}%</span>
      </li>`
    )
    .join("");

  // Content Classification
  const classificationList = document.getElementById("classificationList");
  classificationList.innerHTML = META_DATA.content_classification_plot.categories
    .map(
      (c, i) => `
      <li class="flex items-center justify-between">
        <span class="text-gray-700">${c}</span>
        <span class="dot"></span>
      </li>`
    )
    .join("");

  // Keywords (TOP 5 only)
  const topKeywords = META_DATA.keywords_plot.keywords.slice(0, 5);
  const topPercentages = META_DATA.keywords_plot.percentages.slice(0, 5);

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

  // Sentiment Pie
  const sctx = document.getElementById("sentimentChart");
  if (sentimentChart) sentimentChart.destroy();
  sentimentChart = new Chart(sctx, {
    type: "doughnut",
    data: {
      labels: META_DATA.sentiment_pie.labels,
      datasets: [{
        data: META_DATA.sentiment_pie.values,
        backgroundColor: ["#34d399", "#f87171", "#fbbf24"]
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "right" } },
      cutout: "60%"
    }
  });

  // NER Summary (counts + lists)
renderNER(META_DATA.named_entities);


  // Emotions PIE (instead of list)
  const emotionsCtx = document.getElementById("emotionsChart");
  if (emotionsChart) emotionsChart.destroy();
  emotionsChart = new Chart(emotionsCtx, {
    type: "pie",
    data: {
      labels: META_DATA.emotion_pie.labels,
      datasets: [{
        data: META_DATA.emotion_pie.values,
        backgroundColor: ["#60a5fa", "#f87171", "#9ca3af", "#fbbf24"]
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } }
    }
  });

  // Synopsis
  document.getElementById("synopsis").textContent = META_DATA.synopsis_summary.synopsis;
}


/* =========================
   NER Summary (counts + lists)
========================= */
function renderNER(namedEntities) {
  const nerDiv = document.getElementById("nerSummary");

  nerDiv.innerHTML = `
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center mb-6">
      <div class="p-3 rounded-lg bg-indigo-50">
        <p class="text-lg font-bold">${namedEntities.people_count}</p>
        <p class="text-xs text-gray-600">People</p>
      </div>
      <div class="p-3 rounded-lg bg-indigo-50">
        <p class="text-lg font-bold">${namedEntities.locations_count}</p>
        <p class="text-xs text-gray-600">Locations</p>
      </div>
      <div class="p-3 rounded-lg bg-indigo-50">
        <p class="text-lg font-bold">${namedEntities.organizations_count}</p>
        <p class="text-xs text-gray-600">Organizations</p>
      </div>
      <div class="p-3 rounded-lg bg-indigo-50">
        <p class="text-lg font-bold">${namedEntities.total}</p>
        <p class="text-xs text-gray-600">Total Entities</p>
      </div>
    </div>

    <div class="grid md:grid-cols-3 gap-6 text-sm text-left">
      <div>
        <h4 class="font-semibold mb-2">👤 People</h4>
        <ul class="list-disc list-inside space-y-1 text-gray-700">
          ${namedEntities.people.map(p => `<li>${p}</li>`).join("")}
        </ul>
      </div>

      <div>
        <h4 class="font-semibold mb-2">📍 Locations</h4>
        <ul class="list-disc list-inside space-y-1 text-gray-700">
          ${namedEntities.locations.map(l => `<li>${l}</li>`).join("")}
        </ul>
      </div>

      <div>
        <h4 class="font-semibold mb-2">🏛 Organizations</h4>
        <ul class="list-disc list-inside space-y-1 text-gray-700">
          ${namedEntities.organizations.map(o => `<li>${o}</li>`).join("")}
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
   Ad Insights rendering
========================= */
function renderAdInsights() {
  // Timeline chart
  const ctxA = document.getElementById("adChart");
  if (window._adChart) window._adChart.destroy();
  window._adChart = new Chart(ctxA, {
    type: "line",
    data: {
      labels: AD_DATA.ad_placement_timeline.placements.map(p => p.timestamp),
      datasets: [{
        label: "Suitability",
        data: AD_DATA.ad_placement_timeline.placements.map(p => p.suitability),
        borderColor: "#6366f1",
        backgroundColor: "rgba(99,102,241,0.15)",
        fill: true,
        tension: 0.35
      }]
    },
    options: { responsive: true, maintainAspectRatio: false }
  });

  // List
  const adList = document.getElementById("adList");
  adList.innerHTML = AD_DATA.ad_placement_timeline.placements
    .map(p => `<li>🕒 <b>${p.timestamp}</b> — ${p.scene} (${p.suitability}%)</li>`)
    .join("");

  // Recommendations
  const adRecList = document.getElementById("adRecList");
  adRecList.innerHTML = AD_DATA.ad_recommendations
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

/* =========================
   Evaluation Charts
========================= */


document.getElementById("toEvaluation").addEventListener("click", () => {
  adSection.classList.add("hidden");
  evaluationSection.classList.remove("hidden");
  renderEvaluation(EVAL_DATA);
});

function renderEvaluation(DATA) {
  // --- Overall Confidence Gauge ---
  const reliability = DATA.validation_summary.overall_reliability;
  makeChart("confidenceGauge", {
    type: "doughnut",
    data: {
      labels: ["Reliability", "Remaining"],
      datasets: [{
        data: [reliability, 100 - reliability],
        backgroundColor: ["#34d399", "#e5e7eb"],
        cutout: "80%"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } }
    }
  });

  // --- Confusion Matrix (Chart + Table) ---
  const venn = DATA.confidence_scores.venn_data;
  const total = venn.total;

  // Chart
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

  // Table
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

  // --- Radar Chart ---
  makeChart("triangleRadar", {
    type: "radar",
    data: {
      labels: DATA.performance_metrics.triangle_chart_data.categories,
      datasets: [
        {
          label: "LLM",
          data: DATA.performance_metrics.triangle_chart_data.llm_metrics,
          borderColor: "#6366f1",
          backgroundColor: "rgba(99,102,241,0.2)"
        },
        {
          label: "NLP",
          data: DATA.performance_metrics.triangle_chart_data.nlp_metrics,
          borderColor: "#34d399",
          backgroundColor: "rgba(52,211,153,0.2)"
        }
      ]
    },
    options: { responsive: true, maintainAspectRatio: false }
  });
}


// Keep refs globally so we can destroy on re-render
const evalCharts = {};

function makeChart(id, cfg) {
  const el = document.getElementById(id);
  if (evalCharts[id]) { evalCharts[id].destroy(); }
  evalCharts[id] = new Chart(el.getContext('2d'), cfg);
}

document.getElementById("toEvaluation").addEventListener("click", () => {
  adSection.classList.add("hidden");
  evaluationSection.classList.remove("hidden");
  renderEvaluation(EVAL_DATA);

  // after first paint, ensure charts pick up final sizes
  requestAnimationFrame(() => {
    Object.values(evalCharts).forEach(c => c.resize());
  });
});


/* =========================
   Back nav
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


// Configuration
const API_BASE_URL = 'http://localhost:8000/api/v1';
const DATASET_URL_BASE = 'http://localhost:3000/data/fashionIQ_dataset/images';

// Generate a unique session ID for tracking
const sessionId = crypto.randomUUID();

// DOM Elements
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const textInput = document.getElementById('textInput');
const dressType = document.getElementById('dressType');
const searchBtn = document.getElementById('searchBtn');
const resultsGrid = document.getElementById('resultsGrid');
const statusMsg = document.getElementById('statusMsg');

let currentQueryImageName = "";

// Image Preview Handler
imageInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    imagePreview.src = URL.createObjectURL(file);
    imagePreview.hidden = false;
    // Strip extension to store the raw name for tracking
    currentQueryImageName = file.name.replace(/\.[^/.]+$/, "");
  }
});

// Search API Call
searchBtn.addEventListener('click', async () => {
  const file = imageInput.files[0];
  const query = textInput.value.trim();
  const type = dressType.value;

  if (!file || !query) {
    statusMsg.textContent = "Please provide both an image and text request.";
    statusMsg.style.color = "red";
    return;
  }

  statusMsg.textContent = "Searching vector database...";
  statusMsg.style.color = "var(--text-muted)";
  resultsGrid.innerHTML = "";

  const formData = new FormData();
  formData.append('image', file);
  formData.append('text_query', query);
  formData.append('dress_type', type);
  formData.append('top_k', 10);

  try {
    const response = await fetch(`${API_BASE_URL}/search`, {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (response.ok) {
      renderResults(data.results);
      statusMsg.textContent = `Found ${data.results.length} matches.`;
    } else {
      throw new Error(data.detail || "Search failed");
    }
  } catch (error) {
    console.error(error);
    statusMsg.textContent = "Error: " + error.message;
    statusMsg.style.color = "red";
  }
});

// Render Results Grid
function renderResults(results) {
  results.forEach((item, index) => {
    const card = document.createElement('div');
    card.className = 'result-card';
    // When clicked, track the interaction
    card.onclick = () => handleImageSelection(card, item.image_name);

    const img = document.createElement('img');
    // Construct the image URL based on your dataset location
    img.src = `${DATASET_URL_BASE}/${item.image_name}.png`;
    img.alt = item.image_name;
    img.onerror = () => { img.src = "https://via.placeholder.com/200x250?text=Image+Not+Found"; };

    img.onerror = function () {
      this.onerror = null;
      this.src = "https://via.placeholder.com/200x250?text=Image+Not+Found";
    };
    
    const info = document.createElement('div');
    info.className = 'result-info';
    info.textContent = `Score: ${item.score.toFixed(3)}`;

    card.appendChild(img);
    card.appendChild(info);
    resultsGrid.appendChild(card);
  });
}

// Track User Interaction for MLOps Pipeline
async function handleImageSelection(selectedCard, targetImageName) {
  // Visually highlight the selection
  document.querySelectorAll('.result-card').forEach(c => c.classList.remove('selected'));
  selectedCard.classList.add('selected');

  const payload = {
    session_id: sessionId,
    query_text: textInput.value.trim(),
    reference_image_name: currentQueryImageName,
    selected_target_name: targetImageName
  };

  try {
    await fetch(`${API_BASE_URL}/track`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    console.log("Interaction logged successfully for fine-tuning.");
  } catch (error) {
    console.error("Failed to log interaction:", error);
  }
}
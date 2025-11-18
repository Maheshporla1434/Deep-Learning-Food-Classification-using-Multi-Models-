const modelKeys = {{ model_keys|tojson }};  // Pass model keys from Flask

function updateModelOptions() {
    const modelType = document.getElementById('modelType').value;
    const modelSelect = document.getElementById('model');
    modelSelect.innerHTML = '<option value="">Select Model</option>';
    modelSelect.disabled = !modelType;

    modelKeys.forEach(key => {
        if (key.startsWith(modelType)) {
            const option = document.createElement('option');
            option.value = key;
            option.text = key.replace(modelType + '_', '');
            modelSelect.appendChild(option);
        }
    });
}

function showClassMetrics(className) {
    fetch(`/get_class_metrics/${encodeURIComponent(className)}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('metricsContent').textContent = data.error;
            } else {
                let content = '';
                for (const [model, metrics] of Object.entries(data)) {
                    content += `${model}: Precision: ${metrics.precision || 'N/A'}, Recall: ${metrics.recall || 'N/A'}\n`;
                }
                document.getElementById('metricsContent').textContent = content;
            }
            document.getElementById('classMetrics').style.display = 'block';
        });
}

document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultsDiv = document.getElementById('results');
        if (data.error) {
            resultsDiv.innerHTML = `<p>Error: ${data.error}</p>`;
        } else {
            let matrixHtml = '<h3>Confusion Matrix</h3><table>';
            data.confusion_matrix.forEach(row => {
                matrixHtml += '<tr>';
                row.forEach(cell => matrixHtml += `<td>${cell}</td>`);
                matrixHtml += '</tr>';
            });
            matrixHtml += '</table>';

            resultsDiv.innerHTML = `
                <img src="${data.image_path}" alt="Uploaded Image"><br>
                <p><strong>Predicted Class:</strong> ${data.predicted_class}</p>
                <p><strong>Confidence:</strong> ${data.confidence}%</p>
                ${matrixHtml}
                <h3>Classification Report</h3>
                <pre>${data.classification_report}</pre>
            `;
        }
    })
    .catch(error => {
        document.getElementById('results').innerHTML = '<p>Error: Failed to process request.</p>';
    });
});
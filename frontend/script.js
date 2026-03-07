document.getElementById('predictBtn').addEventListener('click', async () => {
    const raw = document.getElementById('ecgData').value;
    const signal = raw.split(',').map(x => parseFloat(x.trim())).filter(x => !isNaN(x));
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({signal})
    });
    const data = await response.json();
    document.getElementById('result').textContent =
        `Prediction: ${data.prediction}, confidence: ${data.confidence}`;
});
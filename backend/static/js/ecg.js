async function loadECG(id) {
    const resp = await fetch(`/api/ecg/${id}`);
    const data = await resp.json();
    const ctx = document.getElementById('ecgChart').getContext('2d');
    if (window.ecgChart && typeof window.ecgChart.destroy === 'function') {
        window.ecgChart.destroy();
    }
    const dark = document.body.classList.contains('dark');
    // make canvas wider/taller if needed (css can adjust height)
    window.ecgChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.signal.map((_,i)=>i),
            datasets: [{ label: 'ECG', data: data.signal, borderColor: getComputedStyle(document.body).getPropertyValue('--ecg-line'), backgroundColor: getComputedStyle(document.body).getPropertyValue('--ecg-line'), fill:false, tension:0.1 }]
        },
        options: { 
            maintainAspectRatio: false,
            scales:{
                x:{
                    grid:{color: 'transparent'},
                    ticks:{maxRotation:0,minRotation:0,color: getComputedStyle(document.body).getPropertyValue('--color-text')}
                },
                y:{grid:{color: 'transparent'},ticks:{color: getComputedStyle(document.body).getPropertyValue('--color-text')}}
            }
        }
    });
}

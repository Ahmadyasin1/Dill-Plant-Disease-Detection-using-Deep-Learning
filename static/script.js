document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const formData = new FormData(this);
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    const predictionText = document.getElementById('prediction-text');
    const uploadedImage = document.getElementById('uploaded-image');

    if (result.prediction) {
        predictionText.innerText = `Prediction: ${result.prediction}`;
        uploadedImage.style.display = 'block';
        uploadedImage.src = URL.createObjectURL(formData.get('file'));
    } else {
        predictionText.innerText = 'Prediction failed.';
        uploadedImage.style.display = 'none';
    }
});

document.getElementById('reset-form').addEventListener('click', function() {
    document.getElementById('upload-form').reset();
    document.getElementById('prediction-text').innerText = 'Prediction will appear here.';
    document.getElementById('uploaded-image').style.display = 'none';
});

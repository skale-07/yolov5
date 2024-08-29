document.addEventListener('DOMContentLoaded', () => {
    initializeDropdown();
    handleFacialDetectionCheckbox();
});

function initializeDropdown() {
    fetch('/get_common_objects')
        .then(response => response.json())
        .then(data => {
            const commonObjects = data.common_objects;
            const dropdown = document.getElementById('object-dropdown');
            dropdown.innerHTML = '<option value="" disabled selected>Select an object</option>';
            commonObjects.forEach(object => {
                const option = document.createElement('option');
                option.value = object;
                option.textContent = object;
                dropdown.appendChild(option);
            });
        })
        .catch(error => console.error('Error:', error));
}

function toggleSelection(imgElement) {
    const images = document.querySelectorAll('.uploaded-images img');
    if (!lastClickedImage || !event.shiftKey) {
        imgElement.classList.toggle('selected');
        lastClickedImage = imgElement.classList.contains('selected') ? imgElement : null; 
    } else {
        const startIndex = images.indexOf(lastClickedImage);
        const endIndex = images.indexOf(imgElement);
        const [from, to] = startIndex < endIndex ? [startIndex, endIndex] : [endIndex, startIndex];
        for (let i = from; i <= to; i++) images[i].classList.add('selected');
    }
}

function logSelectedImages() {
    const selectedImages = Array.from(document.querySelectorAll('.uploaded-images img.selected')).map(img => img.src);
    console.log(selectedImages);
}

function clearSelectedImages() {
    const selectedImages = [...document.querySelectorAll('.uploaded-images img.selected')].map(img => img.src);
    fetch('/clear_selected', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imagePaths: selectedImages }),
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message);
        selectedImages.forEach(imgSrc => {
            const imgElement = document.querySelector(`img[src="${imgSrc}"]`);
            if (imgElement) imgElement.remove();
        });
    })
    .catch(error => console.error('Error:', error));
}

function downloadSelectedImages() {
    const selectedImages = Array.from(document.querySelectorAll('.uploaded-images img.selected')).map(img => img.src);
    selectedImages.forEach(imgSrc => {
        const link = document.createElement('a');
        link.href = imgSrc;
        link.download = imgSrc.split('/').pop();
        link.click();
    });
}

function updateSliderValue(value) {
    document.getElementById('slider-value').textContent = value;
    document.getElementById('hidden-slider').value = value;
}

document.addEventListener('DOMContentLoaded', function() {
    var deleteButton = document.getElementById('delete-all-btn');
    if(deleteButton) {
        deleteButton.addEventListener('click', function(event) {
            var dropdown = document.getElementById('object-dropdown');
            while (dropdown.firstChild) {
                dropdown.removeChild(dropdown.firstChild);
            }
            console.log('Dropdown reset'); // Log to the console

            // Send a request to the server to clear the session data
            fetch('/clear_objects', {
                method: 'POST',
            })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
            })
            .catch(error => console.error('Error:', error));

            return false; // Prevent the form from being submitted
        });
    } else {
        console.log('Delete button not found');
    }
});

function openPopup() {
    const popup = document.getElementById('facialDetectionPopup');
    const container = document.getElementById('referenceImagesContainer');
    container.innerHTML = ''; // Clear existing images

    // Fetch reference images from the backend
    fetch('/load_reference_images')
        .then(response => response.json())
        .then(data => {
            // Check if reference_images is false
            data.reference_images.forEach((imagePath, index) => {
                const div = document.createElement('div');
                div.classList.add('reference-image');

                const img = document.createElement('img');
                img.src = imagePath;
                img.alt = `Reference ${index + 1}`;
                img.onclick = () => toggleSelection(img);

                div.appendChild(img);
                container.appendChild(div);
            });
            popup.style.display = 'flex';
        })
        .catch(error => console.error('Error loading reference images:', error));
}

function closePopup() {
    const popup = document.getElementById('facialDetectionPopup');
    popup.style.display = 'none';
}


function confirmSelection() {
    const selectedImages = Array.from(document.querySelectorAll('.reference-image img.selected')).map(img => img.src);
    const selectedFaces = selectedImages.map(imgSrc => imgSrc.split('/').pop()); // Extract the filenames

    fetch('/sort_by_selected_faces', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ selectedFaces: selectedFaces }),
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message); // Handle success message
        // You can redirect or update the UI here based on the sorted images
        if (data.redirect) {
            window.location.href = data.redirect; // Redirect if necessary
        }
    })
    .catch(error => console.error('Error:', error));
}



function checkFacialDetection() {
    const facialDetectionCheckbox = document.getElementById('facial-detection-checkbox');
    if (facialDetectionCheckbox.checked) {
        openPopup();
    }
}

window.onload = function() {
    document.getElementById('facial-detection-checkbox').addEventListener('change', checkFacialDetection);
};
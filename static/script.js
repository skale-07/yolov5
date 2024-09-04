// This function will handle the display of uploaded images
document.getElementById('uploadFile').onchange = function (event) {
    const files = event.target.files;  // Get all selected files
    let main = document.querySelector(".mainbody");
    main.innerHTML = '';  // Clear previous images

    // Loop through each file and create an image preview
    for (const file of files) {
        if (file) {
            var img = document.createElement('img');
            img.classList.add("preview");
            img.src = URL.createObjectURL(file);  // Create a URL for the image
            img.style.display = 'block';
            img.style.margin = '10px';
            main.appendChild(img);  // Append the image to the mainbody
        }
    }
}

// Sync the range input with the number input
document.getElementById("num-people").oninput = function(){
    document.getElementById("number-people-label").value = this.value;
}

// Sync the number input with the range input
document.getElementById("number-people-label").oninput = function(){
    document.getElementById("num-people").value = this.value;
}

// Function to collapse or expand the sidebar
function collapse(){
    const sidebody = document.querySelector(".sidebody");
    const collapseBtn = document.querySelector("#collapse");
    sidebody.classList.toggle("collapsed");
    collapseBtn.classList.toggle("collapsed");
}

// Function to delete all preview images
function deleteAll(){
    const imgs = document.querySelectorAll(".preview");
    imgs.forEach(img => img.remove());  // Remove each image from the DOM
}
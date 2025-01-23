// change bg of navbar on scroll
const navbar = document.getElementById('navbar')
window.onscroll = function () {
    // Get the current scroll position
    const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;

    if (scrollTop >= 200) {
        navbar.classList.add("bg-green");
        navbar.classList.remove("bg-transparent");
    } else {
        navbar.classList.add("bg-transparent");
        navbar.classList.remove("bg-green");
    }
};

// JS to handle the input images and send it to flask app
 document.getElementById('uploadForm').addEventListener('submit', function(event) {
            event.preventDefault();

            const formData = new FormData(this);
            const loadingIndicator = document.getElementById('loading');

            // Show loading indicator
            loadingIndicator.style.display = 'flex'; // Use 'flex' to center the content
             console.log('Loading indicator shown');

            fetch('/predict', {method: 'POST', body: formData})
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok ' + response.statusText);
                    }
                    return response.text();
                })
                .then(html => {
                    // Hide loading indicator
                    loadingIndicator.style.display = 'none';
                    console.log('response recieve, loading')
                    document.open();
                    document.write(html);
                    document.close();
                })
                .catch(error => {
                    // Hide loading indicator in case of error
                    loadingIndicator.style.display = 'none';
                    console.error('Error:', error);
                });
        });

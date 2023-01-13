document.addEventListener("DOMContentLoaded", function (ev) {
    document.querySelector('.form').addEventListener('submit', (e) => {
    e.preventDefault();
    // get the form data
    const formData = new FormData(e.target);
    const json = {}
    for (const [key, value]  of formData.entries()) {
        json[key] = value
    }

    fetch('http://localhost:5000/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(json)
    })
        .then(response => response.json())
        .then(data => {
            // update the UI with the response
            let result = "No Fraud"
            if(data.prediction == 1) {
                result = "Fraud"
            }
            console.log(result)
            document.getElementById('result').innerHTML = result;
        })
        .catch(error => console.error(error));
});

})

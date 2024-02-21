// content.js

document.addEventListener("mouseup", function() {
  var selectedText = window.getSelection().toString().trim();
  if (selectedText !== "") {
    fetch('https://f8zwx9odu94zinata9n8ws.streamlit.app/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ selected_text: selectedText }),
    })
    .then(response => response.json())
    .then(data => {
      // Process the response from the Streamlit app
      console.log('Binary Result:', data.binary_result);
      console.log('Multi-Class Result:', data.multi_class_result);
      // Display or use the results as needed
    })
    .catch(error => console.error('Error:', error));
  }
});

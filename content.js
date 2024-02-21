// content.js

document.addEventListener("mouseup", function() {
  var selectedText = window.getSelection().toString().trim();
  if (selectedText !== "") {
    fetch('http://your-streamlit-app-url/highlighted_text', {
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

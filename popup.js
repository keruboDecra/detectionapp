// popup.js

document.addEventListener('DOMContentLoaded', function () {
  var detectButton = document.getElementById('detectButton');

  detectButton.addEventListener('click', function () {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      var activeTab = tabs[0];
      var selectedText = '';

      chrome.tabs.sendMessage(activeTab.id, { type: 'getSelectedText' }, function (response) {
        if (response && response.selectedText) {
          selectedText = response.selectedText;

          // Send the selected text to your server for processing
          fetch('https://f8zwx9odu94zinata9n8ws.streamlit.app/', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: selectedText }),
          })
            .then(response => response.json())
            .then(data => {
              // Handle the response from your server (display in the popup.html)
              var resultContainer = document.getElementById('result-container');
              var resultElement = document.getElementById('result');

              if (data.binary_result == 1) {
                resultElement.innerHTML = `Binary Cyberbullying Prediction: Cyberbullying`;
              } else {
                resultElement.innerHTML = `Binary Cyberbullying Prediction: Not Cyberbullying`;
              }

              if (data.offensive_words.length > 0) {
                resultElement.innerHTML += `<br>Detected offensive words: ${data.offensive_words.join(', ')}`;
              }

              if (data.multi_class_result) {
                resultElement.innerHTML += `<br>Multi-Class Predicted Class: ${data.multi_class_result}`;
              }

              resultContainer.style.display = 'block';
            })
            .catch(error => console.error('Error:', error));
        }
      });
    });
  });
});

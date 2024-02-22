chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  // Send the selected text to your server for processing
  fetch('https://your-server-endpoint', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text: request.text }),
  })
    .then(response => response.json())
    .then(data => {
      // Handle the response from your server (display in popup.html)
      chrome.runtime.sendMessage({ result: data });
    })
    .catch(error => console.error('Error:', error));
});

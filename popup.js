chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  // Update the popup with the results from the server
  document.getElementById('result').innerText = JSON.stringify(request.result);
});

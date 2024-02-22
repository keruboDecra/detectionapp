chrome.contextMenus.create({
  title: "Detect Cyberbullying",
  contexts: ["selection"],
  onclick: function (info, tab) {
    chrome.tabs.sendMessage(tab.id, { text: info.selectionText });
  }
});

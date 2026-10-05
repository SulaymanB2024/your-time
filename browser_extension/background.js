const HOST = 'com.sulayman.personal_activity_ledger';
let lastEvent = '';
let lastEventAt = 0;

function send(payload) {
  chrome.runtime.sendNativeMessage(HOST, payload, () => {
    void chrome.runtime.lastError; // The local host may not yet be installed.
  });
}
function activeTab(tab, eventType) {
  if (!tab || !tab.active || tab.incognito || !tab.url) return;
  let url;
  try { url = new URL(tab.url); } catch { return; }
  if (!['http:', 'https:'].includes(url.protocol)) return;
  const identity = `${eventType}:${tab.id}:${tab.windowId}:${url.hostname}:${tab.title || ''}`;
  if (identity === lastEvent && Date.now() - lastEventAt < 3000) return;
  lastEvent = identity;
  lastEventAt = Date.now();
  send({kind: 'tab_event', event_type: eventType, at: new Date().toISOString(),
    domain: url.hostname, title: (tab.title || '').slice(0, 180),
    audible: !!tab.audible, incognito: false, tab_id: tab.id,
    window_id: tab.windowId});
}
chrome.tabs.onActivated.addListener(async ({tabId}) => {
  try { activeTab(await chrome.tabs.get(tabId), 'tab_active'); } catch { /* closed tab */ }
});
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (tab.active && (changeInfo.status === 'complete' || changeInfo.title || changeInfo.url)) {
    activeTab(tab, 'tab_updated');
  }
});
chrome.windows.onFocusChanged.addListener(async windowId => {
  if (windowId === chrome.windows.WINDOW_ID_NONE) {
    send({kind: 'tab_event', event_type: 'browser_blur', at: new Date().toISOString(),
      domain: null, title: null, audible: false, incognito: false,
      tab_id: 0, window_id: 0});
    return;
  }
  try {
    const tabs = await chrome.tabs.query({active: true, windowId});
    if (tabs[0]) activeTab(tabs[0], 'window_focused');
  } catch { /* window closed */ }
});
chrome.tabs.query({active: true, lastFocusedWindow: true}, tabs => {
  if (tabs[0]) activeTab(tabs[0], 'window_focused');
});

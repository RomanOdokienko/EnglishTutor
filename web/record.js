(function () {
  var API_BASE = String(window.ENGLISH_TUTOR_API_BASE_URL || "").replace(/\/+$/, "");
  var startBtn = document.getElementById("rec-start");
  var stopBtn = document.getElementById("rec-stop");
  var statusEl = document.getElementById("rec-status");
  var fileInput = document.getElementById("rec-file");
  var dateInput = document.getElementById("rec-date");
  var recoveryEl = document.getElementById("rec-recovery");
  var retryBtn = document.getElementById("rec-retry");
  var downloadLink = document.getElementById("rec-download");
  var localNote = document.getElementById("rec-local-note");

  dateInput.value = new Date().toISOString().slice(0, 10);

  var micStream = null;
  var displayStream = null;
  var audioCtx = null;
  var recorder = null;
  var chunks = [];
  var pendingUpload = null;
  var recordingUrl = "";

  function escapeHtml(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function setStatus(html, kind) {
    statusEl.style.display = "block";
    statusEl.className = "rec-status" + (kind ? " " + kind : "");
    statusEl.innerHTML = html;
  }

  function pickMime() {
    var candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"];
    for (var i = 0; i < candidates.length; i++) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(candidates[i])) {
        return { mimeType: candidates[i] };
      }
    }
    return {};
  }

  function cleanupTracks() {
    [micStream, displayStream].forEach(function (stream) {
      if (stream) stream.getTracks().forEach(function (track) { track.stop(); });
    });
    if (audioCtx) { try { audioCtx.close(); } catch (error) {} }
    micStream = null;
    displayStream = null;
    audioCtx = null;
  }

  function safeFilenamePart(value, fallback) {
    var cleaned = String(value || "").trim()
      .replace(/[^a-zA-Z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "");
    return cleaned || fallback;
  }

  function recordingFilename(ext) {
    var date = dateInput.value || new Date().toISOString().slice(0, 10);
    var you = safeFilenamePart(document.getElementById("rec-you").value, "speaker-a");
    var partner = safeFilenamePart(document.getElementById("rec-partner").value, "speaker-b");
    return "english-tutor-" + date + "-" + you + "-" + partner + "." + ext;
  }

  function keepLocalCopy(blob, ext) {
    if (recordingUrl) URL.revokeObjectURL(recordingUrl);
    recordingUrl = URL.createObjectURL(blob);
    var filename = recordingFilename(ext);
    downloadLink.href = recordingUrl;
    downloadLink.download = filename;
    downloadLink.hidden = false;
    recoveryEl.hidden = false;
    localNote.textContent = "Local backup: " + filename;

    // The visible link remains available if browser download settings block
    // this automatic save.
    downloadLink.click();
    return filename;
  }

  function showRetry(show) {
    retryBtn.hidden = !show;
    recoveryEl.hidden = !show && downloadLink.hidden;
  }

  async function start() {
    try {
      showRetry(false);
      setStatus("Requesting microphone…", null);
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

      setStatus("Choose <strong>Entire screen</strong> and tick <strong>Share system audio</strong>…", null);
      displayStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });

      var sysTracks = displayStream.getAudioTracks();
      displayStream.getVideoTracks().forEach(function (track) { track.stop(); });
      if (!sysTracks.length) {
        cleanupTracks();
        setStatus("No system audio was shared. Restart and tick <strong>Share system audio</strong> (pick Entire screen).", "err");
        return;
      }

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      var micSrc = audioCtx.createMediaStreamSource(micStream);
      var sysSrc = audioCtx.createMediaStreamSource(new MediaStream([sysTracks[0]]));
      var merger = audioCtx.createChannelMerger(2);
      micSrc.connect(merger, 0, 0);
      sysSrc.connect(merger, 0, 1);
      var dest = audioCtx.createMediaStreamDestination();
      merger.connect(dest);

      chunks = [];
      recorder = new MediaRecorder(dest.stream, pickMime());
      recorder.ondataavailable = function (event) {
        if (event.data && event.data.size) chunks.push(event.data);
      };
      recorder.onstop = function () {
        var type = recorder.mimeType || "audio/webm";
        var ext = type.indexOf("ogg") >= 0 ? "ogg" : "webm";
        var blob = new Blob(chunks, { type: type });
        cleanupTracks();
        var filename = keepLocalCopy(blob, ext);
        pendingUpload = { blob: blob, ext: ext, filename: filename };
        sendBlob(blob, ext);
      };
      sysTracks[0].addEventListener("ended", function () {
        if (recorder && recorder.state === "recording") stop();
      });

      recorder.start(1000);
      startBtn.disabled = true;
      stopBtn.disabled = false;
      setStatus('<span class="rec-dot"></span>Recording… two channels (you + partner).', null);
    } catch (error) {
      cleanupTracks();
      setStatus("Could not start recording: " + escapeHtml(error && error.message ? error.message : error), "err");
    }
  }

  function stop() {
    stopBtn.disabled = true;
    if (recorder && recorder.state !== "inactive") {
      setStatus("Finishing recording and saving a local copy…", null);
      recorder.stop();
    }
  }

  function buildQuery() {
    var params = new URLSearchParams();
    params.set("date", dateInput.value || "");
    params.set("topic", document.getElementById("rec-topic").value || "Recorded session");
    params.set("recorder", document.getElementById("rec-you").value || "Speaker A");
    params.set("other", document.getElementById("rec-partner").value || "Speaker B");
    return params.toString();
  }

  async function sendBlob(blob, ext) {
    showRetry(false);
    retryBtn.disabled = true;
    setStatus("Uploading " + Math.round(blob.size / 1024) + " KB and transcribing… this can take a minute or two.", null);
    try {
      var url = API_BASE + "/api/upload-audio?" + buildQuery() + "&ext=" + encodeURIComponent(ext);
      var response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: blob,
      });
      var text = await response.text();
      if (!response.ok) {
        setStatus("Failed: " + escapeHtml(text) + "<br>Your recording is safe locally. You can retry the upload.", "err");
        showRetry(true);
        return;
      }

      var data = {};
      try { data = JSON.parse(text); } catch (error) {}
      var channelCount = (data.channels || []).length;
      var message = "Done — session <strong>" + escapeHtml(data.date || "") + "</strong> is ready. "
        + "Detected <strong>" + channelCount + "</strong> channel(s). "
        + '<a href="highlights.html">Open Session</a>.';
      if (data.analysis_status === "processing") {
        message += "<br>Transcript is saved. Metrics and annotations are finishing in the background.";
      }
      showRetry(false);
      if (data.warning) setStatus(message + "<br>⚠️ " + escapeHtml(data.warning), "warn");
      else setStatus(message, "ok");
    } catch (error) {
      setStatus("Upload error: " + escapeHtml(error && error.message ? error.message : error)
        + "<br>Your recording is safe locally. You can retry the upload.", "err");
      showRetry(true);
    } finally {
      retryBtn.disabled = false;
      startBtn.disabled = false;
    }
  }

  startBtn.addEventListener("click", start);
  stopBtn.addEventListener("click", stop);
  retryBtn.addEventListener("click", function () {
    if (pendingUpload) sendBlob(pendingUpload.blob, pendingUpload.ext);
  });
  fileInput.addEventListener("change", function () {
    var file = fileInput.files && fileInput.files[0];
    if (!file) return;
    var ext = (file.name.split(".").pop() || "webm").toLowerCase();
    pendingUpload = { blob: file, ext: ext, filename: file.name };
    localNote.textContent = "Original file remains on your computer.";
    sendBlob(file, ext);
  });
  window.addEventListener("beforeunload", function () {
    if (recordingUrl) URL.revokeObjectURL(recordingUrl);
  });
})();

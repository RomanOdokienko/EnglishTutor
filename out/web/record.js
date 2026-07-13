(function () {
  var API_BASE = (window.ENGLISH_TUTOR_API_BASE_URL || "").replace(/\/$/, "");
  var startBtn = document.getElementById("rec-start");
  var stopBtn = document.getElementById("rec-stop");
  var statusEl = document.getElementById("rec-status");
  var fileInput = document.getElementById("rec-file");
  var dateInput = document.getElementById("rec-date");

  dateInput.value = new Date().toISOString().slice(0, 10);

  var micStream = null, displayStream = null, audioCtx = null, recorder = null, chunks = [];

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
    [micStream, displayStream].forEach(function (s) {
      if (s) s.getTracks().forEach(function (t) { t.stop(); });
    });
    if (audioCtx) { try { audioCtx.close(); } catch (e) {} }
    micStream = displayStream = audioCtx = null;
  }

  async function start() {
    try {
      setStatus("Requesting microphone…", null);
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

      setStatus("Choose <strong>Entire screen</strong> and tick <strong>Share system audio</strong>…", null);
      displayStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });

      var sysTracks = displayStream.getAudioTracks();
      displayStream.getVideoTracks().forEach(function (t) { t.stop(); });
      if (!sysTracks.length) {
        cleanupTracks();
        setStatus("No system audio was shared. Restart and tick <strong>Share system audio</strong> (pick Entire screen).", "err");
        return;
      }

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      var micSrc = audioCtx.createMediaStreamSource(micStream);
      var sysSrc = audioCtx.createMediaStreamSource(new MediaStream([sysTracks[0]]));
      var merger = audioCtx.createChannelMerger(2);
      micSrc.connect(merger, 0, 0);   // mic  -> left  (channel 1 = you)
      sysSrc.connect(merger, 0, 1);   // call -> right (channel 2 = partner)
      var dest = audioCtx.createMediaStreamDestination();
      merger.connect(dest);

      chunks = [];
      recorder = new MediaRecorder(dest.stream, pickMime());
      recorder.ondataavailable = function (e) { if (e.data && e.data.size) chunks.push(e.data); };
      recorder.onstop = function () {
        var type = recorder.mimeType || "audio/webm";
        var blob = new Blob(chunks, { type: type });
        cleanupTracks();
        sendBlob(blob, "webm");
      };
      // If the user stops screen-share from the browser bar, end the recording.
      sysTracks[0].addEventListener("ended", function () { if (recorder && recorder.state === "recording") stop(); });

      recorder.start(1000);
      startBtn.disabled = true;
      stopBtn.disabled = false;
      setStatus('<span class="rec-dot"></span>Recording… two channels (you + partner).', null);
    } catch (err) {
      cleanupTracks();
      setStatus("Could not start recording: " + (err && err.message ? err.message : err), "err");
    }
  }

  function stop() {
    stopBtn.disabled = true;
    if (recorder && recorder.state !== "inactive") {
      setStatus("Finishing recording…", null);
      recorder.stop();
    }
  }

  function buildQuery() {
    var p = new URLSearchParams();
    p.set("date", dateInput.value || "");
    p.set("topic", document.getElementById("rec-topic").value || "Recorded session");
    p.set("recorder", document.getElementById("rec-you").value || "Speaker A");
    p.set("other", document.getElementById("rec-partner").value || "Speaker B");
    return p.toString();
  }

  async function sendBlob(blob, ext) {
    setStatus("Uploading " + Math.round(blob.size / 1024) + " KB and transcribing… this can take a minute or two.", null);
    try {
      var url = API_BASE + "/api/upload-audio?" + buildQuery() + "&ext=" + encodeURIComponent(ext);
      var res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: blob });
      var text = await res.text();
      if (!res.ok) { setStatus("Failed: " + text, "err"); startBtn.disabled = false; return; }
      var data = {};
      try { data = JSON.parse(text); } catch (e) {}
      var ch = (data.channels || []).length;
      var msg = "Done — session <strong>" + (data.date || "") + "</strong> is ready. " +
                "Detected <strong>" + ch + "</strong> channel(s). " +
                '<a href="highlights.html">Open Highlights</a>.';
      if (data.warning) { setStatus(msg + "<br>⚠️ " + data.warning, "warn"); }
      else { setStatus(msg, "ok"); }
    } catch (err) {
      setStatus("Upload error: " + (err && err.message ? err.message : err), "err");
    } finally {
      startBtn.disabled = false;
    }
  }

  startBtn.addEventListener("click", start);
  stopBtn.addEventListener("click", stop);
  fileInput.addEventListener("change", function () {
    var f = fileInput.files && fileInput.files[0];
    if (!f) return;
    var ext = (f.name.split(".").pop() || "webm").toLowerCase();
    sendBlob(f, ext);
  });
})();

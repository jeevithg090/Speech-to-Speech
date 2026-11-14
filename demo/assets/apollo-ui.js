/*
 * Apollo Voice Engine — small shared helpers for the demo UIs.
 * No dependencies; exposes a single global, `ApolloUI`.
 */
(function () {
    "use strict";

    var THEME_KEY = "apollo-theme";
    var darkQuery = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

    function storedTheme() {
        try {
            return localStorage.getItem(THEME_KEY);
        } catch (e) {
            return null;
        }
    }

    function effectiveTheme() {
        var explicit = document.documentElement.getAttribute("data-theme");
        if (explicit === "light" || explicit === "dark") return explicit;
        return darkQuery && darkQuery.matches ? "dark" : "light";
    }

    function syncThemeToggles() {
        var next = effectiveTheme() === "dark" ? "light" : "dark";
        document.querySelectorAll("[data-theme-toggle]").forEach(function (btn) {
            btn.setAttribute("aria-label", "Switch to " + next + " theme");
            btn.setAttribute("title", "Switch to " + next + " theme");
        });
        var meta = document.querySelector('meta[name="theme-color"]');
        if (meta) meta.setAttribute("content", effectiveTheme() === "dark" ? "#0b1220" : "#f4f7fa");
    }

    function toggleTheme() {
        var next = effectiveTheme() === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        try {
            localStorage.setItem(THEME_KEY, next);
        } catch (e) {
            /* storage unavailable: the choice lasts for this page view */
        }
        syncThemeToggles();
    }

    /**
     * Wire a group of toggle buttons (`[data-value]`, `aria-pressed`) so that
     * exactly one is pressed. Arrow keys move between options.
     */
    function singleSelect(group, onChange) {
        var options = Array.prototype.slice.call(group.querySelectorAll("[data-value]"));
        function select(option, focus) {
            options.forEach(function (o) {
                o.setAttribute("aria-pressed", o === option ? "true" : "false");
            });
            if (focus) option.focus();
            if (onChange) onChange(option.getAttribute("data-value"), option);
        }
        options.forEach(function (option, index) {
            option.addEventListener("click", function () {
                select(option, false);
            });
            option.addEventListener("keydown", function (event) {
                var delta = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
                if (!delta) return;
                event.preventDefault();
                select(options[(index + delta + options.length) % options.length], true);
            });
        });
        var initial = options.find(function (o) {
            return o.getAttribute("aria-pressed") === "true";
        });
        return {
            value: function () {
                var pressed = options.find(function (o) {
                    return o.getAttribute("aria-pressed") === "true";
                });
                return pressed ? pressed.getAttribute("data-value") : initial && initial.getAttribute("data-value");
            },
        };
    }

    /** Turn a FastAPI/HTTP error body into a sentence a person can act on. */
    function describeError(status, body) {
        var detail = body && (body.detail || body.error);
        if (typeof detail === "string" && detail) return detail;
        if (Array.isArray(detail) && detail.length && detail[0].msg) {
            return "Please check your input: " + detail[0].msg.replace(/^Value error, /, "") + ".";
        }
        if (status === 413) return "That upload is too large.";
        if (status === 429) return "Too many requests. Please wait a moment and try again.";
        if (status === 503) return "The server is still starting up. Please try again shortly.";
        if (status >= 500) return "The server ran into a problem (" + status + "). Please try again.";
        return "The request failed (" + status + ").";
    }

    /** fetch() + JSON parsing that throws a friendly Error for non-2xx replies. */
    async function fetchJson(url, options) {
        var response;
        try {
            response = await fetch(url, options);
        } catch (e) {
            throw new Error("Can't reach the server. Check that it is still running and try again.");
        }
        var body = null;
        try {
            body = await response.json();
        } catch (e) {
            /* non-JSON body */
        }
        if (!response.ok) throw new Error(describeError(response.status, body));
        return body;
    }

    function formatMs(value) {
        if (typeof value !== "number" || !isFinite(value)) return "--";
        return value < 10 ? value.toFixed(2) : String(Math.round(value));
    }

    /** Human-friendly message for getUserMedia / MediaRecorder failures. */
    function micErrorMessage(err) {
        if (!window.isSecureContext) {
            return "Microphone access needs HTTPS (or localhost). Open this page over a secure connection.";
        }
        switch (err && err.name) {
            case "NotAllowedError":
            case "SecurityError":
                return "Microphone access was blocked. Allow it from the address bar, then try again.";
            case "NotFoundError":
            case "OverconstrainedError":
                return "No microphone was found. Connect one and try again.";
            case "NotReadableError":
                return "The microphone is in use by another app.";
            default:
                return "Couldn't start recording" + (err && err.message ? ": " + err.message : ".");
        }
    }

    function pickMimeType() {
        if (!window.MediaRecorder || !MediaRecorder.isTypeSupported) return "";
        var candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
        for (var i = 0; i < candidates.length; i++) {
            if (MediaRecorder.isTypeSupported(candidates[i])) return candidates[i];
        }
        return "";
    }

    /**
     * Microphone recorder. The permission prompt only appears when start() is
     * called (never on page load). Recording stops on its own after maxMs.
     *   onStart()            recording began
     *   onStop(blob, mime)   recording finished (blob may be empty)
     *   onError(message)     friendly message for the user
     */
    function createRecorder(opts) {
        var maxMs = opts.maxMs || 30000;
        var recorder = null;
        var stream = null;
        var chunks = [];
        var timer = null;

        function cleanup() {
            clearTimeout(timer);
            if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
            stream = null;
        }

        async function start() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
                opts.onError(window.isSecureContext
                    ? "This browser can't record audio. Try a recent Chrome, Edge, Firefox or Safari."
                    : micErrorMessage(null));
                return false;
            }
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: { echoCancellation: true, noiseSuppression: true },
                });
            } catch (err) {
                opts.onError(micErrorMessage(err));
                return false;
            }
            var mime = pickMimeType();
            recorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
            chunks = [];
            recorder.ondataavailable = function (e) {
                if (e.data && e.data.size > 0) chunks.push(e.data);
            };
            recorder.onstop = function () {
                cleanup();
                var type = recorder.mimeType || mime || "audio/webm";
                opts.onStop(new Blob(chunks, { type: type }), type);
            };
            recorder.onerror = function (e) {
                cleanup();
                opts.onError("Recording failed" + (e.error && e.error.message ? ": " + e.error.message : "."));
            };
            recorder.start(250);
            timer = setTimeout(stop, maxMs);
            if (opts.onStart) opts.onStart();
            return true;
        }

        function stop() {
            clearTimeout(timer);
            if (recorder && recorder.state !== "inactive") recorder.stop();
        }

        return {
            start: start,
            stop: stop,
            isRecording: function () {
                return !!recorder && recorder.state === "recording";
            },
        };
    }

    /** Set the visual + accessible state of a `.mic__btn`: idle | recording | busy. */
    function setMicState(button, state, labels) {
        button.setAttribute("data-state", state);
        button.setAttribute("aria-pressed", state === "recording" ? "true" : "false");
        button.setAttribute("aria-label", labels[state]);
        if (state === "busy") button.setAttribute("aria-busy", "true");
        else button.removeAttribute("aria-busy");
    }

    if (darkQuery && darkQuery.addEventListener) darkQuery.addEventListener("change", syncThemeToggles);

    document.addEventListener("DOMContentLoaded", function () {
        document.querySelectorAll("[data-theme-toggle]").forEach(function (btn) {
            btn.addEventListener("click", toggleTheme);
        });
        syncThemeToggles();
    });

    window.ApolloUI = {
        storedTheme: storedTheme,
        toggleTheme: toggleTheme,
        singleSelect: singleSelect,
        fetchJson: fetchJson,
        describeError: describeError,
        formatMs: formatMs,
        micErrorMessage: micErrorMessage,
        createRecorder: createRecorder,
        setMicState: setMicState,
    };
})();

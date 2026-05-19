import { useState, useEffect, useRef } from "react"

// Detect touch-based small-screen devices once at module load.
// Touch + narrow viewport = phone/tablet. A touchscreen desktop still
// gets the desktop path because the viewport is wide.
const isMobile =
  typeof window !== "undefined" &&
  "ontouchstart" in window &&
  window.innerWidth < 1024

export function App() {
  const [messages, setMessages] = useState([])
  const [isThinking, setIsThinking] = useState(false)
  const [isWaitingForWakeWord, setIsWaitingForWakeWord] = useState(true)
  const [hasStarted, setHasStarted] = useState(false)

  // Mobile push-to-talk state. Desktop ignores all of this.
  const [mobileRecording, setMobileRecording] = useState(false)
  const [mobileRecordingSecs, setMobileRecordingSecs] = useState(0)
  const mobileRecordingRef = useRef(false)
  const mobileRecStartRef = useRef(0)
  const mobileTickRef = useRef(null)
  const mobileMaxTimerRef = useRef(null)

  const stateRef = useRef({
    isWaitingForWakeWord: true,
    isThinking: false,
    hasStarted: false,
  })

  const recognitionRef = useRef(null)
  const silenceTimerRef = useRef(null)
  const restartTimerRef = useRef(null)
  const audioRef = useRef(new Audio())
  const scrollBottomRef = useRef(null)

  // These refs are the important fix.
  // They prevent the browser STT from listening while Megan's TTS audio is playing.
  const shouldListenRef = useRef(false)
  const isSpeakingRef = useRef(false)
  const finalTranscriptBufferRef = useRef("")

  // MediaRecorder captures the actual microphone audio for Whisper.
  // Browser STT (Web Speech API) is now demoted to event-trigger only:
  // it listens for "megan" (wake) and "megan over" (turn-end) but its
  // transcribed text is no longer used as content. The audio captured
  // here is what Whisper transcribes on the backend.
  const mediaStreamRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])

  const WAKE_WORD = "megan"
  const EXIT_PHRASES = ["goodbye", "see you later", "bye", "stop listening"]

  // Turn-end is matched with regex (not literal substrings) because Chrome's
  // Web Speech STT often mistranscribes "Megan over" as "make an over",
  // "make and over", "again over", etc. — the same mishear family that gave
  // us "make an over" in the original test session. Accepting these variants
  // means the user rarely has to repeat themselves.
  const TURN_END_PATTERNS = [
    // Canonical phrases
    /\bmegan\s+(?:over|your\s+turn)\b/i,
    // Common STT mishears of "Megan"
    /\bmake\s+(?:an|and|in|it)\s+(?:over|your\s+turn)\b/i,
    /\b(?:maggie|maken|magna|making|again|may\s+again)\s+(?:over|your\s+turn)\b/i,
  ]

  const TURN_END_PHRASES = ["megan over", "megan your turn"]
  const VOICE_API = "/api/voice"

  const normalizeSpeechText = (text) =>
    text
      .toLowerCase()
      .replace(/[.,!?;:]/g, " ")
      .replace(/\s+/g, " ")
      .trim()

  const escapeRegExp = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")

  const findTurnEndPhrase = (text) => {
    const normalizedText = normalizeSpeechText(text)
    for (const pattern of TURN_END_PATTERNS) {
      const m = pattern.exec(normalizedText)
      if (m) return m[0]
    }
    return null
  }

  const stripTurnEndPhrase = (text, phrase) => {
    if (!phrase) return text.trim()

    const flexiblePhrase = phrase
      .split(" ")
      .map(escapeRegExp)
      .join("[\\s.,!?;:]+")

    return text
      .replace(new RegExp(flexiblePhrase, "i"), " ")
      .replace(/\s+/g, " ")
      .trim()
  }

  useEffect(() => {
    scrollBottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isThinking])

  useEffect(() => {
    stateRef.current = { isWaitingForWakeWord, isThinking, hasStarted }
  }, [isWaitingForWakeWord, isThinking, hasStarted])

  // Signal the backend that this is a fresh session. Resets in-session
  // counters used by Megan's encounter_length() bucketing so warmth scales
  // correctly — e.g. so a 3-turn morning hello doesn't get a "great
  // chatting today" farewell. Long-term memory is untouched. Fires once
  // on mount; the 30-min idle gap on the backend is the belt-and-braces
  // backstop.
  useEffect(() => {
    fetch("/api/session/new", { method: "POST" }).catch(() => {
      // Non-fatal: if this fails, the idle-gap backstop still handles
      // most cases, and an over-warm farewell on a stale conversation
      // is a soft failure, not a broken app.
    })
  }, [])

  useEffect(() => {
    return () => {
      clearTimer(silenceTimerRef)
      clearTimer(restartTimerRef)
      try {
        recognitionRef.current?.stop()
      } catch (e) {}
      try {
        audioRef.current?.pause()
      } catch (e) {}
      try {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
          mediaRecorderRef.current.stop()
        }
      } catch (e) {}
      try {
        mediaStreamRef.current?.getTracks().forEach((t) => t.stop())
      } catch (e) {}
    }
  }, [])

  const clearTimer = (timerRef) => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }

  const safeStartRecognition = () => {
    const recognition = recognitionRef.current
    if (!recognition) return
    if (!shouldListenRef.current) return
    if (isSpeakingRef.current) return
    if (stateRef.current.isThinking) return

    try {
      recognition.start()
    } catch (e) {
      // Chrome throws if recognition is already running. This is safe to ignore.
    }
  }

  const safeStopRecognition = () => {
    const recognition = recognitionRef.current
    if (!recognition) return

    try {
      recognition.stop()
    } catch (e) {
      // Chrome throws if recognition is already stopped. This is safe to ignore.
    }
  }

  const pauseListening = () => {
    shouldListenRef.current = false
    clearTimer(silenceTimerRef)
    clearTimer(restartTimerRef)
    finalTranscriptBufferRef.current = ""
    safeStopRecognition()
  }

  const resumeListening = ({ force = false } = {}) => {
    if (!stateRef.current.hasStarted) return
    if (isSpeakingRef.current) return

    // React state updates are asynchronous. When Megan's audio finishes,
    // setIsThinking(false) may not have reached stateRef yet.
    // force=true is used only after we manually mark thinking as false.
    if (!force && stateRef.current.isThinking) return

    shouldListenRef.current = true
    clearTimer(restartTimerRef)
    restartTimerRef.current = setTimeout(() => {
      safeStartRecognition()
    }, 250)
  }

  // ---- MediaRecorder helpers (audio capture for Whisper) -------------------

  const initMediaRecorder = async () => {
    if (mediaStreamRef.current) return true

    try {
      mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      return true
    } catch (err) {
      console.error("Microphone permission denied:", err)
      addMsg(
        `Mic error: ${err.name || "Unknown"} — ${err.message || "(no message)"}`,
        "ai",
      )
      return false
    }
  }

  const pickRecorderMimeType = () => {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/ogg",
      "audio/mp4",
    ]
    for (const t of candidates) {
      if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(t)) {
        return t
      }
    }
    return ""
  }

  const startMediaRecorder = () => {
    if (!mediaStreamRef.current) return
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      return
    }

    audioChunksRef.current = []
    const mimeType = pickRecorderMimeType()
    const recorder = mimeType
      ? new MediaRecorder(mediaStreamRef.current, { mimeType })
      : new MediaRecorder(mediaStreamRef.current)

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) audioChunksRef.current.push(e.data)
    }

    // 250ms timeslice keeps chunks flowing so an early stop still has data.
    recorder.start(250)
    mediaRecorderRef.current = recorder
  }

  const stopMediaRecorderAndGetBlob = () =>
    new Promise((resolve) => {
      const recorder = mediaRecorderRef.current
      if (!recorder || recorder.state === "inactive") {
        resolve(null)
        return
      }
      recorder.onstop = () => {
        const mimeType = recorder.mimeType || "audio/webm"
        const blob = new Blob(audioChunksRef.current, { type: mimeType })
        audioChunksRef.current = []
        resolve(blob)
      }
      try {
        recorder.stop()
      } catch (e) {
        resolve(null)
      }
    })

  // ---- Engine startup --------------------------------------------------------

  const startEngine = async () => {
    if (hasStarted) return

    // Update the ref immediately because React state updates are asynchronous.
    // v2.2.3: also flip isWaitingForWakeWord OFF — we no longer require the
    // user to say "Hey Megan" to activate a session. The greeting itself is
    // the activation; after it finishes the recogniser jumps straight into
    // turn-end listening mode.
    stateRef.current = {
      ...stateRef.current,
      hasStarted: true,
      isWaitingForWakeWord: false,
    }

    setHasStarted(true)
    setIsWaitingForWakeWord(false)

    // Gate listening OFF while the greeting plays. resumeListening() called
    // from finishGreeting() will flip this back on once Megan stops talking.
    shouldListenRef.current = false

    // Acquire mic permission. We DON'T start MediaRecorder here anymore —
    // capturing audio while Megan's greeting is playing through the speakers
    // would feed her own voice back into the next turn's transcript. Recorder
    // starts inside finishGreeting() once the greeting finishes.
    const ok = await initMediaRecorder()
    if (!ok) return

    // Web Speech API setup (desktop only). initRecognition() ends with a
    // safeStartRecognition() call, but that bails immediately because
    // shouldListenRef is false — so recognition is initialised but dormant
    // until finishGreeting() wakes it up.
    if (!isMobile) initRecognition()

    // Fetch and play Megan's greeting. Awaits play start (not playback end);
    // the audio element's onended callback handles the post-greeting transition.
    await playGreeting()
  }

  // v2.2.3: GET /api/greeting, display the text, play the audio. Used by
  // startEngine to initiate every fresh session. The backend already does
  // the LLM call + Piper synth and returns text + audio_url; we just play it.
  const playGreeting = async () => {
    setIsThinking(true)
    isSpeakingRef.current = true

    try {
      const res = await fetch("/api/greeting")
      if (!res.ok) throw new Error(`Greeting HTTP ${res.status}`)
      const data = await res.json()

      if (data.response) {
        addMsg(data.response, "ai")
      }

      if (!data.audio_url) {
        // Backend returned text but no audio (fallback path in /api/greeting).
        // Transition to listening anyway so the conversation can continue.
        finishGreeting()
        return
      }

      audioRef.current.pause()
      audioRef.current.currentTime = 0
      audioRef.current.src = data.audio_url
      audioRef.current.onended = finishGreeting
      audioRef.current.onerror = () => {
        console.warn("Greeting audio failed to play.")
        finishGreeting()
      }

      await audioRef.current.play()
    } catch (err) {
      console.warn("Greeting fetch/play failed:", err)
      finishGreeting()
    }
  }

  // Shared transition fired when the greeting audio finishes (or fails).
  // Mirrors the post-reply transition in handleTurnEnd: drops the speaking
  // flag, clears thinking, and opens the listening window for the user's
  // first real turn. On mobile, just clears the speaking flag — the user
  // taps when ready, no continuous listening.
  const finishGreeting = () => {
    isSpeakingRef.current = false
    stateRef.current = { ...stateRef.current, isThinking: false }
    setIsThinking(false)
    if (!isMobile) {
      startMediaRecorder()
      resumeListening({ force: true })
    }
  }

  const initRecognition = () => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition

    if (!SpeechRecognition) {
      addMsg(
        "Speech recognition is not supported in this browser. Please use Chrome or Edge.",
        "ai",
      )
      return
    }

    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = "en-AU"

    recognition.onresult = (event) => {
      const { isThinking: thinking, isWaitingForWakeWord: waiting } =
        stateRef.current

      // Critical guard: do not process anything while Megan is speaking/thinking.
      if (thinking || isSpeakingRef.current || !shouldListenRef.current) return

      let interimTranscript = ""
      let finalTranscript = ""

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const transcriptPiece = event.results[i][0].transcript

        if (event.results[i].isFinal) {
          finalTranscript += transcriptPiece
        } else {
          interimTranscript += transcriptPiece
        }
      }

      const heardText = `${finalTranscript} ${interimTranscript}`.trim()
      const lowerHeardText = normalizeSpeechText(heardText)

      if (!heardText) return

      // Wake-word branch.
      // CHANGED: previously this returned early after detecting "megan", which
      // discarded any content spoken in the same breath ("Hey Megan, what's
      // the weather, Megan over"). Now we transition state and FALL THROUGH so
      // the turn-end phrase in the same utterance is still processed.
      // We clear the buffer first so any pre-wake noise is discarded.
      if (waiting) {
        if (lowerHeardText.includes(WAKE_WORD)) {
          stateRef.current = {
            ...stateRef.current,
            isWaitingForWakeWord: false,
          }
          setIsWaitingForWakeWord(false)
          finalTranscriptBufferRef.current = ""
          clearTimer(silenceTimerRef)
          // intentionally NOT returning — fall through to turn-end check.
        } else {
          return
        }
      }

      const wantsToExit = EXIT_PHRASES.some((phrase) =>
        lowerHeardText.includes(phrase),
      )

      if (wantsToExit) {
        console.log("Exiting and clearing chat...")
        setMessages([])
        setIsWaitingForWakeWord(true)
        finalTranscriptBufferRef.current = ""
        clearTimer(silenceTimerRef)
        return
      }

      // Browser STT text is no longer used as message content. We only watch
      // the combined buffer + interim transcript for the turn-end phrase
      // ("Megan over" / "Megan your turn" or common mishears). When it fires,
      // the actual content is grabbed from the MediaRecorder audio blob and
      // sent to /api/voice for Whisper.
      //
      // CHANGED: previously the turn-end check only ran when a NEW final
      // transcript arrived, which meant we waited for Chrome to mark a segment
      // as final — sometimes 500-1500ms after the user stopped speaking. Now
      // we also check the live interim transcript so we fire as soon as the
      // words appear. Combined with fuzzy regex matching above, the user
      // rarely has to repeat "Megan over".
      if (finalTranscript.trim().length > 0) {
        finalTranscriptBufferRef.current = `${finalTranscriptBufferRef.current} ${finalTranscript}`.trim()
        clearTimer(silenceTimerRef)
      }

      const combinedForTurnEnd = [
        finalTranscriptBufferRef.current,
        interimTranscript,
      ].filter(Boolean).join(" ").trim()

      if (combinedForTurnEnd) {
        const turnEndPhrase = findTurnEndPhrase(combinedForTurnEnd)
        if (turnEndPhrase) {
          finalTranscriptBufferRef.current = ""
          handleTurnEnd()
        }
      }
    }

    recognition.onerror = (event) => {
      console.warn("Speech recognition error:", event.error)

      // Benign errors — silence, mic glitches, browser quirks. The recognizer
      // recovers on its own. Don't pollute the chat window with these; Megan
      // should stand by silently when Teddy isn't talking.
      // Reference: https://wicg.github.io/speech-api/#dom-speechrecognitionerrorcode
      const BENIGN_ERRORS = new Set([
        "no-speech",       // silence in the room — by far the most common
        "aborted",         // recognizer manually stopped or restarted
        "audio-capture",   // transient mic glitch; recovers next cycle
        "network",         // intermittent connectivity; usually self-heals
      ])

      if (!BENIGN_ERRORS.has(event.error)) {
        addMsg(`SpeechRecognition error: ${event.error}`, "ai")
      }

      // no-speech is common and not fatal. Other errors should not crash the app either.
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        shouldListenRef.current = false
      }
    }

    recognition.onend = () => {
      // Critical guard: do not auto-restart while Megan is speaking, thinking, or deliberately paused.
      if (
        stateRef.current.hasStarted &&
        shouldListenRef.current &&
        !isSpeakingRef.current &&
        !stateRef.current.isThinking
      ) {
        clearTimer(restartTimerRef)
        restartTimerRef.current = setTimeout(() => {
          safeStartRecognition()
        }, 250)
      }
    }

    recognitionRef.current = recognition
    safeStartRecognition()
  }

  const handleTurnEnd = async () => {
    clearTimer(silenceTimerRef)

    // Stop listening as soon as the turn-end phrase fires.
    // This prevents the microphone from collecting noise while Megan is
    // preparing/speaking, and stops the recorder so we can grab the blob.
    pauseListening()
    setIsThinking(true)

    const blob = await stopMediaRecorderAndGetBlob()

    // If the recorder produced nothing (e.g. permission gone, race), recover
    // gracefully and re-arm for the next turn.
    if (!blob || blob.size === 0) {
      console.warn("No audio captured for this turn.")
      addMsg("(no audio captured — please try again)", "ai")
      stateRef.current = { ...stateRef.current, isThinking: false }
      setIsThinking(false)
      if (!isMobile) {
        startMediaRecorder()
        resumeListening({ force: true })
      }
      return
    }

    try {
      const formData = new FormData()
      // Filename is informational; backend sniffs format from content-type.
      const ext = (blob.type || "audio/webm").includes("ogg") ? "ogg" : "webm"
      formData.append("audio", blob, `turn.${ext}`)

      const res = await fetch(VOICE_API, {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        throw new Error(`Backend returned HTTP ${res.status}`)
      }

      const data = await res.json()

      // Show the user bubble with what Whisper actually heard.
      // (Backend sends a "transcript" field; fall back to a placeholder.)
      const userText = (data.transcript || "").trim()
      if (userText) {
        addMsg(userText, "user")
      } else {
        addMsg("(silence)", "user")
      }
      addMsg(data.response, "ai")

      if (data.audio_url) {
        isSpeakingRef.current = true
        shouldListenRef.current = false

        audioRef.current.pause()
        audioRef.current.currentTime = 0
        audioRef.current.src = data.audio_url

        audioRef.current.onended = () => {
          isSpeakingRef.current = false
          stateRef.current = {
            ...stateRef.current,
            isThinking: false,
          }
          setIsThinking(false)
          // Start a fresh recording window for the next turn.
          if (!isMobile) {
            startMediaRecorder()
            resumeListening({ force: true })
          }
        }

        audioRef.current.onerror = () => {
          console.warn("TTS audio failed to play.")
          isSpeakingRef.current = false
          stateRef.current = {
            ...stateRef.current,
            isThinking: false,
          }
          setIsThinking(false)
          if (!isMobile) {
            startMediaRecorder()
            resumeListening({ force: true })
          }
        }

        await audioRef.current.play()
      } else {
        stateRef.current = {
          ...stateRef.current,
          isThinking: false,
        }
        setIsThinking(false)
        if (!isMobile) {
          startMediaRecorder()
          resumeListening({ force: true })
        }
      }
    } catch (err) {
      console.error("Voice request failed:", err)
      isSpeakingRef.current = false
      addMsg("Connection error.", "ai")
      stateRef.current = {
        ...stateRef.current,
        isThinking: false,
        isWaitingForWakeWord: true,
      }
      setIsThinking(false)
      setIsWaitingForWakeWord(true)
      if (!isMobile) {
        startMediaRecorder()
        resumeListening({ force: true })
      }
    }
  }

  // ---- Mobile push-to-talk -------------------------------------------------
  // Tap once to activate Megan (plays greeting, no recording yet).
  // Tap again to start recording, and once more to send the turn.
  // Replaces the wake-phrase listener on phones. Desktop never calls this.
  const handleMobileMicTap = async () => {
    // Defensive: ignore taps while Megan is thinking or speaking.
    if (stateRef.current.isThinking || isSpeakingRef.current) return

    // First tap on a fresh session: activate and play greeting.
    // The user taps again AFTER the greeting finishes to start recording.
    // The disabled guard above ensures we don't start recording on a stray
    // tap mid-greeting.
    if (!stateRef.current.hasStarted) {
      await startEngine()
      return
    }

    if (!mobileRecordingRef.current) {
      // ---- START recording ----
      if (!mediaStreamRef.current) {
        const ok = await initMediaRecorder()
        if (!ok) return
      }
      startMediaRecorder()
      mobileRecordingRef.current = true
      mobileRecStartRef.current = Date.now()
      setMobileRecording(true)
      setMobileRecordingSecs(0)

      // Tick the visible timer every second.
      mobileTickRef.current = setInterval(() => {
        const elapsed = Math.floor(
          (Date.now() - mobileRecStartRef.current) / 1000,
        )
        setMobileRecordingSecs(elapsed)
      }, 1000)

      // Safety ceiling: auto-send after 30 minutes so a forgotten
      // recording can't fill memory or run forever.
      mobileMaxTimerRef.current = setTimeout(
        () => {
          if (mobileRecordingRef.current) {
            handleMobileMicTap()
          }
        },
        30 * 60 * 1000,
      )
    } else {
      // ---- STOP recording and send ----
      mobileRecordingRef.current = false
      setMobileRecording(false)
      clearTimer(mobileTickRef)
      clearTimer(mobileMaxTimerRef)
      setMobileRecordingSecs(0)

      // Reuse the existing turn-end flow. handleTurnEnd will stop the
      // recorder, upload the blob to /api/voice, show transcript + reply,
      // and play Megan's audio. After playback finishes, mobile stays
      // idle (waiting for the next tap) because we gated the auto-restart.
      handleTurnEnd()
    }
  }

  const addMsg = (text, role) => {
    setMessages((prev) => [...prev, { text, role, id: crypto.randomUUID() }])
  }

  // Export the current conversation as a Markdown file the user can save.
  // Format: human-readable header with timestamp, then alternating speakers.
  // Markdown is the friendliest format — opens in any text editor, renders
  // nicely in editors that understand it, and survives copy-paste cleanly.
  const exportConversation = () => {
    if (messages.length === 0) return

    const now = new Date()
    const stamp = now.toLocaleString("en-AU", {
      dateStyle: "full",
      timeStyle: "short",
    })
    const fileStamp = now
      .toISOString()
      .slice(0, 16)
      .replace(/[:T]/g, "-")

    const header = `# Conversation with Megan\n\n_Exported: ${stamp}_\n\n---\n\n`
    const body = messages
      .map((m) => {
        const speaker = m.role === "user" ? "**Teddy**" : "**Megan**"
        return `${speaker}: ${m.text}`
      })
      .join("\n\n")

    const markdown = header + body + "\n"
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `megan-conversation-${fileStamp}.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div style={s.container}>
      <div
        style={{
          ...s.header,
          background: hasStarted ? "white" : "#6366f1",
          color: hasStarted ? "#6366f1" : "white",
        }}
        onClick={startEngine}
      >
        {hasStarted
          ? isWaitingForWakeWord
            ? "MEGAN • STANDBY"
            : "MEGAN • ACTIVE"
          : "TAP TO ACTIVATE MEGAN"}
      </div>

      <div style={s.chatWindow}>
        {messages.length > 0 && (
          <button
            onClick={exportConversation}
            style={{
              position: "absolute",
              top: "8px",
              right: "12px",
              background: "rgba(255, 255, 255, 0.9)",
              border: "1px solid #e2e8f0",
              borderRadius: "8px",
              padding: "4px 10px",
              fontSize: "12px",
              color: "#64748b",
              cursor: "pointer",
              zIndex: 10,
              fontFamily: "inherit",
            }}
            title="Export conversation as Markdown"
          >
            📥 Export
          </button>
        )}
        {messages.map((m) => (
          <div
            key={m.id}
            style={{
              ...s.msg,
              alignSelf: m.role === "user" ? "flex-end" : "flex-start",
            }}
          >
            <div
              style={{
                ...s.bubble,
                background: m.role === "user" ? "#6366f1" : "#f8fafc",
                color: m.role === "user" ? "white" : "#1e293b",
                textAlign: m.role === "ai" ? "left" : "right",
                border: m.role === "ai" ? "1px solid #e2e8f0" : "none",
                borderRadius: "20px",
                borderBottomRightRadius: m.role === "user" ? "4px" : "20px",
                borderBottomLeftRadius: m.role === "ai" ? "4px" : "20px",
              }}
            >
              {m.text}
            </div>
          </div>
        ))}
        {isThinking && <div style={s.thinking}>Thinking...</div>}
        <div ref={scrollBottomRef} />
      </div>

      <div style={s.footer}>
        {isMobile ? (
          <>
            <button
              onClick={handleMobileMicTap}
              disabled={isThinking || isSpeakingRef.current}
              style={{
                ...s.micButton,
                background: mobileRecording ? "#ef4444" : "#6366f1",
                opacity: isThinking || isSpeakingRef.current ? 0.4 : 1,
                cursor:
                  isThinking || isSpeakingRef.current ? "default" : "pointer",
              }}
            >
              {mobileRecording
                ? `● Recording  ${formatMobileTime(mobileRecordingSecs)}  ·  Tap to send`
                : !hasStarted
                  ? "🎤  Tap to start"
                  : isThinking || isSpeakingRef.current
                    ? "Megan is speaking..."
                    : "🎤  Tap to talk"}
            </button>
          </>
        ) : (
          <>
            <div
              style={{
                ...s.indicator,
                background: !hasStarted
                  ? "#cbd5e1"
                  : isThinking || isSpeakingRef.current
                    ? "#f59e0b"
                    : isWaitingForWakeWord
                      ? "#94a3b8"
                      : "#22c55e",
              }}
            ></div>
            <div style={s.statusText}>
              {!hasStarted
                ? "Tap bar to start"
                : isThinking || isSpeakingRef.current
                  ? "Megan is speaking / processing..."
                  : isWaitingForWakeWord
                    ? 'Say "Hey Megan" to resume'
                    : 'Say "Megan, over" when done'}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// Format seconds as MM:SS for the mobile recording timer.
function formatMobileTime(secs) {
  const m = Math.floor(secs / 60)
  const s = secs % 60
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
}

const s = {
  container: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    background: "white",
    fontFamily: "-apple-system, sans-serif",
  },
  header: {
    padding: "20px",
    textAlign: "center",
    fontWeight: "800",
    fontSize: "12px",
    letterSpacing: "2px",
    borderBottom: "1px solid #f1f5f9",
    cursor: "pointer",
    transition: "0.2s",
  },
  chatWindow: {
    flex: 1,
    overflowY: "auto",
    padding: "20px",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  msg: { maxWidth: "75%", width: "fit-content" },
  bubble: { padding: "12px 18px", fontSize: "15px", lineHeight: "1.4" },
  thinking: {
    color: "#6366f1",
    fontSize: "12px",
    margin: "10px",
    fontWeight: "600",
  },
  footer: {
    padding: "25px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "8px",
    borderTop: "1px solid #f1f5f9",
  },
  statusText: { fontSize: "13px", fontWeight: "600", color: "#64748b" },
  indicator: {
    width: "10px",
    height: "10px",
    borderRadius: "50%",
    transition: "0.3s",
  },
  micButton: {
    width: "100%",
    minHeight: "72px",
    border: "none",
    borderRadius: "16px",
    color: "white",
    fontSize: "16px",
    fontWeight: "700",
    letterSpacing: "0.5px",
    transition: "background 0.2s, opacity 0.2s",
    userSelect: "none",
    WebkitUserSelect: "none",
    WebkitTapHighlightColor: "transparent",
  },
}

export default App

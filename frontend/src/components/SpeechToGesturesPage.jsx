import React, { useState, useRef, useEffect } from "react";
import { io } from "socket.io-client";
import { Link } from "react-router-dom";

const socket = io("/");

function SpeechToGesturesPage() {
  const [Result, setResult] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const mediaStream = useRef(null);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const [currentFrame, setCurrentFrame] = useState();

  const canvasRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationIdRef = useRef(null);
  const visualizerSource = useRef(null);

  useEffect(() => {
    socket.on("recognized_text", (data) => {
      setResult(data.text);
    });

    socket.on("gesture_sequence", async (framesData) => {
      for (let frame of framesData) {
        const mimeType = frame.type === "gif" ? "image/gif" : "image/jpeg";
        setCurrentFrame(`data:${mimeType};base64,${frame.data}`);
        await new Promise((resolve) => setTimeout(resolve, frame.delay));
      }
    });

    // Draw initial flat idle visualizer bars
    drawStaticVisualizer();

    // cleanup on unmount
    return () => {
      socket.off("recognized_text");
      socket.off("gesture_sequence");
      stopVisualizer();
    };
  }, []);

  const drawStaticVisualizer = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#e0e0e0";
      const barWidth = canvas.width / 32;
      let x = 0;
      for (let i = 0; i < 32; i++) {
        const barHeight = 6;
        const y = (canvas.height - barHeight) / 2;
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(x, y, barWidth - 4, barHeight, 3);
        } else {
          ctx.rect(x, y, barWidth - 4, barHeight);
        }
        ctx.fill();
        x += barWidth;
      }
    }
  };

  const startVisualizer = (stream) => {
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioContextClass();
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64; // Small FFT size for fewer, thicker bars

      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      audioContextRef.current = audioCtx;
      analyserRef.current = analyser;
      visualizerSource.current = source;

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");

      const draw = () => {
        if (!analyserRef.current) return;
        animationIdRef.current = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const barWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const value = dataArray[i] / 255.0; // 0 to 1
          // Scale bar height to look extremely active and lively
          const barHeight = value * canvas.height * 0.9 + 6;

          // Center the bars vertically
          const y = (canvas.height - barHeight) / 2;

          // Vibrant Gradient
          const gradient = ctx.createLinearGradient(0, y, 0, y + barHeight);
          gradient.addColorStop(0, "#28a745"); // vibrant green top
          gradient.addColorStop(0.5, "#007bff"); // modern blue middle
          gradient.addColorStop(1, "#00d2ff"); // glowing cyan bottom

          ctx.fillStyle = gradient;

          ctx.beginPath();
          if (ctx.roundRect) {
            ctx.roundRect(x, y, barWidth - 4, barHeight, 4);
          } else {
            ctx.rect(x, y, barWidth - 4, barHeight);
          }
          ctx.fill();

          x += barWidth;
        }
      };

      draw();
    } catch (e) {
      console.error("Failed to start audio visualizer:", e);
    }
  };

  const stopVisualizer = () => {
    if (animationIdRef.current) {
      cancelAnimationFrame(animationIdRef.current);
      animationIdRef.current = null;
    }
    if (visualizerSource.current) {
      visualizerSource.current.disconnect();
      visualizerSource.current = null;
    }
    if (audioContextRef.current) {
      if (audioContextRef.current.state !== "closed") {
        audioContextRef.current.close();
      }
      audioContextRef.current = null;
    }
    analyserRef.current = null;

    // Draw flat static visualizer when idle
    drawStaticVisualizer();
  };

  const send_audio = (audioBlob) => {
    audioBlob.arrayBuffer().then((buffer) => {
      socket.emit("audio_utterence", { buffer: buffer, type: audioBlob.type });
    });
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStream.current = stream;
      mediaRecorder.current = new MediaRecorder(stream, { mimeType: "audio/webm" });
      audioChunks.current = [];

      mediaRecorder.current.start();
      setIsRecording(true);
      startVisualizer(stream);

      mediaRecorder.current.ondataavailable = (event) => {
        audioChunks.current.push(event.data);
      };

      mediaRecorder.current.onstop = () => {
        const audioBlob = new Blob(audioChunks.current, { type: "audio/webm" });
        send_audio(audioBlob);

        stream.getTracks().forEach((track) => track.stop());
      };
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      setIsRecording(false);
      stopVisualizer();
    }
  };

  return (
    <div
      style={{
        maxWidth: "900px",
        margin: "0 auto",
        padding: "25px",
        background: "#fff",
        borderRadius: "12px",
        boxShadow: "0px 4px 20px rgba(0,0,0,0.08)",
        fontFamily: "Arial, sans-serif",
      }}
    >
      {/* Pulse Dot Animation style injection */}
      <style>{`
        @keyframes pulse {
          0% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.3); opacity: 0.7; }
          100% { transform: scale(1); opacity: 1; }
        }
        .pulse-dot {
          animation: pulse 1.5s infinite ease-in-out;
        }
      `}</style>

      {/* Page Title */}
      <h2 style={{ textAlign: "center", marginBottom: "15px" }}>
        🎤 Speech-to-Gestures
      </h2>

      {/* Status Indicator */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: isRecording ? "#e8f5e9" : "#f5f5f5",
          padding: "8px 15px",
          borderRadius: "6px",
          fontWeight: "500",
          marginBottom: "20px",
          transition: "all 0.3s ease",
        }}
      >
        <span
          className={isRecording ? "pulse-dot" : ""}
          style={{
            width: "10px",
            height: "10px",
            borderRadius: "50%",
            background: isRecording ? "#4caf50" : "#757575",
            marginRight: "8px",
            transition: "all 0.3s ease",
          }}
        ></span>
        Status: {isRecording ? "Listening" : "Inactive"}
      </div>

      {/* Audio Visualizer Modulation Bar */}
      <div
        style={{
          background: "#fafafa",
          border: "1px solid #e0e0e0",
          borderRadius: "8px",
          padding: "15px",
          marginBottom: "20px",
          textAlign: "center",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span style={{ fontSize: "0.9rem", color: "#666", marginBottom: "8px", fontWeight: "500" }}>
          {isRecording ? "🎙 Live Voice Modulation Bar" : "🎙 Microphone Inactive"}
        </span>
        <canvas
          ref={canvasRef}
          width="400"
          height="60"
          style={{
            width: "100%",
            maxWidth: "400px",
            height: "60px",
            background: "#f0f2f5",
            borderRadius: "6px",
          }}
        />
      </div>

      {/* Main Content Area */}
      <div
        style={{
          display: "flex",
          gap: "20px",
          flexWrap: "wrap",
          justifyContent: "center",
        }}
      >
        {/* Recognized Text */}
        <div
          style={{
            flex: "1 1 300px",
            background: "#f9f9f9",
            padding: "15px",
            borderRadius: "10px",
          }}
        >
          <h3>📝 Recognized Text</h3>
          <p
            style={{
              fontSize: "1.1rem",
              minHeight: "50px",
              padding: "10px",
              background: "#fff",
              borderRadius: "8px",
              boxShadow: "inset 0px 1px 3px rgba(0,0,0,0.1)",
            }}
          >
            {Result ? Result : "Speak Something"}
          </p>
        </div>

        {/* Gesture Display */}
        <div
          style={{
            flex: "1 1 300px",
            background: "#f9f9f9",
            padding: "15px",
            borderRadius: "10px",
            textAlign: "center",
          }}
        >
          <h3>✋ Gestures</h3>
          <div
            style={{
              background: "#fff",
              borderRadius: "8px",
              padding: "10px",
              minHeight: "150px",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              boxShadow: "inset 0px 1px 3px rgba(0,0,0,0.1)",
            }}
          >
            <span style={{ color: "#bbb" }}>
              {currentFrame ? (
                <img src={currentFrame} alt="Gesture" width="300" />
              ) : (
                "Waiting for speech..."
              )}
            </span>
          </div>
        </div>
      </div>

      {/* Control Buttons */}
      <div
        style={{
          marginTop: "25px",
          display: "flex",
          justifyContent: "center",
          gap: "15px",
          flexWrap: "wrap",
        }}
      >
        <button
          style={{
            padding: "10px 18px",
            background: "#28a745",
            color: "#fff",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            fontWeight: "600",
            opacity: isRecording ? 0.7 : 1,
            pointerEvents: isRecording ? "none" : "auto",
          }}
          onClick={() => startRecording()}
        >
          ▶ Start Listening
        </button>
        <button
          style={{
            padding: "10px 18px",
            background: "#dc3545",
            color: "#fff",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            fontWeight: "600",
            opacity: isRecording ? 1 : 0.7,
            pointerEvents: isRecording ? "auto" : "none",
          }}
          onClick={() => stopRecording()}
        >
          ⏹ Stop Listening
        </button>
        <Link to="/">
          <button
            style={{
              padding: "10px 18px",
              background: "#6c757d",
              color: "#fff",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              fontWeight: "600",
            }}
          >
            ⬅ Back
          </button>
        </Link>
      </div>
    </div>
  );
}

export default SpeechToGesturesPage;

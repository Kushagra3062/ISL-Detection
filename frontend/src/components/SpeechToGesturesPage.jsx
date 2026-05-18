import React, { useState, useRef,useEffect } from "react";
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
  
    // cleanup on unmount
    return () => {
      socket.off("recognized_text");
      socket.off("gesture_sequence");
    };
  });

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

      mediaRecorder.current.ondataavailable = (event) => {
        audioChunks.current.push(event.data);
      };

      mediaRecorder.current.onstop = () => {
        const audioBlob = new Blob(audioChunks.current, { type: "audio/webm" }); // Adjust type if needed
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
          background: "#e8f5e9",
          padding: "8px 15px",
          borderRadius: "6px",
          fontWeight: "500",
          marginBottom: "20px",
        }}
      >
        <span
          style={{
            width: "10px",
            height: "10px",
            borderRadius: "50%",
            background: "#4caf50",
            marginRight: "8px",
          }}
        ></span>
        Status: Listening
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
            {Result?Result:"Speak Something"}
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
            {/* Replace with gesture GIF/image */}
            <span style={{ color: "#bbb" }}>
            {currentFrame && <img src={currentFrame} alt="Gesture" width="300" />}
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
          }}
          onClick={() => stopRecording()}
        >
          ⏹ Stop Listening
        </button>
        <Link to='/'>
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

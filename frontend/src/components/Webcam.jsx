import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import "./Webcam.css";

function Webcamera() {
  const webcamRef = useRef(null);
  const [Word, setWord] = useState("");
  const [Pred, setPred] = useState("");
  const [Con, setCon] = useState();
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [Det, setDet] = useState(false);

  const sendframe = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      try {
        const response = await fetch("/api/frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: imageSrc }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        } else {
          const data = await response.json();
          setWord(data.word);
          setPred(data.prediction);
          setCon(data.confidence);
        }
      } catch (err) {
        console.error("Error fetching data:", err);
      }
    }
  };

  useEffect(() => {
    let interval;
    if (Det) {
      interval = setInterval(sendframe, 100);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [Det]);

  return (
    <div className="camera-card">
      <h2 className="main-title">ISL Detection</h2>

      <div className={`status-bar ${Det ? "active" : "inactive"}`}>
        <span className="status-dot"></span>
        Status: {Det ? "Active" : "Inactive"}
      </div>

      <div className="webcam-container">
        <Webcam
          audio={false}
          mirrored={true}
          ref={webcamRef}
          onUserMedia={() => setIsCameraActive(true)}
          screenshotFormat="image/jpeg"
          className="webcam-feed"
        />
      </div>

      {/* Start / Stop Detection Button */}
      <button
        className={`detection-btn ${Det ? "stop" : "start"}`}
        onClick={() => setDet((prev) => !prev)}
      >
        {Det ? "⏹ Stop Detection" : "▶ Start Detection"}
      </button>

      <div className="prediction-box word-box">
        <div className="prediction-title">📝 Predicted Word</div>
        <div className="prediction-value">
          {Word || ""}
          <span className="cursor">|</span>
        </div>
      </div>

      <div className="prediction-box letter-box">
        <div className="prediction-title">✋ Predicted Letter</div>
        <div className="prediction-value letter-value">{Pred || "--"}</div>
      </div>

      <div className="prediction-box confidence-box">
        <div className="prediction-title">📊 Confidence Score</div>
        <div className="prediction-value">
          {Con ? `${(Con * 100).toFixed(2)}%` : "--"}
        </div>
        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{ width: `${Con * 100 || 0}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
}

export default Webcamera;

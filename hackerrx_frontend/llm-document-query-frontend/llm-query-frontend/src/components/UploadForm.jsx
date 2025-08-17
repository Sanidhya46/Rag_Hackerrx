/**
 * UploadForm.jsx
 *
 * Upload a PDF to the FastAPI backend and store returned document_id.
 */
import React, { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";
import Spinner from "./Spinner";
import { UploadCloud, FileCheck } from "lucide-react";

function UploadForm() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("Drag & drop or click");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const onDrop = useCallback((accepted) => {
    if (accepted.length) {
      setFile(accepted[0]);
      setStatus(`Selected: ${accepted[0].name}`);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    multiple: false,
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setStatus("Please select a file first.");
      return;
    }

    setLoading(true);
    setStatus("Uploading and processing…");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      console.log("[UPLOAD] HTTP status:", response.status);
      const data = await response.json().catch(() => ({}));
      console.log("[UPLOAD] JSON payload:", data);

      if (response.ok && data.document_id) {
        localStorage.setItem("document_id", data.document_id);
        setStatus("Upload successful, redirecting…");
        navigate("/chat");
      } else {
        setStatus(data.detail || data.message || "Upload failed.");
      }
    } catch (err) {
      console.error("[UPLOAD] Network error:", err);
      setStatus("Server error. Try again later.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-form-wrapper">
      {loading && <Spinner />}
      <form onSubmit={handleSubmit} className="upload-form">
        <div {...getRootProps({ className: `dropzone ${isDragActive ? "active" : ""}` })}>
          <input {...getInputProps()} />
          <div className="dropzone-content">
            {file ? <FileCheck size={48} /> : <UploadCloud size={48} />}
            <p>{status}</p>
          </div>
        </div>
        <button type="submit" disabled={!file || loading}>
          {loading ? "Processing…" : "Start Querying"}
        </button>
      </form>
    </div>
  );
}

export default UploadForm;
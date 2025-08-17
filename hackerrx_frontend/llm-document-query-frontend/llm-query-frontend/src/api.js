const BACKEND = "http://localhost:8000";          // FastAPI base URL

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BACKEND}/upload/`, { method: "POST", body: formData });
  if (!res.ok) throw new Error("Upload failed");
  return res.json();  // { doc_id, filename, ... }
}

export async function askChat(doc_id, query) {
  const res = await fetch(`${BACKEND}/chat/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id, query })
  });
  if (!res.ok) throw new Error("Chat request failed");
  return res.json();  // { answer: "..."}
}

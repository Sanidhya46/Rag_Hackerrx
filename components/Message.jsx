import React from 'react';
import { CheckCircle, XCircle, FileText, DollarSign } from 'lucide-react';

function Message({ sender, content, isStructure }) {
  
  // Render structured JSON response
  if (isStructure) {
    const isApproved = content.decision?.toLowerCase() === 'approved';
    return (
      <div className={`message bot-structured`}>
        <div className={`decision-header ${isApproved ? 'approved' : 'rejected'}`}>
          {isApproved ? <CheckCircle /> : <XCircle />}
          <span>Decision: {content.decision}</span>
        </div>
        <div className="details-grid">
          <div className="detail-item">
            <DollarSign className="detail-icon" />
            <div>
              <p className="detail-label">Amount</p>
              <p className="detail-value">{content.amount}</p>
            </div>
          </div>
          <div className="detail-item justification">
            <FileText className="detail-icon" />
            <div>
              <p className="detail-label">Justification</p>
              <ul className="justification-list">
                {content.justification?.map((clause, index) => (
                  <li key={index}>{clause}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Render a simple text message
  return (
    <div className={`message ${sender}`}>
      <div className="message-content">
        <p>{content}</p>
      </div>
    </div>
  );
}

export default Message;
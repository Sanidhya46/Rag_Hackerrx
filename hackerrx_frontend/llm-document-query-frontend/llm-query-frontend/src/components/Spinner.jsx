import React from 'react';

function Spinner({ simple = false }) {
  if (simple) {
    return <div className="spinner-simple" />;
  }
  return (
    <div className="spinner-overlay">
      <div className="spinner-full" />
    </div>
  );
}

export default Spinner;

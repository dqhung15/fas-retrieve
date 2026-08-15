CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS interactions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id VARCHAR(255) NOT NULL,
  query_text TEXT NOT NULL,
  reference_image_name VARCHAR(255) NOT NULL,
  selected_target_name VARCHAR(255),
  is_used_for_training BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_training_flag ON interactions(is_used_for_training);
CREATE INDEX idx_created_at ON interactions(created_at);
from services.api.app.model_loader import ml_state
from services.api.app.schemas import TrackInteractionRequest

def log_interaction(payload: TrackInteractionRequest):
  cursor = ml_state.db_conn.cursor()
  try:
    cursor.execute(
      """
      INSERT INTO interactions 
      (session_id, query_text, reference_image_name, selected_target_name) 
      VALUES (%s, %s, %s, %s)
      """,
      (payload.session_id, payload.query_text, payload.reference_image_name, payload.selected_target_name)
    )
    ml_state.db_conn.commit()
  except Exception as e:
    ml_state.db_conn.rollback()
    raise e
  finally:
    cursor.close()
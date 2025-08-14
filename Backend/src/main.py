from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.agents import get_agent_executor
import requests

app = FastAPI(title="Insurance Chatbot Backend")

class QueryRequest(BaseModel):
    user_input: str
    schema: Optional[str] = None  # <-- Add schema as optional input

@app.post("/test")
async def test_query(request: QueryRequest):
    executor = get_agent_executor()
    try:
        # Pass both input and schema to the executor
        inputs = {"input": request.user_input}
        if request.schema:
            inputs["schema"] = request.schema

        result = await executor.invoke(inputs)
        return {"result": result.get("output", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Example schema string you can customize
    schema_str = (
        "Table clients: id (uuid), nom (text), prenom (text), date_naissance (date), email (text), telephone (text), adresse (text), created_at (timestamp with time zone); "
        "Table contrats: id (uuid), client_id (uuid), produit_id (uuid), numero_contrat (text), date_debut (date), date_fin (date), montant_annuel (numeric), statut (text), created_at (timestamp with time zone); "
        "Table paiements: id (uuid), contrat_id (uuid), date_paiement (date), montant (numeric), mode_paiement (text), created_at (timestamp with time zone); "
        "Table produits_assurance: id (uuid), nom (text), description (text), created_at (timestamp with time zone); "
        "Table sinistres: id (uuid), contrat_id (uuid), date_sinistre (date), description (text), montant_estime (numeric), statut (text), created_at (timestamp with time zone)"
    )

    response = requests.post(
        "http://localhost:8000/test",
        json={"user_input": "Show me insurance claims summary", "schema": schema_str}
    )

    print("Status:", response.status_code)
    print("Response:", response.json())
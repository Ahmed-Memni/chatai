import requests

schema = """
Table clients: id (uuid), nom (text), prenom (text), date_naissance (date), email (text), telephone (text), adresse (text), created_at (timestamp with time zone)
Table contrats: id (uuid), client_id (uuid), produit_id (uuid), numero_contrat (text), date_debut (date), date_fin (date), montant_annuel (numeric), statut (text), created_at (timestamp with time zone)
Table paiements: id (uuid), contrat_id (uuid), date_paiement (date), montant (numeric), mode_paiement (text), created_at (timestamp with time zone)
Table produits_assurance: id (uuid), nom (text), description (text), created_at (timestamp with time zone)
Table sinistres: id (uuid), contrat_id (uuid), date_sinistre (date), description (text), montant_estime (numeric), statut (text), created_at (timestamp with time zone)
"""

response = requests.post(
    "http://localhost:8000/test",
    json={
        "user_input": "What are the emails of all clients?",
        "schema": schema
    }
)

print("Status:", response.status_code)
print("Response:", response.json())
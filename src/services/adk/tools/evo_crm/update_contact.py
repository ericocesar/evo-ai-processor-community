"""
Update Contact Tool

This tool allows agents to update contact information in the CRM.
"""

import uuid
from typing import Dict, Any, Optional, List
from google.adk.tools import FunctionTool, ToolContext
from src.utils.logger import setup_logger
from src.services.adk.tools.evo_crm.base import EvoCrmClient

logger = setup_logger(__name__)


def _extract_conversation_id_from_metadata(tool_context: ToolContext) -> Optional[str]:
    """Extracts conversation ID from tool_context metadata."""
    evoai_crm_data = tool_context.state.get("evoai_crm_data", {})
    return evoai_crm_data.get("conversation_id") or evoai_crm_data.get("conversation", {}).get("id")


def _extract_contact_id_from_metadata(tool_context: ToolContext) -> Optional[str]:
    """Extracts contact ID from tool_context metadata."""
    if not tool_context or not hasattr(tool_context, 'state'):
        return None
    
    state = tool_context.state
    
    # Try to get contact directly from state (most common case)
    contact = state.get("contact")
    if isinstance(contact, dict) and contact.get("id"):
        return str(contact.get("id"))
    
    # Try evoai_crm_data
    evoai_crm_data = state.get("evoai_crm_data", {})
    if isinstance(evoai_crm_data, dict):
        # Try contact inside evoai_crm_data
        contact_data = evoai_crm_data.get("contact", {})
        if isinstance(contact_data, dict) and contact_data.get("id"):
            return str(contact_data.get("id"))
        
        # Try direct keys
        contact_id = evoai_crm_data.get("contactId") or evoai_crm_data.get("contact_id")
        if contact_id:
            return str(contact_id)
    
    # Try direct state keys
    contact_id = state.get("contactId") or state.get("contact_id")
    if contact_id:
        return str(contact_id)
    
    return None


def create_update_contact_tool() -> FunctionTool:
    """
    Factory function to create an 'update_contact' tool.
    """
    client = EvoCrmClient()

    async def update_contact(
        name: Optional[str] = None,
        email: Optional[str] = None,
        phone_number: Optional[str] = None,
        identifier: Optional[str] = None,
        website: Optional[str] = None,
        location: Optional[str] = None,
        industry: Optional[str] = None,
        tax_id: Optional[str] = None,
        custom_attributes: Optional[Dict[str, Any]] = None,
        additional_attributes: Optional[Dict[str, Any]] = None,
        labels: Optional[List[str]] = None,
        contact_id: Optional[str] = None,
        tool_context: Optional[ToolContext] = None,
    ) -> Dict[str, Any]:
        """Update contact information in the CRM.
        
        Use this tool when:
        - The user provides new contact information
        - You need to update contact details (name, email, phone, etc.)
        - The user requests to change their contact information
        - You want to add labels or custom attributes to a contact
        
        Args:
            name: Contact's full name (optional)
            email: Contact's email address (optional)
            phone_number: Contact's phone number in E.164 format (e.g., +5511999999999) (optional)
            identifier: Contact's unique identifier (optional)
            website: Contact's website URL (optional)
            location: Contact's location/address (optional)
            industry: Contact's industry (optional)
            tax_id: Contact's tax ID (optional, max 14 characters)
            custom_attributes: Dictionary of custom attributes to update (optional). Use this for all business and contact fields (e.g., tipo_cliente, concessionaria, cep, cidade, estado, rua, bairro, numero_casa, complemento_endereco, valor_medio_conta, economia_mensal, ganho, uc_numero, cpf_titular, nome_titular, data_nascimento_titular).
            additional_attributes: Dictionary of additional system/integration attributes (optional). Do NOT use this for custom contact fields.
            labels: List of label names to add to the contact (optional)
            contact_id: The ID of the contact to update (optional, will be automatically extracted from context)
            tool_context: The tool context containing session information (automatically provided)
            
        Returns:
            Dictionary with update status and details:
            {
                "status": "success" | "error",
                "message": "Human-readable message",
                "contact_id": "...",
                "contact": {...}  # Updated contact data
            }
        """
        try:
            # Extract contact_id from metadata if not provided
            effective_contact_id = contact_id
            if not effective_contact_id and tool_context:
                effective_contact_id = _extract_contact_id_from_metadata(tool_context)
                if effective_contact_id:
                    logger.info(f"Extracted contact_id from metadata: {effective_contact_id}")
            
            # Validate required parameters
            if not effective_contact_id:
                return {
                    "status": "error",
                    "message": "contact_id is required. It should be automatically extracted from the conversation context, but if not available, please provide it explicitly.",
                    "contact_id": None,
                }
            
            # Build request body with only provided fields
            request_body: Dict[str, Any] = {}
            
            if name is not None:
                request_body["name"] = name
            if email is not None:
                request_body["email"] = email
            if phone_number is not None:
                request_body["phone_number"] = phone_number
            if identifier is not None:
                request_body["identifier"] = identifier
            if website is not None:
                request_body["website"] = website
            if location is not None:
                request_body["location"] = location
            if industry is not None:
                request_body["industry"] = industry
            if tax_id is not None:
                request_body["tax_id"] = tax_id
            # Defensive check: if the agent incorrectly passed defined custom attributes
            # inside additional_attributes, automatically move them to custom_attributes.
            if additional_attributes:
                known_custom_keys = {
                    "tipo_cliente", "nome_titular", "cpf_titular", "data_nascimento_titular",
                    "uc_numero", "concessionaria", "cep", "estado", "cidade", "bairro", "rua",
                    "numero_casa", "complemento_endereco", "valor_medio_conta", "economia_mensal",
                    "ganho", "interesse_continuar", "qualificacao_energia", "economia_estimada_10",
                    "tipo_ligacao"
                }
                
                if custom_attributes is None:
                    custom_attributes = {}
                
                # Move keys
                keys_to_move = [k for k in list(additional_attributes.keys()) if k in known_custom_keys]
                for k in keys_to_move:
                    logger.warning(
                        f"[update_contact] Defensive check: moving key '{k}' "
                        f"from additional_attributes to custom_attributes."
                    )
                    custom_attributes[k] = additional_attributes.pop(k)
                
                # Clear additional_attributes if now empty
                if not additional_attributes:
                    additional_attributes = None

            if custom_attributes is not None:
                # Filter out None and empty/blank strings
                cleaned_custom = {}
                for k, v in custom_attributes.items():
                    if v is not None:
                        if isinstance(v, str):
                            if v.strip() != "":
                                cleaned_custom[k] = v.strip()
                        else:
                            cleaned_custom[k] = v
                if cleaned_custom:
                    request_body["custom_attributes"] = cleaned_custom
                
            if additional_attributes is not None:
                # Filter out None and empty/blank strings
                cleaned_additional = {}
                for k, v in additional_attributes.items():
                    if v is not None:
                        if isinstance(v, str):
                            if v.strip() != "":
                                cleaned_additional[k] = v.strip()
                        else:
                            cleaned_additional[k] = v
                if cleaned_additional:
                    request_body["additional_attributes"] = cleaned_additional
            if labels is not None:
                request_body["labels"] = labels
            
            # Validate that at least one field is being updated
            if not request_body:
                return {
                    "status": "error",
                    "message": "At least one field must be provided to update the contact.",
                    "contact_id": effective_contact_id,
                }
            
            logger.info(
                f"Updating contact {effective_contact_id} with fields: {list(request_body.keys())}"
            )
            
            # Make API request to update contact
            endpoint = f"/contacts/{effective_contact_id}"
            
            try:
                response = await client.put(
                    endpoint=endpoint,
                    json_data=request_body,
                )
                
                logger.info(
                    f"Successfully updated contact {effective_contact_id}"
                )
                
                # Extract contact info from response
                contact_data = response.get("payload", {}) if isinstance(response, dict) else response
                
                success_message = (
                    f"Contact {effective_contact_id} successfully updated"
                    + (f" (name: {contact_data.get('name', 'N/A')})" if contact_data.get('name') else "")
                )
                
                return {
                    "status": "success",
                    "message": success_message,
                    "contact_id": effective_contact_id,
                    "contact": contact_data,
                }
                
            except Exception as api_error:
                error_message = str(api_error)
                
                # Provide more specific error messages
                if "404" in error_message or "not found" in error_message.lower():
                    error_message = (
                        f"Contact {effective_contact_id} not found. "
                        "Please verify the contact ID is correct."
                    )
                elif "401" in error_message or "unauthorized" in error_message.lower():
                    error_message = (
                        "Authentication failed. Please check EVOAI_CRM_API_TOKEN configuration."
                    )
                elif "400" in error_message or "bad request" in error_message.lower():
                    error_message = (
                        f"Invalid request. Please check the provided data is valid. "
                        f"Common issues: invalid email format, invalid phone number format (must be E.164: +5511999999999), "
                        f"or invalid tax_id length (max 14 characters)."
                    )
                elif "422" in error_message or "unprocessable" in error_message.lower():
                    error_message = (
                        f"Validation error. Please check the provided data. "
                        f"Common issues: email already exists, phone number already exists, or invalid format."
                    )
                
                logger.error(f"Failed to update contact: {error_message}")
                
                return {
                    "status": "error",
                    "message": error_message,
                    "contact_id": effective_contact_id,
                    "error": str(api_error),
                }
                
        except Exception as e:
            error_msg = f"Unexpected error updating contact: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "contact_id": effective_contact_id if 'effective_contact_id' in locals() else None,
                "error": str(e),
            }

    update_contact.__name__ = "update_contact_attributes"
    update_contact.__doc__ = """Update contact information in the CRM.
    
    Use this tool to update contact details such as name, email, phone number,
    location, industry, custom attributes, labels, and more.
    
    When to use:
    - User provides new contact information
    - Need to update contact details
    - User requests to change their information
    - Want to add labels or custom attributes
    
    Args:
        name: Contact's full name (optional)
        email: Contact's email address (optional)
        phone_number: Contact's phone number in E.164 format (e.g., +5511999999999) (optional)
        identifier: Contact's unique identifier (optional)
        website: Contact's website URL (optional)
        location: Contact's location/address (optional)
        industry: Contact's industry (optional)
        tax_id: Contact's tax ID (optional, max 14 characters)
        custom_attributes: Dictionary of custom attributes to update (optional). Use this for all business and contact fields (e.g. tipo_cliente, concessionaria, cep, cidade, estado, rua, bairro, numero_casa, complemento_endereco, valor_medio_conta, economia_mensal, ganho, uc_numero, cpf_titular, nome_titular, data_nascimento_titular).
        additional_attributes: Dictionary of additional system/integration attributes (optional). Do NOT use this for custom contact fields.
        labels: List of label names to add to the contact (optional)
        contact_id: The ID of the contact to update (required, auto-extracted if not provided)
    
    Returns:
        Dictionary with update status and contact details
    """
    
    return FunctionTool(func=update_contact)


"""
ADK Tool factory para parsing de faturas de energia elétrica.

Uso no agent config:
    { "enable_fatura_parser": true }

A tool aceita URL pública ou base64 do arquivo (PDF ou imagem).
"""

import base64
import logging
from typing import Optional
from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

# MIME types aceitos
_MIME_PERMITIDOS = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
}

# Limite de 10MB para evitar abusos
_MAX_SIZE_BYTES = 10 * 1024 * 1024


def create_parse_fatura_tool() -> FunctionTool:
    """Factory que cria a tool parse_fatura_energia para o ADK."""

    async def parse_fatura_energia(
        fonte: str,
        content_type: str = "application/pdf",
    ) -> dict:
        """Extrai dados estruturados de uma fatura de energia elétrica (PDF ou imagem).

        Use esta ferramenta SEMPRE que o usuário enviar uma fatura de energia
        para análise. Ela extrai automaticamente os campos necessários para
        a qualificação (valor, concessionária, consumo, vencimento, etc.).

        Args:
            fonte: URL pública do arquivo OU string base64 do conteúdo.
                   Exemplos:
                   - "https://exemplo.com/fatura.pdf"
                   - "JVBERi0xLjQgMSAwIG9iag==" (base64)
            content_type: MIME type do arquivo.
                   Valores aceitos: "application/pdf", "image/png",
                   "image/jpeg", "image/webp".
                   Padrão: "application/pdf"

        Returns:
            Dicionário com:
            - sucesso (bool): se a extração foi bem-sucedida
            - campos (dict): dados extraídos da fatura:
                - concessionaria: Nome da distribuidora
                - numero_cliente: Código do cliente/instalação
                - vencimento: Data de vencimento (YYYY-MM-DD)
                - valor_total: Valor total a pagar (float, em R$)
                - consumo_kwh: Consumo do período em kWh (float)
                - classe_consumidor: Classe tarifária (Residencial, Comercial...)
                - periodo_consumo: Mês/período de referência
                - numero_fatura: Número da fatura
            - confianca: float de 0 a 1 (qualidade da extração)
            - metodo_extracao: "texto", "imagem" ou "falhou"
            - mensagem_erro: descrição do erro, se houver
        """
        import httpx
        from src.services.adk.tools.fatura_energia.parser import parse_fatura

        # Normalizar content_type
        content_type_clean = content_type.split(";")[0].strip().lower()

        # Validar MIME type
        if content_type_clean not in _MIME_PERMITIDOS:
            return {
                "sucesso": False,
                "mensagem_erro": (
                    f"Tipo de arquivo não suportado: '{content_type_clean}'. "
                    f"Tipos aceitos: PDF, PNG, JPEG, WEBP."
                ),
            }

        file_bytes: Optional[bytes] = None

        # Determinar se fonte é URL ou base64
        fonte_stripped = fonte.strip()
        is_url = fonte_stripped.startswith("http://") or fonte_stripped.startswith("https://")

        if is_url:
            # Download via URL com timeout agressivo
            try:
                async with httpx.AsyncClient(
                    timeout=15.0,
                    follow_redirects=False,  # Segurança: não seguir redirects arbitrários
                ) as client:
                    response = await client.get(fonte_stripped)
                    response.raise_for_status()
                    file_bytes = response.content
            except httpx.TimeoutException:
                return {
                    "sucesso": False,
                    "mensagem_erro": "Timeout ao baixar o arquivo. Verifique se a URL está acessível.",
                }
            except httpx.HTTPStatusError as e:
                return {
                    "sucesso": False,
                    "mensagem_erro": f"Erro HTTP {e.response.status_code} ao baixar arquivo.",
                }
            except Exception as e:
                logger.error(f"Erro ao baixar arquivo: {e}")
                return {
                    "sucesso": False,
                    "mensagem_erro": f"Não foi possível baixar o arquivo: {str(e)}",
                }
        else:
            # Decodificar base64
            try:
                file_bytes = base64.b64decode(fonte_stripped)
            except Exception:
                return {
                    "sucesso": False,
                    "mensagem_erro": "Formato inválido. Forneça uma URL https:// ou base64 válido.",
                }

        # Validar tamanho
        if len(file_bytes) > _MAX_SIZE_BYTES:
            return {
                "sucesso": False,
                "mensagem_erro": "Arquivo muito grande. Limite máximo de 10MB.",
            }

        # Processar conforme tipo
        is_pdf = "pdf" in content_type_clean
        result = parse_fatura(
            pdf_bytes=file_bytes if is_pdf else None,
            image_bytes=file_bytes if not is_pdf else None,
            content_type=content_type_clean,
        )

        # Para imagens, não há extração via parser — informar ao agente
        if not is_pdf and not result.get("campos"):
            result["aviso"] = (
                "Imagem recebida. Para extrair dados de imagens de fatura, "
                "descreva os campos visíveis na imagem para o usuário ou utilize "
                "um modelo de visão computacional."
            )

        return result

    # Atribuir nome e docstring explicitamente (exigido pelo ADK)
    parse_fatura_energia.__name__ = "parse_fatura_energia"

    return parse_fatura_energia

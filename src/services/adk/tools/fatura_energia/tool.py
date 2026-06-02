"""
ADK Tool factory para parsing de faturas de energia elétrica.

Uso no agent config:
    { "enable_fatura_parser": true }

A tool aceita URL pública ou base64 do arquivo (PDF ou imagem).
"""

import base64
import logging
from typing import Optional
from google.adk.tools import FunctionTool, ToolContext

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
        fonte: Optional[str] = None,
        content_type: str = "application/pdf",
        tool_context: ToolContext = None,
    ) -> dict:
        """Extrai dados estruturados de uma fatura de energia elétrica (PDF ou imagem).

        Use esta ferramenta SEMPRE que o usuário enviar uma fatura de energia
        para análise. Ela extrai automaticamente os campos necessários para
        a qualificação (valor, concessionária, consumo, vencimento, etc.).

        Quando o usuário ENVIAR um arquivo (PDF ou imagem) na mensagem,
        chame esta ferramenta SEM fornecer o parâmetro 'fonte' — ela carregará
        o arquivo automaticamente da sessão.

        Quando o arquivo vier como URL ou base64 explícito, passe em 'fonte'.

        Args:
            fonte: (Opcional) URL pública do arquivo OU string base64 do conteúdo.
                   Se omitido, a ferramenta carrega o arquivo mais recente da sessão.
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

        # Se fonte não for fornecido (ou for "undefined"), carregar da sessão
        _fonte = (fonte or "").strip()
        if not _fonte or _fonte.lower() == "undefined":
            if tool_context is None:
                return {
                    "sucesso": False,
                    "mensagem_erro": "Nenhum arquivo fornecido e contexto de sessão indisponível.",
                }
            try:
                artifact_keys = await tool_context.list_artifact_keys()
                # Filtrar por extensões de fatura (PDF e imagens)
                _fatura_exts = (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif")
                fatura_keys = [k for k in artifact_keys if k.lower().endswith(_fatura_exts)]
                if not fatura_keys:
                    return {
                        "sucesso": False,
                        "mensagem_erro": (
                            "Nenhum arquivo de fatura encontrado na sessão. "
                            "Peça ao usuário que envie o arquivo PDF ou imagem da fatura."
                        ),
                    }
                # Pegar o mais recente (último da lista)
                artifact_filename = fatura_keys[-1]
                artifact_part = await tool_context.load_artifact(filename=artifact_filename)
                if artifact_part is None or not (hasattr(artifact_part, "inline_data") and artifact_part.inline_data):
                    return {
                        "sucesso": False,
                        "mensagem_erro": f"Não foi possível carregar o arquivo '{artifact_filename}' da sessão.",
                    }
                file_bytes = artifact_part.inline_data.data
                content_type_clean = artifact_part.inline_data.mime_type or content_type_clean
                logger.info(f"parse_fatura_energia: carregando '{artifact_filename}' da sessão ({len(file_bytes)} bytes)")
            except Exception as artifact_err:
                logger.error(f"Erro ao carregar artefato da sessão: {artifact_err}")
                return {
                    "sucesso": False,
                    "mensagem_erro": f"Erro ao acessar o arquivo da sessão: {str(artifact_err)}",
                }

            # Processar conforme tipo
            is_pdf = "pdf" in content_type_clean
            result = parse_fatura(
                pdf_bytes=file_bytes if is_pdf else None,
                image_bytes=file_bytes if not is_pdf else None,
                content_type=content_type_clean,
            )
            if not is_pdf and not result.get("campos"):
                result["aviso"] = (
                    "Imagem recebida. Para extrair dados de imagens de fatura, "
                    "descreva os campos visíveis na imagem para o usuário ou utilize "
                    "um modelo de visão computacional."
                )
            return result

        # A partir daqui: fonte foi fornecido explicitamente
        fonte_stripped = _fonte
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

import httpx
import logging
from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)


def create_via_cep_tool() -> FunctionTool:
    async def via_cep(cep: str) -> dict:
        """Consulta endereço brasileiro pelo CEP usando a API ViaCEP.

        Args:
            cep: CEP de 8 dígitos (apenas números, ex: 59131170)

        Returns:
            Dicionário com dados do endereço (logradouro, bairro, localidade, uf, etc.)
            ou {"erro": true} se o CEP não for encontrado.
        """
        cep_clean = cep.replace("-", "").strip()
        if not cep_clean.isdigit() or len(cep_clean) != 8:
            return {
                "error": "CEP inválido. Informe 8 dígitos numéricos."
            }

        url = f"https://viacep.com.br/ws/{cep_clean}/json/"
        logger.info(f"Consultando ViaCEP: {url}")

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                return data
        except httpx.TimeoutException:
            return {"error": "Timeout ao consultar ViaCEP"}
        except httpx.HTTPStatusError as e:
            return {"error": f"Erro HTTP {e.response.status_code} ao consultar ViaCEP"}
        except Exception as e:
            logger.error(f"Erro no via_cep tool: {e}")
            return {"error": f"Erro inesperado: {str(e)}"}

    via_cep.__name__ = "via_cep"
    via_cep.__doc__ = """Consulta endereço brasileiro pelo CEP usando a API ViaCEP.

    Args:
        cep: CEP de 8 dígitos (apenas números, ex: 59131170)

    Returns:
        Dicionário com os dados do endereço. Exemplo de resposta:
        {
            "cep": "59131-170",
            "logradouro": "Rua Barão de Cocais",
            "complemento": "",
            "bairro": "Pajuçara",
            "localidade": "Natal",
            "uf": "RN",
            "estado": "Rio Grande do Norte",
            "regiao": "Nordeste",
            "ibge": "2408102",
            "ddd": "84",
            "siafi": "1761"
        }
        Se o CEP não existir, retorna {"erro": true}.
    """

    return via_cep

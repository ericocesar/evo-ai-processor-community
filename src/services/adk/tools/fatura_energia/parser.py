"""
Parser de faturas de energia elétrica.

Suporta:
- PDFs digitais: extração de texto via pdfplumber
- PDFs escaneados: renderização via pymupdf (fallback)
- Imagens (PNG/JPEG/WEBP): retorna bytes para uso com LLM vision

Extrai campos estruturados usando regex calibrado para as principais
distribuidoras brasileiras (CEMIG, ENEL, LIGHT, CELPE, CPFL, COELBA, etc.)
"""

import re
import io
import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns para extração de campos
# ---------------------------------------------------------------------------

# Valor total a pagar
_RE_VALOR = re.compile(
    r"(?:valor\s+total|total\s+a\s+pagar|valor\s+a\s+pagar|total\s+da\s+fatura)"
    r"[\s\S]{0,60}?R?\$?\s*([\d]{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))",
    re.IGNORECASE,
)

# Consumo em kWh
_RE_CONSUMO = re.compile(
    r"(?:consumo\s*(?:de\s*)?(?:energia)?|kwh\s+faturado)"
    r"[\s\S]{0,40}?([\d.,]+)\s*kwh",
    re.IGNORECASE,
)

# Data de vencimento
_RE_VENCIMENTO = re.compile(
    r"(?:vencimento|data\s+de\s+vencimento|vcto\.?|vecto\.?)"
    r"[\s:.\-]+(\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4})",
    re.IGNORECASE,
)

# Número do cliente / instalação
_RE_CLIENTE = re.compile(
    r"(?:n[uú]mero\s+do\s+cliente|c[oó]digo\s+(?:do\s+)?cliente|"
    r"n[uú]mero\s+da\s+instala[cç][aã]o|instala[cç][aã]o|cod\.?\s*cliente)"
    r"[\s:.\-]+([\w\d.\-/]+)",
    re.IGNORECASE,
)

# Período de referência
_RE_PERIODO = re.compile(
    r"(?:per[ií]odo\s+de\s+(?:refer[eê]ncia|faturamento)|mes\s+de\s+refer[eê]ncia|"
    r"refer[eê]ncia)"
    r"[\s:.\-]+([\w/\-\s]+(?:\d{4}|\d{2}/\d{4}))",
    re.IGNORECASE,
)

# Classe de consumidor
_RE_CLASSE = re.compile(
    r"(?:classe\s*(?:de\s*)?consumidor|modalidade|tarifa\s+(?:vigente|aplicada)?)"
    r"[\s:.\-]+([A-ZÀ-Ú][A-Za-zÀ-ú\s/-]{3,40})",
    re.IGNORECASE,
)

# Número da fatura / documento
_RE_NUM_FATURA = re.compile(
    r"(?:n[uú]mero\s+da\s+(?:fatura|nota)|fatura\s+n[oº]?|documento\s+n[oº]?)"
    r"[\s:.\-]+([\w\d.\-/]+)",
    re.IGNORECASE,
)

# Concessionárias brasileiras conhecidas
_CONCESSIONARIAS = [
    "CEMIG", "ENEL", "LIGHT", "CELPE", "CPFL", "COELBA", "CELG", "CELESC",
    "ELEKTRO", "EDP", "ENERGISA", "COPEL", "ENEVA", "AMPLA", "BANDEIRANTE",
    "ELETROPAULO", "CELPA", "COELCE", "CEMAR", "COSERN", "CEAL", "CERON",
    "CEPISA", "CELTINS", "DMED", "EMG", "ESS", "RGE", "CNEE", "EFLJC",
    "SULGIPE", "ELFSM", "CERR", "COOPERALIANÇA", "AES", "CHESP", "CEB",
    "NEOENERGIA", "EQUATORIAL",
]

_RE_CONCESSIONARIA = re.compile(
    r"(?:" + "|".join(re.escape(c) for c in _CONCESSIONARIAS) + r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Funções de extração
# ---------------------------------------------------------------------------

def _normalizar_valor(raw: str) -> Optional[float]:
    """Converte string de valor brasileiro para float."""
    try:
        clean = raw.strip().replace(".", "").replace(",", ".")
        return float(clean)
    except (ValueError, AttributeError):
        return None


def _normalizar_data(raw: str) -> str:
    """Normaliza data para formato ISO 8601 (YYYY-MM-DD)."""
    raw = raw.strip()
    # Tenta DD/MM/YYYY ou DD-MM-YYYY
    m = re.match(r"(\d{2})[/\-\.](\d{2})[/\-\.](\d{4})", raw)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    # Tenta DD/MM/YY
    m = re.match(r"(\d{2})[/\-\.](\d{2})[/\-\.](\d{2})$", raw)
    if m:
        year = int(m.group(3))
        full_year = 2000 + year if year < 50 else 1900 + year
        return f"{full_year}-{m.group(2)}-{m.group(1)}"
    return raw


def _extrair_campos(text: str) -> dict:
    """Extrai campos estruturados de fatura a partir de texto bruto."""
    result = {}

    # Valor total
    m = _RE_VALOR.search(text)
    if m:
        result["valor_total"] = _normalizar_valor(m.group(1))

    # Consumo kWh
    m = _RE_CONSUMO.search(text)
    if m:
        try:
            result["consumo_kwh"] = float(m.group(1).replace(".", "").replace(",", "."))
        except ValueError:
            pass

    # Vencimento
    m = _RE_VENCIMENTO.search(text)
    if m:
        result["vencimento"] = _normalizar_data(m.group(1))

    # Número do cliente
    m = _RE_CLIENTE.search(text)
    if m:
        result["numero_cliente"] = m.group(1).strip()

    # Período de consumo
    m = _RE_PERIODO.search(text)
    if m:
        result["periodo_consumo"] = m.group(1).strip()

    # Classe de consumidor
    m = _RE_CLASSE.search(text)
    if m:
        result["classe_consumidor"] = m.group(1).strip()

    # Número da fatura
    m = _RE_NUM_FATURA.search(text)
    if m:
        result["numero_fatura"] = m.group(1).strip()

    # Concessionária
    m = _RE_CONCESSIONARIA.search(text)
    if m:
        result["concessionaria"] = m.group(0).upper()

    return result


# ---------------------------------------------------------------------------
# Extração de PDF digital (texto)
# ---------------------------------------------------------------------------

def extrair_texto_pdf(pdf_bytes: bytes) -> str:
    """Extrai texto de PDF usando pdfplumber. Retorna string com todo o texto."""
    try:
        import pdfplumber  # type: ignore
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    except ImportError:
        logger.warning("pdfplumber não instalado — usando fallback pymupdf")
        return extrair_texto_pdf_fitz(pdf_bytes)
    except Exception as e:
        logger.error(f"Erro ao extrair texto do PDF com pdfplumber: {e}")
        return ""


def extrair_texto_pdf_fitz(pdf_bytes: bytes) -> str:
    """Extrai texto de PDF usando pymupdf (fallback). Retorna string com todo o texto."""
    try:
        import fitz  # type: ignore  # pymupdf
        text_parts = []
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text_parts.append(page.get_text())
        return "\n".join(text_parts)
    except ImportError:
        logger.error("pymupdf (fitz) não instalado. Instale com: pip install pymupdf")
        return ""
    except Exception as e:
        logger.error(f"Erro ao extrair texto do PDF com pymupdf: {e}")
        return ""


def pdf_para_imagem_base64(pdf_bytes: bytes, dpi: int = 150) -> Optional[str]:
    """
    Renderiza a primeira página de um PDF como PNG e retorna base64.
    Usado como fallback quando o PDF é escaneado e não contém texto.
    """
    try:
        import fitz  # type: ignore  # pymupdf
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        # Aumentar resolução para melhor OCR visual
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        return base64.b64encode(png_bytes).decode("utf-8")
    except ImportError:
        logger.warning("pymupdf não instalado — não é possível converter PDF para imagem")
        return None
    except Exception as e:
        logger.error(f"Erro ao converter PDF para imagem: {e}")
        return None


# ---------------------------------------------------------------------------
# Função principal de parsing
# ---------------------------------------------------------------------------

def parse_fatura(
    pdf_bytes: Optional[bytes] = None,
    image_bytes: Optional[bytes] = None,
    content_type: str = "application/pdf",
) -> dict:
    """
    Processa uma fatura de energia e retorna campos estruturados.

    Args:
        pdf_bytes: Bytes do PDF (se arquivo for PDF)
        image_bytes: Bytes da imagem (se arquivo for imagem)
        content_type: MIME type do arquivo

    Returns:
        Dicionário com campos extraídos + metadados de confiança
    """
    raw_text = ""
    pdf_imagem_b64 = None

    if pdf_bytes and "pdf" in content_type.lower():
        raw_text = extrair_texto_pdf(pdf_bytes)

        if len(raw_text.strip()) < 50:
            # PDF provavelmente escaneado — tenta renderização visual
            logger.info("Texto insuficiente no PDF — tentando renderização como imagem")
            pdf_imagem_b64 = pdf_para_imagem_base64(pdf_bytes)
            raw_text = ""  # Sem texto, o agente deve usar LLM vision

    campos = _extrair_campos(raw_text) if raw_text else {}

    # Calcular confiança baseada na quantidade de campos extraídos
    campos_principais = ["valor_total", "concessionaria", "vencimento", "consumo_kwh"]
    campos_encontrados = sum(1 for c in campos_principais if c in campos)
    confianca = campos_encontrados / len(campos_principais)

    return {
        "sucesso": True,
        "confianca": confianca,  # 0.0 a 1.0
        "campos": campos,
        "raw_text": raw_text[:2000] if raw_text else None,  # Limitado para não sobrecarregar contexto
        "pdf_como_imagem_base64": pdf_imagem_b64,  # Para uso com LLM vision se necessário
        "metodo_extracao": "texto" if raw_text else ("imagem" if pdf_imagem_b64 else "falhou"),
    }

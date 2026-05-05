"""LLM and Embedding model initialization for Azure OpenAI.

All LLM instances use max_retries=5 with built-in exponential backoff
for rate limit (429) handling.
"""

from functools import lru_cache

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from app.core.config import get_settings

# Azure OpenAI SDK handles 429 retries internally when max_retries > 0.
# We set max_retries=5 (up from 3) for better rate limit resilience.
_MAX_RETRIES = 5
_REQUEST_TIMEOUT = 180  # 3 minutes (up from 2 for large batches)


@lru_cache()
def get_azure_llm() -> AzureChatOpenAI:
    """Get Azure OpenAI Chat LLM instance for generation (temperature=0.3)."""
    settings = get_settings()
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment_name,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=0.3,
        max_tokens=4096,
        request_timeout=_REQUEST_TIMEOUT,
        max_retries=_MAX_RETRIES,
    )


@lru_cache()
def get_question_llm() -> AzureChatOpenAI:
    """Get Azure OpenAI Chat LLM for question generation (higher temperature for diversity)."""
    settings = get_settings()
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment_name,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=0.7,
        max_tokens=4096,
        request_timeout=_REQUEST_TIMEOUT,
        max_retries=_MAX_RETRIES,
    )


@lru_cache()
def get_answer_llm() -> AzureChatOpenAI:
    """Get Azure OpenAI Chat LLM for answer generation (low temperature for faithfulness)."""
    settings = get_settings()
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment_name,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
        max_tokens=4096,
        request_timeout=_REQUEST_TIMEOUT,
        max_retries=_MAX_RETRIES,
    )


@lru_cache()
def get_validation_llm() -> AzureChatOpenAI:
    """Get separate Azure OpenAI model for quality validation (avoids self-evaluation bias)."""
    settings = get_settings()
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_validation_deployment_name,
        azure_endpoint=settings.azure_openai_validation_endpoint,
        api_key=settings.azure_openai_validation_api_key,
        api_version=settings.azure_openai_validation_api_version,
        temperature=0.0,
        max_tokens=4096,
        request_timeout=_REQUEST_TIMEOUT,
        max_retries=_MAX_RETRIES,
    )


@lru_cache()
def get_azure_embeddings() -> AzureOpenAIEmbeddings:
    """Get Azure OpenAI Embeddings instance (text-embedding-ada-002)."""
    settings = get_settings()
    return AzureOpenAIEmbeddings(
        azure_deployment=settings.azure_openai_embedding_deployment_name,
        azure_endpoint=settings.azure_openai_embedding_endpoint,
        api_key=settings.azure_openai_embedding_api_key,
        api_version=settings.azure_openai_embedding_api_version,
        request_timeout=_REQUEST_TIMEOUT,
        max_retries=_MAX_RETRIES,
    )

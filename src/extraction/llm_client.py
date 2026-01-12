from typing import Literal
from tenacity import retry, stop_after_attempt, wait_exponential
import openai

try:
    # Try new google-genai package first (recommended)
    from google import genai
    from google.genai import types
    GENAI_VERSION = "new"
except ImportError:
    try:
        # Fall back to deprecated google.generativeai
        import google.generativeai as genai
        GENAI_VERSION = "old"
    except ImportError:
        GENAI_VERSION = None

from ..config import LLMSettings


class LLMClient:
    """
    Client for LLM API calls
    Supports: OpenAI (gpt-3.5-turbo, gpt-4) and Google Gemini (gemini-pro)
    """
    
    def __init__(self, settings: LLMSettings):
        self.settings = settings
        
        # Initialize clients based on provider
        if settings.provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env")
            self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
            print(f"Initialized OpenAI client with model: {settings.model}")
            
        elif settings.provider == "gemini":
            if not settings.gemini_api_key:
                raise ValueError("Gemini API key is required. Set GEMINI_API_KEY in .env")
            
            if GENAI_VERSION is None:
                raise ValueError("Google Generative AI package not installed. Install with: pip install google-genai")
            
            if GENAI_VERSION == "new":
                # Use new google-genai package
                self.genai_client = genai.Client(api_key=settings.gemini_api_key)
                self.gemini_model_name = settings.model
                print(f"Initialized Gemini client (google-genai) with model: {settings.model}")
            else:
                # Use deprecated google.generativeai package
                genai.configure(api_key=settings.gemini_api_key)
                self.gemini_model = genai.GenerativeModel(settings.model)
                print(f"Initialized Gemini client (deprecated) with model: {settings.model}")
        
        else:
            raise ValueError(f"Unknown provider: {settings.provider}. Use 'openai' or 'gemini'")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def extract(self, prompt: str) -> str:
        """
        Call LLM for extraction.
        Returns raw response text.
        
        Automatically uses the configured provider (OpenAI or Gemini)
        """
        print(f"Calling {self.settings.provider} LLM ({self.settings.model})")
        
        if self.settings.provider == "openai":
            return self._extract_openai(prompt)
        elif self.settings.provider == "gemini":
            return self._extract_gemini(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.settings.provider}")
    
    def _extract_openai(self, prompt: str) -> str:
        """
        Extract using OpenAI
        Free tier: gpt-3.5-turbo (best for testing)
        Paid: gpt-4-turbo-preview (best quality)
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a knowledge extraction expert. Extract structured data exactly as instructed. Return ONLY valid JSON, no markdown formatting."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                timeout=self.settings.timeout
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise
    
    def _extract_gemini(self, prompt: str) -> str:
        """
        Extract using Google Gemini
        Free tier: gemini-pro (generous limits, great quality)
        
        Gemini free tier limits (as of 2024):
        - 60 requests per minute
        - 1 million tokens per minute
        - Free forever for moderate use
        """
        try:
            # Gemini needs a specific instruction format
            full_prompt = f"""You are a knowledge extraction expert. Extract structured data exactly as instructed.

CRITICAL: Return ONLY a valid JSON array. Do not include any markdown formatting, code blocks, or explanatory text.

{prompt}"""
            
            if GENAI_VERSION == "new":
                # Use new google-genai package
                response = self.genai_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.settings.temperature,
                        max_output_tokens=self.settings.max_tokens,
                    )
                )
                
                # Extract text from response
                if not response.text:
                    raise ValueError("Gemini returned empty response")
                
                return response.text
            else:
                # Use deprecated google.generativeai package
                generation_config = {
                    "temperature": self.settings.temperature,
                    "max_output_tokens": self.settings.max_tokens,
                }
                
                # Generate response
                response = self.gemini_model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                
                # Gemini returns response in .text attribute
                if not response.text:
                    raise ValueError("Gemini returned empty response")
                
                return response.text
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            raise


# ============================================================================
# HELPER: Get recommended free models
# ============================================================================

def get_free_model_recommendations():
    """
    Returns recommended free/cheap models for each provider
    """
    return {
        "openai": {
            "free_tier": None,  # OpenAI doesn't have a permanent free tier
            "cheapest": "gpt-3.5-turbo",  # ~$0.0005 per 1K tokens (input)
            "best_value": "gpt-3.5-turbo-16k",
            "note": "Use free trial credits or pay-as-you-go (very cheap for testing)"
        },
        "gemini": {
            "free_tier": "gemini-pro",  # FREE FOREVER with generous limits!
            "limits": {
                "requests_per_minute": 60,
                "tokens_per_minute": 1_000_000,
                "requests_per_day": 1500
            },
            "best_value": "gemini-pro",
            "note": "Completely free for personal/testing use - RECOMMENDED"
        }
    }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage for both providers
    """
    from ..config import LLMSettings
    
    print("=" * 70)
    print("LLM Client - Free API Testing")
    print("=" * 70)
    
    # Show recommendations
    print("\nFREE/CHEAP MODEL RECOMMENDATIONS:")
    recommendations = get_free_model_recommendations()
    
    print("\n1. GOOGLE GEMINI (RECOMMENDED - COMPLETELY FREE)")
    print(f"   Model: {recommendations['gemini']['free_tier']}")
    print(f"   Limits: {recommendations['gemini']['limits']}")
    print(f"   Note: {recommendations['gemini']['note']}")
    
    print("\n2. OPENAI")
    print(f"   Cheapest: {recommendations['openai']['cheapest']}")
    print(f"   Note: {recommendations['openai']['note']}")
    
    print("\n" + "=" * 70)
    print("To use Gemini (FREE), set in your .env:")
    print("DEFAULT_LLM_PROVIDER=gemini")
    print("DEFAULT_LLM_MODEL=gemini-pro")
    print("GEMINI_API_KEY=your_gemini_key_here")
    print("=" * 70)
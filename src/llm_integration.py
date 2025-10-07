"""
LLM Integration for IncomeTaxGPT
Handles prompt engineering, generation, and citation extraction
"""

from typing import Dict, List, Optional
import re
import json
import os

# Try to import ML dependencies, but don't fail if they're not available
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"Warning: ML dependencies not available: {e}")

class IncomeTaxGPT:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 load_in_4bit: bool = True):
        if not ML_AVAILABLE:
            raise ImportError("ML dependencies not available. Cannot initialize local LLM.")
        """
        Initialize IncomeTaxGPT with LLM
        
        Args:
            model_name: HuggingFace model name
            load_in_4bit: Use 4-bit quantization for lower memory
        """
        print(f"Loading model: {model_name}")
        
        # Quantization config for low memory usage
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        print("✓ Model loaded successfully")
        
        # System prompt template
        self.system_prompt = """You are IncomeTaxGPT, an AI assistant specialized in Indian Income Tax law. Your role is to provide accurate, helpful, and legally grounded guidance on tax-related queries.

CRITICAL INSTRUCTIONS:
1. Base ALL responses on the provided legal context
2. ALWAYS cite specific sections when making legal claims
3. Format citations as [Section XXX] or [Source N]
4. If information is not in the context, clearly state this
5. Provide plain-English explanations alongside legal text
6. End every response with a disclaimer

DISCLAIMER: "This is informational guidance only and not professional tax advice. For complex situations, please consult a qualified Chartered Accountant."
"""
    
    def create_prompt(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """
        Create formatted prompt for LLM
        
        Args:
            query: User query
            context: Retrieved context from documents
            conversation_history: Previous conversation (optional)
        
        Returns:
            Formatted prompt string
        """
        # Build conversation
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add history if available
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current query with context
        user_message = f"""Based on the following legal context, answer the user's question.

LEGAL CONTEXT:
{context}

USER QUESTION:
{query}

Provide a clear, accurate answer with proper citations."""
        
        messages.append({"role": "user", "content": user_message})
        
        # Format for Llama-2 chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def generate_response(self, 
                         prompt: str,
                         max_new_tokens: int = 512,
                         temperature: float = 0.3,
                         top_p: float = 0.9) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: Formatted prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more focused)
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response (remove prompt)
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response
    
    def extract_citations(self, response: str) -> List[str]:
        """
        Extract section citations from response
        """
        # Patterns for citations
        patterns = [
            r'\[Section\s+(\d+[A-Z]*)\]',
            r'Section\s+(\d+[A-Z]*)',
            r'\[Source\s+(\d+)\]'
        ]
        
        citations = set()
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            citations.update(matches)
        
        return sorted(list(citations))
    
    def answer_query(self,
                    query: str,
                    retrieval_pipeline,
                    tax_calculators=None,
                    conversation_history: List[Dict] = None) -> Dict:
        """
        Main function to answer user query
        
        Args:
            query: User question
            retrieval_pipeline: RetrievalPipeline instance
            tax_calculators: TaxCalculators instance (optional)
            conversation_history: Previous messages (optional)
        
        Returns:
            Dictionary with answer, citations, and metadata
        """
        # Check if query needs calculation
        calculation_keywords = ['calculate', 'compute', 'how much', 'amount']
        needs_calculation = any(kw in query.lower() for kw in calculation_keywords)
        
        calculation_result = None
        if needs_calculation and tax_calculators:
            calculation_result = self._try_calculation(query, tax_calculators)
        
        # Retrieve relevant context
        retrieval_result = retrieval_pipeline.retrieve(query, top_k=5)
        context = retrieval_result['context']
        
        # Add calculation to context if available
        if calculation_result:
            context = f"CALCULATION RESULT:\n{calculation_result}\n\n{context}"
        
        # Create prompt
        prompt = self.create_prompt(query, context, conversation_history)
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Extract citations
        citations = self.extract_citations(response)
        
        # Build final response
        result = {
            'query': query,
            'answer': response,
            'citations': citations,
            'retrieved_chunks': retrieval_result['chunks'],
            'query_metadata': retrieval_result['query_metadata'],
            'calculation': calculation_result,
            'num_sources': len(retrieval_result['chunks'])
        }
        
        return result
    
    def _try_calculation(self, query: str, tax_calculators) -> Optional[str]:
        """
        Attempt to perform tax calculation based on query
        Returns formatted calculation result or None
        """
        query_lower = query.lower()
        
        # HRA calculation
        if 'hra' in query_lower and 'calculate' in query_lower:
            # Try to extract values from query (simplified)
            # In production, use better NER or ask user for inputs
            return "HRA calculation requires: basic salary, HRA received, rent paid, city type. Please provide these details."
        
        # 80C calculation
        if '80c' in query_lower and any(kw in query_lower for kw in ['calculate', 'deduction', 'limit']):
            result = tax_calculators.calculate_80c_deduction({})
            return f"Section 80C Deduction Limit: ₹{result.result:,.2f}\n{result.explanation}"
        
        return None
    
    def stream_response(self, prompt: str, max_new_tokens: int = 512):
        """
        Stream response token by token (for UI)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        streamer = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Decode tokens one by one
        for token_id in streamer.sequences[0]:
            token = self.tokenizer.decode([token_id], skip_special_tokens=True)
            yield token


# Simplified version using API (FREE alternative for testing)
class IncomeTaxGPT_API:
    """
    API-based version using free/cheap providers
    Options:
    1. Hugging Face Inference API (free tier available)
    2. Together AI (cheap, fast)
    3. Groq (fast, free tier)
    """
    
    def __init__(self, api_key: str = None, provider: str = "huggingface"):
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.provider = provider
        
        if provider == "huggingface":
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_key)
            self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        
        self.system_prompt = """You are IncomeTaxGPT, an AI assistant specialized in Indian Income Tax law.

CRITICAL INSTRUCTIONS:
1. Base ALL responses on the provided legal context
2. ALWAYS cite specific sections when making legal claims
3. Format citations as [Section XXX]
4. Provide plain-English explanations
5. End with disclaimer: "This is informational guidance only, not professional tax advice."
"""
    
    def answer_query(self, query: str, context: str) -> str:
        """
        Answer query using API
        """
        prompt = f"""{self.system_prompt}

LEGAL CONTEXT:
{context}

USER QUESTION: {query}

Provide a clear answer with citations."""
        
        # Call API based on provider
        if self.provider == "huggingface":
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                max_new_tokens=512,
                temperature=0.3
            )
            return response
        
        return "API response"


# Example usage
if __name__ == "__main__":
    import os
    
    # Check if you want to use local model or API
    USE_LOCAL = False  # Set to True if you have GPU
    
    if USE_LOCAL:
        print("="*70)
        print("INITIALIZING LOCAL MODEL (Requires GPU)")
        print("="*70)
        
        # Initialize local model
        # NOTE: First run will download ~13GB model
        gpt = IncomeTaxGPT(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            load_in_4bit=True
        )
        
        # Test query
        test_context = """
[Source 1]
Document: Income Tax Act, 1961
Section: 80C - Deduction in respect of life insurance premia, etc.
Year: 2024

Deduction under section 80C is available for various investments and expenses.
The maximum deduction allowed under this section is Rs. 1,50,000.

Eligible investments include:
- Life Insurance Premium
- Public Provident Fund (PPF)
- Employee Provident Fund (EPF)
- Equity Linked Savings Scheme (ELSS)
- National Savings Certificate (NSC)
- Tax-saving Fixed Deposits
- Principal repayment of Home Loan
- Tuition fees for children
---
"""
        
        test_query = "What is the maximum deduction I can claim under Section 80C?"
        
        print(f"\nQuery: {test_query}")
        print("\nGenerating response...\n")
        
        # Create prompt and generate
        prompt = gpt.create_prompt(test_query, test_context)
        response = gpt.generate_response(prompt)
        
        print("="*70)
        print("RESPONSE:")
        print("="*70)
        print(response)
        
        # Extract citations
        citations = gpt.extract_citations(response)
        print(f"\nCitations found: {citations}")
        
    else:
        print("="*70)
        print("API-BASED MODE (No local GPU needed)")
        print("="*70)
        print("\nFor testing without GPU, use one of these FREE options:")
        print("\n1. Hugging Face Inference API (free tier)")
        print("   - Sign up at: https://huggingface.co")
        print("   - Get token from: https://huggingface.co/settings/tokens")
        print("\n2. Groq API (fast & free)")
        print("   - Sign up at: https://console.groq.com")
        print("\n3. Together AI (cheap)")
        print("   - Sign up at: https://together.ai")
        print("\nSet API key: export HUGGINGFACE_API_KEY='your-key'")
        print("\nOr use OpenAI-compatible APIs for testing.")
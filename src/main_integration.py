"""
Complete IncomeTaxGPT System Integration
Main pipeline connecting all components
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/incometaxgpt.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IncomeTaxGPTSystem:
    """
    Main system class integrating all components
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the complete system
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing IncomeTaxGPT System...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.embedding_system = None
        self.retrieval_pipeline = None
        self.llm = None
        self.calculators = None
        
        self.initialized = False
    
    def _load_config(self, config_path: str) -> Dict:
        """Load system configuration"""
        default_config = {
            "data_paths": {
                "raw": "data/raw",
                "processed": "data/processed",
                "embeddings": "data/embeddings"
            },
            "models": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "base_llm": "meta-llama/Llama-2-7b-chat-hf",
                "use_local_llm": False,
                "api_provider": "huggingface"
            },
            "retrieval": {
                "top_k": 5,
                "semantic_weight": 0.7,
                "max_context_tokens": 2000
            },
            "generation": {
                "max_new_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.9
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
        else:
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("Loading components...")
            
            # Import modules with error handling
            try:
                from src.embedding_system import EmbeddingSystem
                self.embedding_system_available = True
            except Exception as e:
                logger.warning(f"Embedding system not available: {e}")
                self.embedding_system_available = False
                self.embedding_system = None
            
            try:
                from src.retrieval_pipeline import RetrievalPipeline
                self.retrieval_pipeline_available = True
            except Exception as e:
                logger.warning(f"Retrieval pipeline not available: {e}")
                self.retrieval_pipeline_available = False
                self.retrieval_pipeline = None
            
            try:
                from src.tax_calculators import TaxCalculators
                self.tax_calculators_available = True
            except Exception as e:
                logger.warning(f"Tax calculators not available: {e}")
                self.tax_calculators_available = False
                self.tax_calculators = None
            
            # Initialize embedding system (if available)
            if self.embedding_system_available:
                logger.info("Initializing embedding system...")
                self.embedding_system = EmbeddingSystem(
                    processed_dir=self.config["data_paths"]["processed"],
                    embeddings_dir=self.config["data_paths"]["embeddings"],
                    model_name=self.config["models"]["embedding_model"]
                )
                
                # Load or build indices
                try:
                    self.embedding_system.load_indices()
                    logger.info("SUCCESS: Loaded existing indices")
                except FileNotFoundError:
                    logger.warning("Indices not found. Building new indices...")
                    self.embedding_system.build_all_indices()
                    logger.info("SUCCESS: Built new indices")
                
                # Initialize retrieval pipeline
                logger.info("Initializing retrieval pipeline...")
                self.retrieval_pipeline = RetrievalPipeline(self.embedding_system)
            else:
                logger.info("Skipping embedding system initialization")
                self.embedding_system = None
                self.retrieval_pipeline = None
            
            # Initialize calculators (if available)
            if self.tax_calculators_available:
                logger.info("Initializing tax calculators...")
                self.calculators = TaxCalculators()
            else:
                logger.info("Tax calculators not available")
                self.calculators = None
            
            # Initialize LLM
            logger.info("Initializing LLM...")
            try:
                if self.config["models"]["use_local_llm"]:
                    from src.llm_integration import IncomeTaxGPT
                    self.llm = IncomeTaxGPT(
                        model_name=self.config["models"]["base_llm"],
                        load_in_4bit=True
                    )
                    logger.info("SUCCESS: Local LLM initialized successfully")
                else:
                    from src.llm_integration import IncomeTaxGPT_API
                    self.llm = IncomeTaxGPT_API(
                        provider=self.config["models"]["api_provider"]
                    )
                    logger.info("SUCCESS: API LLM initialized successfully")
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}")
                logger.info("Using fallback response system")
                self.llm = None
            
            self.initialized = True
            logger.info("SUCCESS: System initialized successfully!")
            
        except Exception as e:
            logger.error(f"ERROR: Initialization failed: {e}")
            raise
    
    def query(self, user_query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Process user query through complete pipeline
        
        Args:
            user_query: User's question
            conversation_history: Previous conversation messages
        
        Returns:
            Complete response with answer, citations, and metadata
        """
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        logger.info(f"Processing query: {user_query}")
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant context (if available)
            if self.retrieval_pipeline:
                logger.debug("Retrieving context...")
                retrieval_result = self.retrieval_pipeline.retrieve(
                    user_query,
                    top_k=self.config["retrieval"]["top_k"]
                )
            else:
                logger.warning("Retrieval pipeline not available, using fallback")
                retrieval_result = {
                    'chunks': [],
                    'scores': [],
                    'context': 'No retrieval context available',
                    'metadata': {'method': 'fallback'},
                    'query_metadata': {'query_type': 'fallback'}
                }
            
            # Step 2: Check if calculation is needed
            calculation_result = self._handle_calculation(user_query)
            
            # Step 3: Prepare context
            context = retrieval_result['context']
            if calculation_result:
                context = f"CALCULATION:\n{calculation_result}\n\n{context}"
            
            # Step 4: Generate response
            logger.debug("Generating response...")
            if self.llm:
                try:
                    if self.config["models"]["use_local_llm"]:
                        # Full local LLM
                        response = self.llm.answer_query(
                            user_query,
                            self.retrieval_pipeline,
                            self.calculators,
                            conversation_history
                        )
                    else:
                        # API-based LLM
                        answer = self.llm.answer_query(user_query, context)
                        response = {
                            'query': user_query,
                            'answer': answer,
                            'retrieved_chunks': retrieval_result['chunks'],
                            'calculation': calculation_result
                        }
                except Exception as e:
                    logger.warning(f"LLM query failed: {e}")
                    logger.info("Using fallback response due to LLM error")
                    answer = self._generate_fallback_response(user_query, context, calculation_result)
                    response = {
                        'query': user_query,
                        'answer': answer,
                        'retrieved_chunks': retrieval_result['chunks'],
                        'calculation': calculation_result
                    }
            else:
                # Fallback response when LLM is not available
                logger.warning("LLM not available, using fallback response")
                answer = self._generate_fallback_response(user_query, context, calculation_result)
                response = {
                    'query': user_query,
                    'answer': answer,
                    'retrieved_chunks': retrieval_result['chunks'],
                    'calculation': calculation_result
                }
            
            # Step 5: Add metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            response['metadata'] = {
                'processing_time': processing_time,
                'num_sources': len(retrieval_result['chunks']),
                'query_type': retrieval_result['query_metadata']['query_type'],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"SUCCESS: Query processed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing query: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return {
                'query': user_query,
                'answer': f"Error processing query: {str(e)}",
                'error': True,
                'metadata': {
                    'processing_time': processing_time,
                    'num_sources': 0,
                    'query_type': 'error',
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def _handle_calculation(self, query: str) -> Optional[str]:
        """Detect and handle calculation requests"""
        query_lower = query.lower()
        
        # HRA calculation pattern
        if 'hra' in query_lower and 'calculate' in query_lower:
            return "For HRA calculation, please provide: basic salary, HRA received, rent paid, and city type."
        
        # 80C information
        if '80c' in query_lower:
            if self.calculators:
                result = self.calculators.calculate_80c_deduction({})
                return f"Section 80C maximum deduction limit: ₹{result.result:,.2f}"
            else:
                return "Section 80C maximum deduction limit: ₹1,50,000 (tax calculators not available)"
        
        return None
    
    def _generate_fallback_response(self, user_query: str, context: str, calculation_result: str = None) -> str:
        """Generate a basic fallback response when LLM is not available"""
        query_lower = user_query.lower()
        
        # Basic tax information responses
        if '80c' in query_lower or '80 c' in query_lower:
            return """Based on your query about Section 80C:

Section 80C allows deductions up to ₹1,50,000 for various investments and expenses including:
- Employee Provident Fund (EPF)
- Public Provident Fund (PPF)
- Equity Linked Saving Schemes (ELSS)
- Life Insurance Premiums
- Principal amount of home loan EMI
- Tuition fees for children
- National Savings Certificate (NSC)
- Tax Saving Fixed Deposits

The maximum deduction limit under Section 80C is ₹1,50,000 per financial year.

Note: This is a basic response. For detailed calculations and personalized advice, please consult a tax professional."""
        
        elif 'hra' in query_lower or 'house rent allowance' in query_lower:
            return """House Rent Allowance (HRA) is exempt from tax subject to certain conditions:

1. Actual HRA received
2. Actual rent paid minus 10% of salary
3. 50% of salary (for metro cities) or 40% (for non-metro cities)

The least of these three amounts is exempt from tax.

Metro cities include Delhi, Mumbai, Chennai, and Kolkata.

Note: This is a basic response. For detailed calculations, please provide your specific salary and rent details."""
        
        elif 'tax' in query_lower and 'calculate' in query_lower:
            return """For tax calculations, please provide:
- Your total annual income
- Age (for determining tax slabs)
- Any deductions you're claiming
- Type of income (salary, business, etc.)

Basic tax slabs (FY 2023-24):
- Up to ₹2,50,000: No tax
- ₹2,50,001 - ₹5,00,000: 5%
- ₹5,00,001 - ₹10,00,000: 20%
- Above ₹10,00,000: 30%

Note: This is a basic response. For accurate calculations, please consult a tax professional."""
        
        else:
            return f"""Thank you for your question: "{user_query}"

I'm currently operating in fallback mode due to technical limitations. While I can provide basic information about Indian income tax laws, for detailed and personalized tax advice, I recommend:

1. Consulting a qualified tax professional
2. Referring to the official Income Tax Department website
3. Using official tax calculation tools

For basic queries about:
- Section 80C deductions
- HRA calculations  
- Tax slabs
- Basic tax information

I can provide general guidance. Please rephrase your question with specific details for better assistance.

Note: This response is generated without AI assistance and should not be considered as professional tax advice."""
    
    def process_batch_queries(self, queries: List[str]) -> List[Dict]:
        """Process multiple queries in batch"""
        logger.info(f"Processing batch of {len(queries)} queries...")
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            result = self.query(query)
            results.append(result)
        
        return results
    
    def evaluate_on_test_set(self, test_file: str) -> Dict:
        """
        Evaluate system on test dataset
        
        Args:
            test_file: Path to JSON file with test queries and expected answers
        
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating on test set: {test_file}")
        
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        results = {
            'total': len(test_data),
            'correct': 0,
            'partially_correct': 0,
            'incorrect': 0,
            'details': []
        }
        
        for item in test_data:
            query = item.get('query', '')
            expected_keywords = [k.lower() for k in item.get('expected_keywords', [])]
            
            response = self.query(query)
            answer_text = response.get('answer', '')
            answer_lc = answer_text.lower()
            
            # Keyword-based evaluation: pass if any expected keyword is present
            if expected_keywords:
                hits = [kw for kw in expected_keywords if kw in answer_lc]
                hit_ratio = len(hits) / max(1, len(expected_keywords))
            else:
                hits = []
                hit_ratio = 0.0
            
            if hit_ratio >= 0.6 or len(hits) >= 2:
                results['correct'] += 1
                correctness = 'correct'
            elif hit_ratio > 0:
                results['partially_correct'] += 1
                correctness = 'partially_correct'
            else:
                results['incorrect'] += 1
                correctness = 'incorrect'
            
            results['details'].append({
                'query': query,
                'expected_keywords': expected_keywords,
                'hits': hits,
                'got': answer_text,
                'correctness': correctness,
                'hit_ratio': hit_ratio
            })
        
        logger.info(f"Evaluation complete: {results['correct']}/{results['total']} correct")
        return results
    
    def export_knowledge_base_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        stats = self.embedding_system.chunks
        
        return {
            'total_chunks': len(stats),
            'total_documents': len(set(c['doc_id'] for c in stats)),
            'sections_covered': len(set(c['section_number'] for c in stats)),
            'year_range': f"{min(c['year'] for c in stats)} - {max(c['year'] for c in stats)}",
            'avg_chunk_length': sum(len(c['text']) for c in stats) / len(stats)
        }

def setup_project_structure():
    """Create necessary directories for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'data/embeddings',
        'models/base',
        'models/fine-tuned',
        'logs',
        'outputs',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Project structure created")

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IncomeTaxGPT System")
    parser.add_argument('--setup', action='store_true', help='Setup project structure')
    parser.add_argument('--init', action='store_true', help='Initialize system')
    parser.add_argument('--query', type=str, help='Process a single query')
    parser.add_argument('--batch', type=str, help='Process batch queries from file')
    parser.add_argument('--evaluate', type=str, help='Evaluate on test set')
    parser.add_argument('--stats', action='store_true', help='Show knowledge base stats')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_project_structure()
    
    elif args.init:
        system = IncomeTaxGPTSystem()
        system.initialize()
        print("✅ System initialized and ready!")
    
    elif args.query:
        system = IncomeTaxGPTSystem()
        system.initialize()
        response = system.query(args.query)
        print("\n" + "="*70)
        print("QUERY:", response['query'])
        print("="*70)
        print(response['answer'])
        print("\n" + "-"*70)
        print(f"Sources: {response['metadata']['num_sources']}")
        print(f"Processing time: {response['metadata']['processing_time']:.2f}s")
    
    elif args.batch:
        system = IncomeTaxGPTSystem()
        system.initialize()
        with open(args.batch, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        results = system.process_batch_queries(queries)
        # Save results
        with open('outputs/batch_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Processed {len(results)} queries. Results saved to outputs/batch_results.json")
    
    elif args.evaluate:
        system = IncomeTaxGPTSystem()
        system.initialize()
        eval_results = system.evaluate_on_test_set(args.evaluate)
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"Total queries: {eval_results['total']}")
        print(f"Correct: {eval_results['correct']}")
        print(f"Partially correct: {eval_results['partially_correct']}")
        print(f"Incorrect: {eval_results['incorrect']}")
        print(f"Accuracy: {eval_results['correct']/eval_results['total']*100:.1f}%")
    
    elif args.stats:
        system = IncomeTaxGPTSystem()
        system.initialize()
        stats = system.export_knowledge_base_stats()
        print("\n" + "="*70)
        print("KNOWLEDGE BASE STATISTICS")
        print("="*70)
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    else:
        parser.print_help()
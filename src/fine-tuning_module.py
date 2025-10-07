"""
Fine-tuning Module for IncomeTaxGPT
Implements Parameter-Efficient Fine-Tuning using LoRA/QLoRA
"""

import os
from typing import List, Dict
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer

class IncomeTaxGPTTrainer:
    def __init__(self, 
                 base_model: str = "meta-llama/Llama-2-7b-chat-hf",
                 output_dir: str = "models/fine-tuned"):
        """
        Initialize trainer for fine-tuning
        
        Args:
            base_model: Base model to fine-tune
            output_dir: Directory to save fine-tuned model
        """
        self.base_model_name = base_model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Quantization config for QLoRA
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # LoRA config
        self.lora_config = LoraConfig(
            r=16,                # Rank
            lora_alpha=32,       # Alpha parameter
            target_modules=[    # Which layers to adapt
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    def create_training_dataset(self, 
                               training_data: List[Dict],
                               tokenizer) -> Dataset:
        """
        Create dataset from training examples
        
        Args:
            training_data: List of dicts with 'instruction', 'context', 'response'
            tokenizer: Tokenizer instance
        
        Returns:
            HuggingFace Dataset
        """
        formatted_data = []
        
        for example in training_data:
            # Format in chat template
            messages = [
                {
                    "role": "system",
                    "content": "You are IncomeTaxGPT, an expert on Indian Income Tax law."
                },
                {
                    "role": "user",
                    "content": f"{example['instruction']}\n\nContext: {example.get('context', '')}"
                },
                {
                    "role": "assistant",
                    "content": example['response']
                }
            ]
            
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            formatted_data.append({"text": text})
        
        return Dataset.from_list(formatted_data)
    
    def generate_synthetic_dataset(self, 
                                   retrieval_pipeline,
                                   num_samples: int = 100) -> List[Dict]:
        """
        Generate synthetic training data from tax documents
        
        This creates Q&A pairs from the legal corpus
        """
        training_data = []
        
        # Template queries for different tax scenarios
        query_templates = [
            "What is the deduction limit under Section {section}?",
            "Explain Section {section} of the Income Tax Act",
            "How to calculate {topic}?",
            "What are the conditions for {topic}?",
            "Who is eligible for {topic} benefits?",
        ]
        
        # Key sections to cover
        important_sections = [
            "80C", "80D", "80G", "10(13A)", "24", "54", "87A"
        ]
        
        topics = [
            "HRA exemption",
            "home loan interest",
            "capital gains",
            "tax rebate",
            "standard deduction"
        ]
        
        print(f"Generating {num_samples} synthetic training examples...")
        
        # Generate examples
        for i in range(num_samples):
            if i % 2 == 0:
                # Section-based query
                section = important_sections[i % len(important_sections)]
                query = query_templates[i % len(query_templates)].format(
                    section=section,
                    topic=topics[i % len(topics)]
                )
            else:
                # Topic-based query
                topic = topics[i % len(topics)]
                query = f"Explain how to calculate {topic} for tax purposes"
            
            # Retrieve relevant context
            result = retrieval_pipeline.retrieve(query, top_k=2)
            
            if not result['chunks']:
                continue
            
            # Create context from chunks
            context = "\n".join([
                f"Section {c['section_number']}: {c['text'][:300]}"
                for c in result['chunks'][:2]
            ])
            
            # Generate response (this would ideally use GPT-4 or manual annotation)
            # For now, create a template response
            response = f"""Based on the Income Tax Act:

{result['chunks'][0]['text'][:400]}

[Section {result['chunks'][0]['section_number']}]

Disclaimer: This is informational guidance only, not professional tax advice."""
            
            training_data.append({
                'instruction': query,
                'context': context,
                'response': response
            })
        
        return training_data
    
    def train(self, 
             training_data: List[Dict],
             epochs: int = 3,
             batch_size: int = 4,
             learning_rate: float = 2e-4):
        """
        Fine-tune the model
        
        Args:
            training_data: List of training examples
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        print(f"\n{'='*70}")
        print("STARTING FINE-TUNING")
        print(f"{'='*70}\n")
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load base model
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        
        # Add LoRA adapters
        print("Adding LoRA adapters...")
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        
        # Create dataset
        print("Preparing dataset...")
        dataset = self.create_training_dataset(training_data, tokenizer)
        print(f"Dataset size: {len(dataset)} examples")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            learning_rate=learning_rate,
            warmup_steps=50,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            report_to="none"
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=1024,
            dataset_text_field="text"
        )
        
        # Train
        print("\nStarting training...\n")
        trainer.train()
        
        # Save final model
        print("\nSaving fine-tuned model...")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        print(f"\n✅ Training complete! Model saved to: {self.output_dir}")
    
    def save_training_data(self, 
                          training_data: List[Dict],
                          filename: str = "training_data.json"):
        """Save training data for future use"""
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        print(f"Training data saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # This is a complete fine-tuning example
    # Adjust based on your available resources
    
    print("="*70)
    print("INCOMETAX-GPT FINE-TUNING")
    print("="*70)
    
    # Option 1: Manual training data
    manual_training_data = [
        {
            'instruction': 'What is the maximum deduction limit under Section 80C?',
            'context': 'Section 80C provides deduction for investments in specified schemes.',
            'response': '''Under Section 80C of the Income Tax Act, 1961, the maximum deduction allowed is Rs. 1,50,000 per financial year.

This deduction is available for investments in:
- Public Provident Fund (PPF)
- Employee Provident Fund (EPF)
- Equity Linked Savings Scheme (ELSS)
- Life Insurance Premium
- National Savings Certificate (NSC)
- Tax-saving Fixed Deposits
- Principal repayment of Home Loan
- Tuition fees for children

[Section 80C, Income Tax Act, 1961]

Disclaimer: This is informational guidance only, not professional tax advice.'''
        },
        {
            'instruction': 'How is HRA exemption calculated?',
            'context': 'Section 10(13A) provides exemption for House Rent Allowance.',
            'response': '''HRA exemption under Section 10(13A) is calculated as the minimum of:

1. Actual HRA received
2. 50% of basic salary (for metro cities) or 40% (for non-metro)
3. Rent paid minus 10% of basic salary

Only the minimum of these three amounts is exempt from tax.

[Section 10(13A), Rule 2A of Income Tax Rules]

Disclaimer: This is informational guidance only, not professional tax advice.'''
        }
    ]
    
    # Initialize trainer
    trainer = IncomeTaxGPTTrainer(
        base_model="meta-llama/Llama-2-7b-chat-hf",
        output_dir="models/incometax-gpt"
    )
    
    # Save manual training data
    trainer.save_training_data(manual_training_data)
    
    print("\n📝 To generate synthetic data, integrate with retrieval pipeline:")
    print("""
    from retrieval_pipeline import RetrievalPipeline
    from embedding_system import EmbeddingSystem
    
    embedding_sys = EmbeddingSystem()
    embedding_sys.load_indices()
    retrieval = RetrievalPipeline(embedding_sys)
    
    synthetic_data = trainer.generate_synthetic_dataset(retrieval, num_samples=500)
    trainer.save_training_data(synthetic_data, 'synthetic_training_data.json')
    """)
    
    print("\n🚀 To start training:")
    print("    trainer.train(manual_training_data, epochs=3, batch_size=4)")
    print("\n⚠️  NOTE: Training requires ~12GB GPU RAM with 4-bit quantization")
    print("   Use Google Colab Pro or cloud GPU if needed")
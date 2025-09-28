#!/usr/bin/env python3
"""
Interactive LoRA QA Model Demo
Simple script to use the trained LoRA adapter for question answering

Usage:
    python use_lora_model.py

Then type your questions and get AI-generated answers!
"""

import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRAQABot:
    def __init__(self, model_path="./toy_qa_lora_model"):
        """Initialize the LoRA QA bot"""
        self.model_path = model_path
        self.base_model_name = "distilgpt2"
        self.tokenizer = None
        self.model = None

        print("🤖 Loading LoRA QA Model...")
        self.load_model()
        print("✅ Model loaded successfully!")

    def load_model(self):
        """Load the base model and LoRA adapter"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name, torch_dtype=torch.float32, device_map=None
            )

            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure you have trained the model first by running: python lora_qa_demo.py")
            sys.exit(1)

    def generate_answer(self, question, max_tokens=100, temperature=0.7):
        """Generate an answer for the given question"""

        # Format the question as the model expects
        if not question.strip().startswith("Question:"):
            prompt = f"Question: {question.strip()}\nAnswer:"
        else:
            prompt = f"{question.strip()}\nAnswer:"

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9,
                top_k=50,
            )

        # Decode and extract the generated part
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response[len(prompt) :].strip()

        # Clean up the answer (remove extra repetitions if any)
        if "\n" in answer:
            answer = answer.split("\n")[0].strip()

        return answer


def print_welcome():
    """Print welcome message"""
    print("=" * 60)
    print("🧠 LoRA QA Bot - Interactive Question Answering")
    print("=" * 60)
    print("Ask me questions about:")
    print("  • Programming (Python, ML, AI)")
    print("  • Technology concepts")
    print("  • Data science topics")
    print("  • And more!")
    print()
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit', 'exit', or 'q' to exit")
    print("  - Type 'help' for this message")
    print("=" * 60)


def main():
    """Main interactive loop"""

    # Check if model exists
    model_path = "./toy_qa_lora_model"
    if not os.path.exists(model_path):
        print("❌ Model not found!")
        print("Please train the model first by running:")
        print("   python lora_qa_demo.py")
        return

    # Initialize the bot
    try:
        bot = LoRAQABot(model_path)
    except Exception as e:
        print(f"❌ Failed to initialize bot: {e}")
        return

    # Print welcome message
    print_welcome()

    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("\n💭 Your Question: ").strip()

            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye! Thanks for using LoRA QA Bot!")
                break
            elif user_input.lower() == "help":
                print_welcome()
                continue
            elif not user_input:
                print("❓ Please enter a question.")
                continue

            # Generate answer
            print("\n🤖 Thinking...")
            answer = bot.generate_answer(user_input)

            # Display result
            print(f"\n📝 Answer: {answer}")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using LoRA QA Bot!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()

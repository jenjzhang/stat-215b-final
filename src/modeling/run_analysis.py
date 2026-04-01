"""
Entry point: run multilevel model + multiple testing for both models.
"""
from modeling import multilevel, testing

if __name__ == "__main__":
    for model in ["gpt4o", "llama"]:
        multilevel.run(model)
        testing.run(model)

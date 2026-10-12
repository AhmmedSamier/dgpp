"""A fixed, varied-prompt companion to the repeated-counting workload."""
from serve_admission_bench import Bench

PROMPTS = (
    'Write a Python function that merges two sorted lists.\n```python\ndef merge(a, b):\n',
    'Explain how rain forms, step by step, for a curious ten-year-old.\nRain forms',
    'Write a JSON array of fictional cities, each with a name and population.\n[\n',
    'Translate the following English words into French and explain each: tree, river, mountain, sunlight.\n',
    'Continue this story: The old lighthouse keeper found a sealed letter under the door.\n',
    'Explain the differences between a stack and a queue, with examples.\nA stack',
    'List the squares of the positive integers, showing the calculation for each.\n1 * 1 = 1\n',
    'Write a detailed recipe for vegetable soup using carrots, onions, potatoes and beans.\nIngredients:\n',
)


class ArrivalBench(Bench):
    def stream(self, name, prompt, tokens, *args, **kwargs):
        if getattr(self.args, 'arrival_pattern', 'counting') == 'varied' and name.startswith('arrival-'):
            prompt = PROMPTS[int(name.split('-')[1]) % len(PROMPTS)]
        return super().stream(name, prompt, tokens, *args, **kwargs)

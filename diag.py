import inspect
from openxai.metrics import eval_pred_faithfulness

sig = inspect.signature(eval_pred_faithfulness)
print("eval_pred_faithfulness signature:", sig)
for name, param in sig.parameters.items():
    print(f"  {name}: {param.kind} (default={param.default})")

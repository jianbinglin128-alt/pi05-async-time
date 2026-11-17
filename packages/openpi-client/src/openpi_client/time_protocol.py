"""Fixed action protocol for the Future timing runtime.

These are action counts, independent of the model's flow denoising iterations.
Episodes may terminate before all ten actions in the final chunk are consumed.
"""

PREDICTED_STEPS = 10
EXECUTED_STEPS = PREDICTED_STEPS
CONTEXT_HORIZON = 9

# Kernel pretraining

Run multiple observation-only Linux kernel workers and train a small next-block
model with bounded batching. Real three-worker MPS learning is recorded in the
[setup and results](../../docs/kernel-pretraining.md).

Build the benign guest using [kernel setup](../../docs/kernel-examples.md).
AWS hosts and connections are operator-managed; CUDA and DDP remain unverified.

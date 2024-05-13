# OffRoPE
Using randomized offsets for better context length generalization of RoPE embeddings. 

Many works have sought to create LMs which can be extended to larger contexts during inference, a task which positional embeddings make more difficult. 
New embeddings such as ALiBi have been introduced to combat this issue, however, they often create decreased performance within the same context length.
Because the rotations are the same between each set of positions in the context window there is little reason why RoPE should experience such a precipitous performance drop during inference. 
Inspired by BERT-like methodologies, we propose using random offsets to the RoPE embeddings in order to force the model to learn about the rotations being performed such that the context window can be better expanded during inference.
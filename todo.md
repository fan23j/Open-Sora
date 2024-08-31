training
1. get comprable unconditional results
2. integrate multi-bbx adapter
3. integrate statvu condition adapter
4. add back text embeds?

optimizations
1. apex
2. enable flash attention 
3. max out bs, can we beat 128 effective frames?
4. find some way to reduce # params, i.e., use a lighter weight bb

model
1. switch STDiT backbone v2 -> v3

data
1. NBA 15-16 -> 22-23
2. better bbxs?

eval
1. dev quantitative eval pipeline

logging
1. add training sample viz for all conditions
2. remove compile statements / extra verbosity
3. format the save sample progress bar
4. ensure progress bar accurately reflects batch size
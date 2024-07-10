# Training Data Analysis

---

## Groups

##### divide by n
18,66

##### group by same color:
- connected: 22, 85,95,97,98, 99,3,4,6,7,8,9,12,15,16,23,26,27,29,30,35,36,40,41,48,50,51,52,56,57,61,62,63,65,69,70,72,73,77,78,80,81,82,83,84,
86,87,89,90,34,37,45,55,59,67
- with clear border: 5,10,13,20,24,25,28,32,39,46,54,58,60,64,71,74,79, 93
- with thick-band border: 92
- plain: 0,1,2,18,44,76,91,11,14, 42,47,49,31

##### connection:
- 14,17,19,21,33,38,43,53,68,75,76,88,96


#### inter-ig relations

###### ig1 split ig2
Rule: if ig1 split ig2 to n sections, return these sections in a list
GroupList():-split(g)

target: group
rule feature: one group split the matrix into several sections 
action: split  
input: matrix
output: groups

## Actions

##### mask

##### change color
87,89

##### rotation
86

##### complete missing part
73,75,80,81,82,88,95,97

##### connect tiles
44,91

##### duplicate
- shift:0
- reflection symmetry: 82,82
- 

##### Crop
90

##### count
37,78,67

##### scale
100
##### with indicators
0,74,94
##### divide by x 
- with remainder
  - 2
- without remainder



---
## strategy
#### strategy for action type deciding
``` 
for example in examples:
    type, type_conf <-get_most_possible_type(example)
choose the type with highest confidence
```


---

## Clauses

######
Let `G0-G9` represents the groups in output.
Let `A0-A9` represents the groups in input.

Given output groups `G0,G1`, input groups `A0,A1,A2`.
``` 
(C_0) target(Input,Output):-inv_p1(G0),inv_p2(G1),in(G0,Output),in(G1,Output).

inv_p1(G0):-in(A0,Input),in(G0,Output).
- extension 01: inv_p1(G0):-in(A0,Input),in(G0,Output),scale(A0,G0,3).

atoms:
in(a0,input),
in(g0,output),
scale(a0,g0,param)
inv_p0(g0)
inv_p0(g1)
```


#### ChatGPT Prompt

#### Grouping by sections
python code: given a matrix which items are only integers in range of 0-9. 
Slice the matrix to multiple blocks considering both its item value (if they are same) and the positions (if they are connected),
return the sliced matrices in a list.


#### Checking for rule features:
python code: given a matrix, 
identify if there is a path that split the matrix to multiple disconnected rectangular regions


#### Pipeline

Ask LLM to identify
- rectangular regions
- same color regions
- connected regions
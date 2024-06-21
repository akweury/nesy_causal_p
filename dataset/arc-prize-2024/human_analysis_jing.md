# Training Data Analysis

---

## Groups

##### no obvious groups: 
18,19,31,34,37,42,45,47,49,55,59,67,96


##### divide by n
66

##### same color:
- connected shape:22, 85,95,97,98, 99,
- plain: 0,1,2,18,44,76,91

##### regions with borders:
- with clear border: 5,10,13,20,24,25,28,32,39,46,54,58,60,64,71,74,79, 93
- with thick-band border: 92

##### connected:
- same color:
3,4,6,7,8,9,12,15,16,23,26,27,29,30,35,36,40,41,48,50,51,52,56,57,61,62,63,65,69,70,72,73,77,78,80,81,82,83,84,
86,87,89,90

- different color: 11,14,17,21,33,38,43,53,68,75,76,88




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
78

##### scale
100
##### with indicators
0,74,94


---
## strategy
#### strategy for action type deciding
``` 
for example in examples:
    type, type_conf <-get_most_possible_type(example)
choose the type with highest confidence
```



import numpy as np

# Dataset 
X = [[10,10,9],[13,12,13],[5,5,4],[8,7,7]] 

# Encoder function f
def encode1(inp):  
  e1 = []
  for i in inp:
    e1.append(i[0]-i[1]+i[2])
  return e1

# Encoder function f_tild
def encode2(inp):
  e2 = []
  for i in inp:
    e2.append((i[0]+i[1]+i[2])/3)
  return e2

def decoder(inp): #Decoder function
  d = []
  for i in range(len(inp)):
    d.append([])
    d[i] = d[i] + [inp[i],inp[i],inp[i]]
  return d

def loss(X,inp): # Loss Function
  l = 0
  for i in range(len(X)):
    t1 = np.array(X[i]) - np.array(inp[i])
    l = l + (t1[0]**2 + t1[1]**2 + t1[2]**2)
  return l/len(X)

# Calculating Loss for the first pair of enoder-decoder
f = encode1(X)
g = decoder(f)
print("Loss for the first pair: ", loss(X,g))

# Calculating Loss for the second pair of enoder-decoder
f_t = encode2(X)
g_t = decoder(f_t)
print("Loss for the second pair: ", loss(X,g_t))
